"""
Matrix-free centrality computation via Triton kernel.

Computes Out[b,h,i] = sum_j softmax(Q[b,h,i,:] @ K[b,h,j,:].T / sqrt(D)) * v[b,j]
without materializing the (S,S) attention matrix.

Uses Flash Attention-style online softmax for numerical stability.

Key Features:
- O(N) memory complexity (vs O(N^2) for explicit attention)
- Causal masking (j <= i) built-in
- Float32 accumulators for numerical stability with bfloat16 inputs
- Autotuned for H100 (132 SMs, large register file)
"""

import math
from typing import Optional

import torch
import triton
import triton.language as tl


def _get_autotune_configs():
    """
    Generate autotune configurations optimized for H100.

    H100 specifications:
    - 132 SMs (vs 108 on A100)
    - Larger register file per SM
    - Higher memory bandwidth (3.35 TB/s HBM3)

    Block size considerations:
    - Larger blocks = better memory coalescing, fewer kernel launches
    - Smaller blocks = better occupancy for short sequences
    - num_warps: 4-8 warps typically optimal for compute-bound kernels
    - num_stages: 2-4 for pipelining memory loads
    """
    return [
        # Small sequences (S < 256): prioritize occupancy
        triton.Config(
            {'BLOCK_M': 32, 'BLOCK_N': 32},
            num_warps=4,
            num_stages=3,
        ),
        # Medium sequences: balanced
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=4,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=2,
        ),
        # Large sequences: maximize throughput with larger blocks
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 64},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 64, 'BLOCK_N': 128},
            num_warps=8,
            num_stages=3,
        ),
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=8,
            num_stages=2,
        ),
        # Very large sequences: maximum block sizes for H100
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128},
            num_warps=16,
            num_stages=2,
        ),
    ]


@triton.autotune(
    configs=_get_autotune_configs(),
    key=['S', 'D'],  # Autotune based on sequence length and head dim
)
@triton.jit
def _centrality_flash_kernel(
    # Pointers to matrices
    Q_ptr, K_ptr, v_ptr, Out_ptr,
    # Head weighting (ITI-inspired Truth-Head suppression)
    head_weights_ptr,  # (L*H,) per-layer-head weights or dummy ptr
    USE_HEAD_WEIGHTS: tl.constexpr,  # Whether to apply head weighting
    # Strides for Q: (B, H, S, D)
    stride_qb, stride_qh, stride_qs, stride_qd,
    # Strides for K: (B, H, S, D)
    stride_kb, stride_kh, stride_ks, stride_kd,
    # Strides for v: (B, S)
    stride_vb, stride_vs,
    # Strides for Out: (B, H, S)
    stride_ob, stride_oh, stride_os,
    # Dimensions
    S: tl.constexpr,  # Sequence length
    D: tl.constexpr,  # Head dimension
    # Block sizes
    BLOCK_M: tl.constexpr,  # Block size for query positions
    BLOCK_N: tl.constexpr,  # Block size for key positions
    BLOCK_D: tl.constexpr,  # Block size for head dimension
):
    """
    Triton kernel for matrix-free attention-weighted centrality.

    For each query position i, computes:
        out[i] = sum_{j <= i} softmax(Q[i] @ K[j].T / sqrt(D)) * v[j] * w_h

    where w_h is the per-head weight (if USE_HEAD_WEIGHTS=True).

    Uses online softmax (Flash Attention style) to avoid O(S^2) memory.
    Accumulators use float32 for numerical stability even with bfloat16 inputs.

    Head weighting (ITI-inspired):
    - w_h ∈ [0, 1] down-weights Induction Heads that perpetuate misconceptions
    - Calibrated offline using TruthfulQA samples
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch index
    pid_h = tl.program_id(1)  # Head index
    pid_m = tl.program_id(2)  # Query block index

    # Compute query positions for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Mask for valid query positions
    mask_m = offs_m < S

    # Base pointers for this batch and head
    Q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    K_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb

    # Load query block: (BLOCK_M, BLOCK_D)
    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)

    # Scale factor for attention scores
    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    # Online softmax accumulators (float32 for numerical stability)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)  # Running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum of exp
    acc_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running weighted sum

    # Maximum key position for causal masking (per query in block)
    max_j = offs_m + 1  # Each query i can attend to j in [0, i]

    # Iterate over key blocks
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        # Load key block: (BLOCK_N, BLOCK_D)
        k_ptrs = K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)

        # Load v values for this key block: (BLOCK_N,)
        v_ptrs = v_base + offs_n * stride_vs
        v_block = tl.load(v_ptrs, mask=mask_n, other=0.0)

        # Compute attention scores: (BLOCK_M, BLOCK_N)
        # scores[i, j] = Q[i] @ K[j].T / sqrt(D)
        scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        # Apply causal mask: j <= i
        # Each query position i can only attend to key positions j <= i
        causal_mask = offs_n[None, :] <= offs_m[:, None]
        scores = tl.where(causal_mask, scores, float('-inf'))

        # Also mask invalid key positions
        scores = tl.where(mask_n[None, :], scores, float('-inf'))

        # Online softmax update
        # 1. Find new max per row
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        # 2. Compute rescaling factors
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new[:, None])

        # 3. Update running sum of exponentials
        l_new = l_i * alpha + tl.sum(beta, axis=1)

        # 4. Update running weighted sum
        # acc_new = acc_old * alpha + sum(beta * v * w_h)
        beta_v = beta * v_block[None, :]

        # Apply head weighting if enabled (ITI-style Truth-Head suppression)
        if USE_HEAD_WEIGHTS:
            # Load per-head weight (pid_h indexes into flattened L*H tensor)
            head_weight = tl.load(head_weights_ptr + pid_h)
            beta_v = beta_v * head_weight

        acc_new = acc_i * alpha + tl.sum(beta_v, axis=1)

        # 5. Update state
        m_i = m_new
        l_i = l_new
        acc_i = acc_new

    # Finalize: divide by normalizer
    out = acc_i / (l_i + 1e-10)

    # Store output: Out[b, h, i]
    out_ptrs = Out_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_m * stride_os
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m)


def centrality_flash_fwd(
    Q: torch.Tensor,  # (B, H, S, D)
    K: torch.Tensor,  # (B, H, S, D)
    v: torch.Tensor,  # (B, S)
    head_weights: Optional[torch.Tensor] = None,  # (L*H,) per-layer-head weights
) -> torch.Tensor:    # (B, H, S)
    """
    Forward pass: compute attention-weighted centrality per head.

    Computes Out[b,h,i] = sum_{j<=i} softmax(Q[b,h,i,:] @ K[b,h,j,:].T / sqrt(D)) * v[b,j] * w_h
    without materializing the (S,S) attention matrix.

    Args:
        Q: Query vectors (B, H, S, D) where H = L*heads_per_layer
        K: Key vectors (B, H, S, D)
        v: Current centrality vector (B, S)
        head_weights: Optional (L*H,) per-layer-head weights in [0,1].
            If provided, each head's contribution is scaled by its weight.
            Used for ITI-inspired Truth-Head weighting to suppress Induction Heads.

    Returns:
        out: Attention-weighted centrality per head (B, H, S)

    Notes:
        - Uses Flash Attention-style online softmax for O(N) memory
        - Accumulators use float32 for numerical stability
        - Causal masking is built-in (j <= i)
        - Autotuned for H100 with multiple block size configurations
        - Head weighting incurs zero overhead when disabled (compile-time conditional)
    """
    # Validate inputs
    B, H, S, D = Q.shape
    assert K.shape == (B, H, S, D), f"K shape {K.shape} != expected {(B, H, S, D)}"
    assert v.shape == (B, S), f"v shape {v.shape} != expected {(B, S)}"
    assert Q.is_cuda, "Q must be on CUDA device"
    assert K.is_cuda, "K must be on CUDA device"
    assert v.is_cuda, "v must be on CUDA device"

    # Head weights setup with dummy pointer safety
    # CRITICAL: Triton segfaults on null pointer even with conditional guard
    # because the compiler may speculate the load. Pass a valid dummy pointer.
    use_head_weights = head_weights is not None
    if use_head_weights:
        assert head_weights.shape == (H,), f"head_weights shape {head_weights.shape} != expected {(H,)}"
        assert head_weights.is_cuda, "head_weights must be on CUDA device"
        head_weights_ptr = head_weights
    else:
        # Use v as safe dummy pointer - never actually loaded due to compile-time conditional
        head_weights_ptr = v

    # Allocate output
    out = torch.empty((B, H, S), device=Q.device, dtype=Q.dtype)

    # BLOCK_D must be power of 2 for efficient memory access
    BLOCK_D = triton.next_power_of_2(D)

    # Grid function for autotuned kernel
    # The grid depends on BLOCK_M which is determined by autotune
    def grid(meta):
        return (B, H, triton.cdiv(S, meta['BLOCK_M']))

    # Launch autotuned kernel
    _centrality_flash_kernel[grid](
        Q, K, v, out,
        # Head weighting (ITI-inspired)
        head_weights_ptr, use_head_weights,
        # Q strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # K strides
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        # v strides
        v.stride(0), v.stride(1) if v.dim() > 1 else 1,
        # Out strides
        out.stride(0), out.stride(1), out.stride(2),
        # Dimensions (used as autotune keys)
        S=S, D=D,
        # Block size for head dimension
        BLOCK_D=BLOCK_D,
    )

    return out


def centrality_flash_reference(
    Q: torch.Tensor,  # (B, H, S, D)
    K: torch.Tensor,  # (B, H, S, D)
    v: torch.Tensor,  # (B, S)
) -> torch.Tensor:    # (B, H, S)
    """
    Reference implementation for testing (explicit attention computation).

    WARNING: This materializes the full (S, S) attention matrix.
    Only use for testing with small sequences.
    """
    B, H, S, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # Compute attention scores: (B, H, S, S)
    attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(S, S, device=Q.device, dtype=torch.bool), diagonal=1)
    attn_scores.masked_fill_(causal_mask, float('-inf'))

    # Softmax
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(Q.dtype)

    # Weighted sum with v: (B, H, S, S) @ (B, 1, S, 1) -> (B, H, S)
    # Expand v for broadcasting: (B, S) -> (B, 1, S)
    v_expanded = v.unsqueeze(1)  # (B, 1, S)

    # out[b,h,i] = sum_j attn_probs[b,h,i,j] * v[b,j]
    out = torch.einsum('bhij,bj->bhi', attn_probs, v)

    return out
