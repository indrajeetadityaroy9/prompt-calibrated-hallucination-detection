"""
FlashAuthority: Fused Triton Kernel for Recursive Authority Flow.

Fuses Attention Computation + Structural Gain + Recursive Accumulation into a single
GPU kernel, reducing memory I/O from O(N²) to O(N) and eliminating Python loop overhead.

This kernel computes:
    A[t] = Σ_context α_tj + γ_t × Σ_gen α_tj × A[j]

Where attention weights α are computed on-the-fly from Q, K states.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _flash_authority_kernel(
    Q_ptr, K_ptr, Gamma_ptr, A_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_gb, stride_gs,
    stride_ab, stride_ah, stride_as,
    scale,
    Z, H, N_CTX,
    PROMPT_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused recursive authority computation kernel.

    Parallelizes over (Batch, Head). Sequential over tokens due to causal dependency.
    """
    # Program IDs
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    # Base pointers for this batch/head
    Q_base = Q_ptr + off_b * stride_qb + off_h * stride_qh
    K_base = K_ptr + off_b * stride_kb + off_h * stride_kh
    Gamma_base = Gamma_ptr + off_b * stride_gb
    A_base = A_ptr + off_b * stride_ab + off_h * stride_ah

    # Dimension offsets
    offs_d = tl.arange(0, HEAD_DIM)

    # Sequential loop over tokens (causal dependency)
    for t in range(N_CTX):
        if t < PROMPT_LEN:
            # Base case: prompt tokens have authority = 1.0
            tl.store(A_base + t * stride_as, 1.0)
        else:
            # Load query vector for position t: [HEAD_DIM]
            q_t = tl.load(Q_base + t * stride_qs + offs_d * stride_qd)

            # First pass: compute max score for numerical stability
            max_score = -float('inf')

            for k_start in range(0, t + 1, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k <= t

                # Load K block: [BLOCK_K, HEAD_DIM]
                k_block = tl.load(
                    K_base + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=mask_k[:, None],
                    other=0.0
                )

                # Compute attention scores: Q @ K^T
                scores = tl.sum(q_t[None, :] * k_block, axis=1) * scale
                scores = tl.where(mask_k, scores, -float('inf'))

                block_max = tl.max(scores)
                max_score = tl.maximum(max_score, block_max)

            # Second pass: compute softmax and accumulate flows
            sum_exp = 0.0
            flow_context = 0.0
            flow_gen = 0.0

            for k_start in range(0, t + 1, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)
                mask_k = offs_k <= t

                # Reload K (should be in L2 cache)
                k_block = tl.load(
                    K_base + offs_k[:, None] * stride_ks + offs_d[None, :] * stride_kd,
                    mask=mask_k[:, None],
                    other=0.0
                )

                # Recompute scores
                scores = tl.sum(q_t[None, :] * k_block, axis=1) * scale

                # Softmax numerator
                exp_scores = tl.exp(scores - max_score)
                exp_scores = tl.where(mask_k, exp_scores, 0.0)
                sum_exp += tl.sum(exp_scores)

                # Load past authority values A[j] for j < t
                a_past = tl.load(A_base + offs_k * stride_as, mask=mask_k, other=0.0)

                # Separate context vs generation flows
                is_context = offs_k < PROMPT_LEN

                # Context flow: Σ α_tj for j in prompt
                flow_context += tl.sum(tl.where(is_context & mask_k, exp_scores, 0.0))

                # Generation flow: Σ α_tj × A[j] for j in generated
                flow_gen += tl.sum(tl.where((~is_context) & mask_k, exp_scores * a_past, 0.0))

            # Load structural gain γ[t]
            gamma_t = tl.load(Gamma_base + t * stride_gs)

            # Compute final authority
            if sum_exp > 1e-9:
                inv_sum = 1.0 / sum_exp
                authority = flow_context * inv_sum + gamma_t * flow_gen * inv_sum
            else:
                authority = 0.0

            # Clamp to [0, 1] and store
            authority = tl.minimum(tl.maximum(authority, 0.0), 1.0)
            tl.store(A_base + t * stride_as, authority)


def flash_authority_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    gamma: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    Compute recursive authority flow using fused Triton kernel.

    Args:
        q: Query states [B, H, S, D] (float16/bfloat16)
        k: Key states [B, H, S, D] (float16/bfloat16)
        gamma: Structural gain [B, S] (float32)
        prompt_len: Number of prompt tokens

    Returns:
        authority: [B, S] averaged over heads
    """
    assert q.is_cuda and k.is_cuda, "FlashAuthority requires CUDA tensors"
    assert q.shape == k.shape, f"Q/K shape mismatch: {q.shape} vs {k.shape}"

    B, H, S, D = q.shape

    # Ensure contiguous memory layout
    q = q.contiguous()
    k = k.contiguous()
    gamma = gamma.contiguous()

    # Output buffer: [B, H, S] - per-head authority
    authority_per_head = torch.empty((B, H, S), device=q.device, dtype=torch.float32)

    # Attention scale
    scale = 1.0 / math.sqrt(D)

    # Block size (tunable)
    BLOCK_K = min(64, triton.next_power_of_2(S))

    # Launch kernel: parallelize over (batch, heads)
    grid = (B, H)

    _flash_authority_kernel[grid](
        q, k, gamma, authority_per_head,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        gamma.stride(0), gamma.stride(1),
        authority_per_head.stride(0), authority_per_head.stride(1), authority_per_head.stride(2),
        scale,
        B, H, S,
        PROMPT_LEN=prompt_len,
        HEAD_DIM=D,
        BLOCK_K=BLOCK_K,
    )

    # Average over heads
    return authority_per_head.mean(dim=1)


def compute_flash_authority_v3(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    structural_gain: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    High-level API for FlashAuthority kernel.

    Handles input validation, GQA expansion, and provides a clean interface.

    Args:
        query_states: [B, H_q, S, D] or [B, S, H_q, D] query vectors
        key_states: [B, H_k, S, D] or [B, S, H_k, D] key vectors (H_k may differ from H_q for GQA)
        structural_gain: [B, S] per-token structural gain (γ_t)
        prompt_length: Number of context/prompt tokens

    Returns:
        authority: [B, S] authority scores in [0, 1]
    """
    # Input is expected to be [B, H, S, D] format
    # GQA Expansion: For models with Grouped Query Attention (Llama-3, Mistral, Qwen),
    # K has fewer heads than Q. Expand K to match Q for the kernel.
    num_q_heads = query_states.shape[1]
    num_k_heads = key_states.shape[1]

    if num_q_heads != num_k_heads:
        # Expand K heads to match Q heads
        n_rep = num_q_heads // num_k_heads
        B, H_k, S, D = key_states.shape
        # [B, H_k, S, D] -> [B, H_k, n_rep, S, D] -> [B, H_q, S, D]
        key_states = key_states[:, :, None, :, :].expand(
            B, H_k, n_rep, S, D
        ).reshape(B, num_q_heads, S, D).contiguous()

    return flash_authority_forward(query_states, key_states, structural_gain, prompt_length)
