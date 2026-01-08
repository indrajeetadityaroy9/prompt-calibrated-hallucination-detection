"""
Triton Kernel for Fused Stability Gate.

Fuses the following operations into a single kernel:
    - L2 normalization of h_attn
    - L2 normalization of h_block
    - Cosine similarity computation
    - Divergence calculation (1 - cos_sim)
    - Exponential gate (exp(-sensitivity * divergence))

This eliminates 4+ kernel launches and intermediate tensor allocations.

Memory Complexity:
    - Input: O(B*S*D) for h_attn and h_block
    - Output: O(B*S) gate values
    - No intermediate tensors
"""

import triton
import triton.language as tl
import torch


def _is_hopper() -> bool:
    """Check if running on H100 (Hopper architecture)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _get_gate_autotune_configs():
    """Autotune configurations for stability gate kernel."""
    configs = [
        # Conservative configs (all GPUs)
        triton.Config({'BLOCK_D': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 1024}, num_warps=8, num_stages=2),
    ]

    if _is_hopper():
        configs.extend([
            triton.Config({'BLOCK_D': 2048}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_D': 4096}, num_warps=16, num_stages=3),
        ])
    else:
        configs.extend([
            triton.Config({'BLOCK_D': 2048}, num_warps=8, num_stages=3),
        ])

    return configs


@triton.autotune(configs=_get_gate_autotune_configs(), key=['D'])
@triton.jit
def _fused_stability_gate_kernel(
    h_attn_ptr,         # (B, S, D)
    h_block_ptr,        # (B, S, D)
    out_ptr,            # (B, S)
    sensitivity,        # float scalar
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    stride_ab, stride_as, stride_ad,
    stride_bb, stride_bs, stride_bd,
    stride_ob, stride_os,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for stability gate computation.

    For each position (b, s):
        1. Load h_attn[b, s, :] and h_block[b, s, :] in chunks
        2. Accumulate: norm_a_sq, norm_b_sq, dot_ab
        3. Compute: cos_sim = dot_ab / (norm_a * norm_b)
        4. Compute: gate = exp(-sensitivity * (1 - cos_sim))
        5. Store gate[b, s]
    """
    # Flatten (b, s) into single program id
    pid = tl.program_id(0)
    total_positions = B * S

    if pid >= total_positions:
        return

    # Decode batch and sequence indices
    b = pid // S
    s = pid % S

    # Accumulate norms and dot product
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    dot_ab = 0.0

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Load h_attn[b, s, d_start:d_start+BLOCK_D]
        h_attn_ptrs = h_attn_ptr + b * stride_ab + s * stride_as + offs_d * stride_ad
        h_attn_block = tl.load(h_attn_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Load h_block[b, s, d_start:d_start+BLOCK_D]
        h_block_ptrs = h_block_ptr + b * stride_bb + s * stride_bs + offs_d * stride_bd
        h_block_block = tl.load(h_block_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Accumulate statistics
        norm_a_sq += tl.sum(h_attn_block * h_attn_block, axis=0)
        norm_b_sq += tl.sum(h_block_block * h_block_block, axis=0)
        dot_ab += tl.sum(h_attn_block * h_block_block, axis=0)

    # Compute cosine similarity
    norm_a = tl.sqrt(norm_a_sq + 1e-10)
    norm_b = tl.sqrt(norm_b_sq + 1e-10)
    cos_sim = dot_ab / (norm_a * norm_b + 1e-10)

    # Compute divergence and gate
    divergence = 1.0 - cos_sim
    gate = tl.exp(-sensitivity * divergence)

    # Clamp to [0, 1] for numerical safety
    gate = tl.minimum(tl.maximum(gate, 0.0), 1.0)

    # Store output
    out_ptrs = out_ptr + b * stride_ob + s * stride_os
    tl.store(out_ptrs, gate)


def fused_stability_gate(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    sensitivity: float = 10.0,
) -> torch.Tensor:
    """
    Compute stability gate using fused Triton kernel.

    Fuses L2 normalization, cosine similarity, and exponential gate
    into a single memory-efficient kernel.

    Args:
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        sensitivity: Gate sensitivity (higher = sharper gate)

    Returns:
        gate: (B, S) gate values in [0, 1]
            - 1.0 = MLP agrees with attention (stable)
            - 0.0 = MLP overrides attention (unstable)

    Note:
        This is equivalent to:
            divergence = 1 - CosineSim(h_attn, h_block)
            gate = exp(-sensitivity * divergence)
    """
    assert h_attn.is_cuda, "Triton kernel requires CUDA tensors"
    assert h_block.is_cuda, "Triton kernel requires CUDA tensors"
    assert h_attn.dim() == 3, f"Expected 3D h_attn, got {h_attn.dim()}D"
    assert h_block.dim() == 3, f"Expected 3D h_block, got {h_block.dim()}D"
    assert h_attn.shape == h_block.shape, f"Shape mismatch: {h_attn.shape} vs {h_block.shape}"

    B, S, D = h_attn.shape
    device = h_attn.device

    # Ensure contiguous
    h_attn = h_attn.contiguous()
    h_block = h_block.contiguous()

    with torch.cuda.device(device):
        out = torch.empty(B, S, device=device, dtype=torch.float32)

        # Grid: one program per (b, s) position
        grid = (B * S,)

        _fused_stability_gate_kernel[grid](
            h_attn, h_block, out,
            sensitivity,
            B=B, S=S, D=D,
            stride_ab=h_attn.stride(0), stride_as=h_attn.stride(1), stride_ad=h_attn.stride(2),
            stride_bb=h_block.stride(0), stride_bs=h_block.stride(1), stride_bd=h_block.stride(2),
            stride_ob=out.stride(0), stride_os=out.stride(1),
        )

    return out
