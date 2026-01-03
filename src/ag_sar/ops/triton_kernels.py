"""
Triton GPU Kernels for Matrix-Free Centrality.

O(N) memory centrality computation optimized for H100 (Hopper).
Only available on Linux with NVIDIA GPUs.

H100 Optimization Notes:
    - Larger BLOCK sizes (256) utilize H100's massive L2/SRAM (50MB)
    - Higher num_warps (8-16) for Hopper's increased warp occupancy
    - num_stages=3-4 for optimal pipelining with TMA
    - BFloat16 for stable numerics at high speed
"""

import sys

if sys.platform != "linux":
    raise ImportError(
        "Triton kernels are only available on Linux. "
        "Use centrality_kernel_fallback from torch_functional.py instead."
    )

import triton
import triton.language as tl
import torch
import math
from typing import Optional


def _is_hopper() -> bool:
    """Check if running on H100 (Hopper architecture)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9  # Hopper is compute capability 9.0


def _get_autotune_configs():
    """
    H100/Hopper-optimized autotune configurations.

    H100 has:
        - 50MB L2 cache (vs 40MB on A100)
        - 256KB shared memory per SM
        - Higher register count
        - TMA (Tensor Memory Accelerator)

    Larger block sizes reduce memory round-trips and better utilize
    the massive on-chip memory hierarchy.
    """
    # Base configs that work on all GPUs
    configs = [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
    ]

    # H100-specific configs with larger blocks and higher warp counts
    if _is_hopper():
        configs.extend([
            # H100 sweetspots: larger blocks, more warps, more stages
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=3),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=16, num_stages=3),
            # Maximum block size for H100's SRAM
            triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256}, num_warps=16, num_stages=4),
        ])
    else:
        # Conservative configs for A100/older GPUs
        configs.extend([
            triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=2),
        ])

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=['S', 'D'])
@triton.jit
def _centrality_kernel(
    Q_ptr, K_ptr, v_ptr, Out_ptr,
    surprisal_ptr, head_scores_ptr,
    USE_SGSS: tl.constexpr,
    STEERING_ALPHA: tl.constexpr,
    STEERING_BETA: tl.constexpr,
    RESPONSE_START: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vs,
    stride_ob, stride_oh, stride_os,
    stride_sb, stride_ss,
    S: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """
    Triton kernel for matrix-free attention-weighted centrality.

    Computes: out[i] = sum_{j <= i} softmax(Q[i] @ K[j].T / sqrt(D)) * v[j]
    Using Flash Attention-style online softmax for O(N) memory.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < S

    Q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    K_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    v_base = v_ptr + pid_b * stride_vb

    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)

    scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < S

        k_ptrs = K_base + offs_n[:, None] * stride_ks + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)

        v_ptrs = v_base + offs_n * stride_vs
        v_block = tl.load(v_ptrs, mask=mask_n, other=0.0)

        scores = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32))) * scale

        causal_mask = offs_n[None, :] <= offs_m[:, None]
        scores = tl.where(causal_mask, scores, float('-inf'))
        scores = tl.where(mask_n[None, :], scores, float('-inf'))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new[:, None])
        l_new = l_i * alpha + tl.sum(beta, axis=1)

        acc_i = acc_i * alpha + tl.sum(beta * v_block[None, :], axis=1)
        m_i = m_new
        l_i = l_new

    out = acc_i / l_i
    Out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
    out_ptrs = Out_base + offs_m * stride_os
    tl.store(out_ptrs, out, mask=mask_m)


def centrality_kernel(
    Q: torch.Tensor,
    K: torch.Tensor,
    v: torch.Tensor,
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
) -> torch.Tensor:
    """
    Launch Triton centrality kernel.

    Args:
        Q: (B, H, S, D) Query vectors
        K: (B, H, S, D) Key vectors
        v: (B, S) Value signal
        surprisal: Optional (B, S) surprisal for SGSS
        head_scores: Optional (H,) head calibration scores
        steering_alpha: SGSS strength
        steering_beta: SGSS sensitivity
        response_start: Response token start index

    Returns:
        out: (B, H, S) Attention-weighted centrality
    """
    B, H, S, D = Q.shape
    device = Q.device

    out = torch.empty((B, H, S), device=device, dtype=Q.dtype)

    USE_SGSS = surprisal is not None and head_scores is not None
    if not USE_SGSS:
        surprisal = torch.zeros((B, S), device=device, dtype=Q.dtype)
        head_scores = torch.zeros((H,), device=device, dtype=Q.dtype)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B, H, triton.cdiv(S, 64))

    _centrality_kernel[grid](
        Q, K, v, out,
        surprisal, head_scores,
        USE_SGSS, steering_alpha, steering_beta, response_start,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        v.stride(0), v.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        surprisal.stride(0), surprisal.stride(1),
        S=S, D=D, BLOCK_D=BLOCK_D,
    )

    return out
