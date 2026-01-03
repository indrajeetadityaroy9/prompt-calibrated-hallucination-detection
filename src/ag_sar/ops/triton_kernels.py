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
        - Higher register count (but still limited per thread)
        - TMA (Tensor Memory Accelerator)

    Note: Large block sizes can cause register spilling. We use moderate
    block sizes with higher warp counts for better occupancy on H100.
    """
    # Conservative configs that work on all GPUs including H100
    # Smaller blocks reduce register pressure while maintaining throughput
    configs = [
        # Small blocks - lowest register pressure, works everywhere
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        # Medium blocks - good balance
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ]

    # H100-specific configs - use moderate blocks with more warps
    if _is_hopper():
        configs.extend([
            # H100: moderate blocks, high warp count for occupancy
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
            # Slightly larger for high-memory cases
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        ])
    else:
        # A100/older GPU configs
        configs.extend([
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        ])

    return configs


@triton.autotune(configs=_get_autotune_configs(), key=['D'])  # Only key on D, not S - avoids recompile per sequence
@triton.jit
def _centrality_kernel(
    Q_ptr, K_ptr, v_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vs,
    stride_ob, stride_oh, stride_os,
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
) -> torch.Tensor:
    """
    Launch Triton centrality kernel.

    Args:
        Q: (B, H, S, D) Query vectors
        K: (B, H, S, D) Key vectors
        v: (B, S) Value signal

    Returns:
        out: (B, H, S) Attention-weighted centrality
    """
    B, H, S, D = Q.shape
    device = Q.device

    # Triton requires current CUDA device to match tensor device
    with torch.cuda.device(device):
        out = torch.empty((B, H, S), device=device, dtype=Q.dtype)

        BLOCK_D = triton.next_power_of_2(D)

        # Use minimum BLOCK_M (32) for safe grid sizing - kernel handles bounds internally
        # This ensures grid covers all tokens regardless of which autotune config is selected
        grid = (B, H, triton.cdiv(S, 32))

        _centrality_kernel[grid](
            Q, K, v, out,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            v.stride(0), v.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            S=S, D=D, BLOCK_D=BLOCK_D,
        )

    return out
