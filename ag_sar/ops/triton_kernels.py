"""
Triton kernels for optimized AG-SAR operations.

Kernels:
1. fused_rmsnorm_indexed_linear: Optimized for JSD signal (X @ W[indices].T)

Triton 3.x Compatible Version.
"""

import torch
import triton
import triton.language as tl

# Check Triton version for compatibility
TRITON_VERSION = tuple(int(x) for x in triton.__version__.split('.')[:2])
TRITON_3_PLUS = TRITON_VERSION[0] >= 3


@triton.jit
def fused_rmsnorm_indexed_linear_kernel_v3(
    x_ptr,              # [Batch, Dim]
    weight_ptr,         # [Vocab, Dim]
    indices_ptr,        # [K]
    out_ptr,            # [Batch, K]
    stride_x_batch, stride_x_dim,
    stride_w_vocab, stride_w_dim,
    stride_out_batch, stride_out_k,
    Dim: tl.constexpr,
    K: tl.constexpr,
    eps,
    BLOCK_DIM: tl.constexpr,
):
    """
    Triton 3.x compatible version.
    Computes: out = RMSNorm(x) @ Weight[indices].T

    Grid: (Batch, K) - one program per output element
    """
    batch_pid = tl.program_id(0)
    k_pid = tl.program_id(1)

    # -----------------------------------------------------------
    # 1. Compute RMS of X row (FUSED)
    # -----------------------------------------------------------
    x_row_start = x_ptr + batch_pid * stride_x_batch

    sum_sq = tl.zeros([1], dtype=tl.float32)
    for off in range(0, Dim, BLOCK_DIM):
        cols = off + tl.arange(0, BLOCK_DIM)
        mask = cols < Dim
        val = tl.load(x_row_start + cols * stride_x_dim, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(val * val)

    rms = tl.sqrt((sum_sq / Dim) + eps)
    rms_inv = 1.0 / rms

    # -----------------------------------------------------------
    # 2. Load index for this K position
    # -----------------------------------------------------------
    w_idx = tl.load(indices_ptr + k_pid)

    # -----------------------------------------------------------
    # 3. Compute dot product: x_norm @ weight[w_idx]
    # -----------------------------------------------------------
    acc = tl.zeros([1], dtype=tl.float32)

    for off in range(0, Dim, BLOCK_DIM):
        cols = off + tl.arange(0, BLOCK_DIM)
        mask_dim = cols < Dim

        # Load X chunk and normalize
        x_val = tl.load(x_row_start + cols * stride_x_dim, mask=mask_dim, other=0.0).to(tl.float32)
        x_norm = x_val * rms_inv

        # Load Weight row
        w_ptr = weight_ptr + w_idx * stride_w_vocab + cols * stride_w_dim
        w_val = tl.load(w_ptr, mask=mask_dim, other=0.0).to(tl.float32)

        # Accumulate dot product
        acc += tl.sum(x_norm * w_val)

    # Store result (extract scalar from 1-element tensor)
    out_offset = batch_pid * stride_out_batch + k_pid * stride_out_k
    tl.store(out_ptr + out_offset, tl.sum(acc))


# -------------------------------------------------------------------------
# Python Wrappers
# -------------------------------------------------------------------------

def fused_rmsnorm_linear_subset(x: torch.Tensor, weight: torch.Tensor, indices: torch.Tensor, eps=1e-6):
    """
    x: [Batch, Dim] or [Dim]
    weight: [Vocab, Dim]
    indices: [K]

    Returns: [Batch, K] logits for indexed vocabulary subset
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)

    batch, dim = x.shape
    k = indices.shape[0]
    out = torch.empty((batch, k), device=x.device, dtype=x.dtype)

    BLOCK_DIM = min(1024, triton.next_power_of_2(dim))

    # Grid: one program per (batch, k) output
    grid = (batch, k)

    fused_rmsnorm_indexed_linear_kernel_v3[grid](
        x, weight, indices, out,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        dim, k, eps,
        BLOCK_DIM=BLOCK_DIM,
    )
    return out
