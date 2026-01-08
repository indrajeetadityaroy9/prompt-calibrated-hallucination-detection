"""
Triton Kernels for Fused Semantic Dispersion.

Hybrid approach:
    - PyTorch: TopK selection (already well-optimized)
    - Triton: Fused embedding gather + L2 norm + cosine distance

This eliminates the multiple memory round-trips in the original implementation:
    Original: topk → embedding → normalize → centroid → normalize → cosine → reduce
    Fused: topk (PyTorch) → single Triton kernel for rest

H100 Optimization Notes:
    - BLOCK_D sized for hidden dim (4096-8192)
    - FP32 accumulation for numerical stability
    - Coalesced memory access for embeddings
"""

import triton
import triton.language as tl
import torch
from typing import Literal


def _is_hopper() -> bool:
    """Check if running on H100 (Hopper architecture)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _get_dispersion_autotune_configs():
    """Autotune configurations for semantic dispersion kernel."""
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


@triton.autotune(configs=_get_dispersion_autotune_configs(), key=['D', 'K'])
@triton.jit
def _fused_embed_cosine_top1_kernel(
    top_indices_ptr,    # (N, K) - indices from TopK
    top_probs_ptr,      # (N, K) - probabilities from TopK (normalized)
    embed_ptr,          # (V, D) - embedding matrix
    out_ptr,            # (N,) - output dispersion
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    stride_in, stride_ik,
    stride_pn, stride_pk,
    stride_ev, stride_ed,
    stride_on,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for Top-1 Projection dispersion.

    For each position n:
        1. Load K embeddings using top_indices
        2. Compute L2 norms
        3. Compute cosine similarity to top-1
        4. Weighted average distance
    """
    pid_n = tl.program_id(0)

    if pid_n >= N:
        return

    # Load top-1 index (highest probability)
    top1_idx = tl.load(top_indices_ptr + pid_n * stride_in + 0 * stride_ik)

    # Compute top-1 embedding norm (streaming over D)
    top1_norm_sq = 0.0

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        top1_embed_ptrs = embed_ptr + top1_idx * stride_ev + offs_d * stride_ed
        top1_embed = tl.load(top1_embed_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        top1_norm_sq += tl.sum(top1_embed * top1_embed, axis=0)

    top1_norm = tl.sqrt(top1_norm_sq + 1e-10)

    # Compute weighted distance to top-1
    dispersion = 0.0

    for k in range(K):
        # Load index and probability for this top-k token
        idx_k = tl.load(top_indices_ptr + pid_n * stride_in + k * stride_ik)
        prob_k = tl.load(top_probs_ptr + pid_n * stride_pn + k * stride_pk).to(tl.float32)

        # Compute embedding norm and dot product with top-1
        embed_k_norm_sq = 0.0
        dot_product = 0.0

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load embedding for token k
            embed_k_ptrs = embed_ptr + idx_k * stride_ev + offs_d * stride_ed
            embed_k = tl.load(embed_k_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            # Load top-1 embedding
            top1_embed_ptrs = embed_ptr + top1_idx * stride_ev + offs_d * stride_ed
            top1_embed = tl.load(top1_embed_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            embed_k_norm_sq += tl.sum(embed_k * embed_k, axis=0)
            dot_product += tl.sum(embed_k * top1_embed, axis=0)

        embed_k_norm = tl.sqrt(embed_k_norm_sq + 1e-10)

        # Cosine similarity and distance
        cos_sim = dot_product / (embed_k_norm * top1_norm + 1e-10)
        distance = 1.0 - cos_sim

        # Weighted contribution
        dispersion += prob_k * distance

    # Clamp to [0, 1]
    dispersion = tl.minimum(tl.maximum(dispersion, 0.0), 1.0)

    tl.store(out_ptr + pid_n * stride_on, dispersion)


@triton.autotune(configs=_get_dispersion_autotune_configs(), key=['D', 'K'])
@triton.jit
def _fused_embed_cosine_centroid_kernel(
    top_indices_ptr,    # (N, K)
    top_probs_ptr,      # (N, K)
    embed_ptr,          # (V, D)
    out_ptr,            # (N,)
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    D: tl.constexpr,
    stride_in, stride_ik,
    stride_pn, stride_pk,
    stride_ev, stride_ed,
    stride_on,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for Centroid Variance dispersion.

    For each position n:
        1. Load K embeddings using top_indices
        2. Compute weighted centroid (two-pass: accumulate then normalize)
        3. Compute weighted cosine distance from centroid

    This matches the PyTorch implementation:
        centroid = sum(prob_k * embed_k)
        dispersion = sum(prob_k * (1 - cos_sim(embed_k, centroid)))
    """
    pid_n = tl.program_id(0)

    if pid_n >= N:
        return

    # ==========================================================================
    # Pass 1: Compute weighted centroid norm (for later normalization)
    # We stream through D dimension to compute |centroid|^2
    # ==========================================================================
    centroid_norm_sq = 0.0

    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # Accumulate weighted sum for this D chunk
        centroid_chunk = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k in range(K):
            idx_k = tl.load(top_indices_ptr + pid_n * stride_in + k * stride_ik)
            prob_k = tl.load(top_probs_ptr + pid_n * stride_pn + k * stride_pk).to(tl.float32)

            embed_k_ptrs = embed_ptr + idx_k * stride_ev + offs_d * stride_ed
            embed_k = tl.load(embed_k_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            centroid_chunk += prob_k * embed_k

        centroid_norm_sq += tl.sum(centroid_chunk * centroid_chunk, axis=0)

    centroid_norm = tl.sqrt(centroid_norm_sq + 1e-10)

    # ==========================================================================
    # Pass 2: Compute weighted distances from centroid
    # For each embedding k, compute cos_sim(embed_k, centroid)
    # ==========================================================================
    dispersion = 0.0

    for k in range(K):
        idx_k = tl.load(top_indices_ptr + pid_n * stride_in + k * stride_ik)
        prob_k = tl.load(top_probs_ptr + pid_n * stride_pn + k * stride_pk).to(tl.float32)

        # Compute embed_k norm and dot product with centroid
        embed_k_norm_sq = 0.0
        dot_product = 0.0

        for d_start in range(0, D, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            # Load embedding k
            embed_k_ptrs = embed_ptr + idx_k * stride_ev + offs_d * stride_ed
            embed_k = tl.load(embed_k_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            embed_k_norm_sq += tl.sum(embed_k * embed_k, axis=0)

            # Recompute centroid chunk for dot product
            centroid_chunk = tl.zeros([BLOCK_D], dtype=tl.float32)
            for kk in range(K):
                idx_kk = tl.load(top_indices_ptr + pid_n * stride_in + kk * stride_ik)
                prob_kk = tl.load(top_probs_ptr + pid_n * stride_pn + kk * stride_pk).to(tl.float32)

                embed_kk_ptrs = embed_ptr + idx_kk * stride_ev + offs_d * stride_ed
                embed_kk = tl.load(embed_kk_ptrs, mask=mask_d, other=0.0).to(tl.float32)

                centroid_chunk += prob_kk * embed_kk

            dot_product += tl.sum(embed_k * centroid_chunk, axis=0)

        embed_k_norm = tl.sqrt(embed_k_norm_sq + 1e-10)

        # Cosine similarity with centroid
        cos_sim = dot_product / (embed_k_norm * centroid_norm + 1e-10)
        distance = 1.0 - cos_sim

        # Weighted contribution
        dispersion += prob_k * distance

    # Clamp to [0, 1]
    dispersion = tl.minimum(tl.maximum(dispersion, 0.0), 1.0)

    tl.store(out_ptr + pid_n * stride_on, dispersion)


def fused_semantic_dispersion(
    probs: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    method: Literal["top1_projection", "centroid_variance"] = "top1_projection",
) -> torch.Tensor:
    """
    Compute semantic dispersion using fused Triton kernel.

    Hybrid approach:
        1. PyTorch TopK (already optimized)
        2. Triton fused embedding gather + cosine computation

    Args:
        probs: (N, V) probability distribution (already softmax'd)
        embed_matrix: (V, D) output embedding matrix
        k: Number of top tokens to consider (typically 5-10)
        method: "top1_projection" or "centroid_variance"

    Returns:
        dispersion: (N,) semantic dispersion scores in [0, 1]

    Note:
        This is only used for large vocabularies (V >= 32000).
        For smaller vocabs, the PyTorch implementation is faster.
    """
    assert probs.is_cuda, "Triton kernel requires CUDA tensors"
    assert embed_matrix.is_cuda, "Triton kernel requires CUDA tensors"
    assert probs.dim() == 2, f"Expected 2D probs, got {probs.dim()}D"
    assert embed_matrix.dim() == 2, f"Expected 2D embed_matrix, got {embed_matrix.dim()}D"

    N, V = probs.shape
    V_embed, D = embed_matrix.shape
    assert V == V_embed, f"Vocab size mismatch: probs={V}, embed_matrix={V_embed}"

    device = probs.device

    # Step 1: PyTorch TopK (highly optimized)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)  # (N, K)

    # Renormalize probabilities within TopK
    top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # Ensure contiguous
    top_indices = top_indices.contiguous().to(torch.int32)
    top_probs = top_probs.contiguous()
    embed_matrix = embed_matrix.contiguous()

    with torch.cuda.device(device):
        out = torch.empty(N, device=device, dtype=torch.float32)

        grid = (N,)

        if method == "top1_projection":
            _fused_embed_cosine_top1_kernel[grid](
                top_indices, top_probs, embed_matrix, out,
                N=N, K=k, V=V, D=D,
                stride_in=top_indices.stride(0), stride_ik=top_indices.stride(1),
                stride_pn=top_probs.stride(0), stride_pk=top_probs.stride(1),
                stride_ev=embed_matrix.stride(0), stride_ed=embed_matrix.stride(1),
                stride_on=out.stride(0),
            )
        elif method == "centroid_variance":
            _fused_embed_cosine_centroid_kernel[grid](
                top_indices, top_probs, embed_matrix, out,
                N=N, K=k, V=V, D=D,
                stride_in=top_indices.stride(0), stride_ik=top_indices.stride(1),
                stride_pn=top_probs.stride(0), stride_pk=top_probs.stride(1),
                stride_ev=embed_matrix.stride(0), stride_ed=embed_matrix.stride(1),
                stride_on=out.stride(0),
            )
        else:
            raise ValueError(f"Unknown method: {method}. Use 'top1_projection' or 'centroid_variance'")

    return out
