"""
AG-SAR Low-Level Operations.

Backend abstraction layer that auto-selects Triton (Linux) or PyTorch fallback.
All operations are O(N) memory and designed for streaming inference.
"""

import sys

# Backend selection: Triton on Linux, PyTorch fallback elsewhere
_TRITON_AVAILABLE = False

if sys.platform == "linux":
    try:
        import triton
        from .triton_kernels import centrality_kernel
        _TRITON_AVAILABLE = True
    except ImportError:
        pass

if not _TRITON_AVAILABLE:
    from .torch_functional import centrality_kernel_fallback as centrality_kernel

# Always export pure PyTorch functions for direct use
from .torch_functional import (
    # GQA Support
    align_gqa_heads,
    get_gqa_config,
    # Register Filter (Mechanism 1)
    EMAState,
    fisher_kurtosis,
    welford_update,
    compute_register_mask,
    # Authority Flow (Mechanism 2)
    compute_authority_flow,
    compute_authority_flow_vectorized,
    # Spectral Roughness (Mechanism 3)
    compute_spectral_roughness,
    compute_spectral_roughness_gqa,
    compute_mlp_divergence,
    # SnapKV Eviction (Mechanism 4)
    compute_snapkv_eviction,
    compress_kv_cache,
)

__all__ = [
    # Backend
    "centrality_kernel",
    "_TRITON_AVAILABLE",
    # GQA
    "align_gqa_heads",
    "get_gqa_config",
    # Register Filter
    "EMAState",
    "fisher_kurtosis",
    "welford_update",
    "compute_register_mask",
    # Authority Flow
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    # Spectral Roughness
    "compute_spectral_roughness",
    "compute_spectral_roughness_gqa",
    "compute_mlp_divergence",
    # SnapKV
    "compute_snapkv_eviction",
    "compress_kv_cache",
]
