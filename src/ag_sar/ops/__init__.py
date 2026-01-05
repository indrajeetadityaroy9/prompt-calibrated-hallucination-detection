"""
AG-SAR Low-Level Operations.

Backend abstraction layer that auto-selects Triton (Linux) or PyTorch fallback.
All operations are O(N) memory and designed for streaming inference.
"""

import sys

# Backend selection: Triton on Linux, PyTorch fallback elsewhere
# Environment variable AG_SAR_USE_TORCH=1 forces PyTorch fallback
import os
_TRITON_AVAILABLE = False
_FORCE_TORCH = os.environ.get("AG_SAR_USE_TORCH", "0") == "1"

if sys.platform == "linux" and not _FORCE_TORCH:
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
    # Authority Flow
    compute_authority_flow,
    compute_authority_flow_vectorized,
    # MLP Divergence & Stability Gate
    compute_mlp_divergence,
    compute_stability_gate,
)

__all__ = [
    # Backend
    "centrality_kernel",
    "_TRITON_AVAILABLE",
    # GQA
    "align_gqa_heads",
    "get_gqa_config",
    # Authority Flow
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    # MLP Divergence & Stability Gate
    "compute_mlp_divergence",
    "compute_stability_gate",
]
