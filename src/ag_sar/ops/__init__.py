"""
AG-SAR Low-Level Operations Module.

This module provides the core tensor operations for Authority Flow computation.
It implements a backend abstraction layer that selects between Triton GPU kernels
(when available) and pure PyTorch fallback (always available).

Mechanism Implementation:
    - Authority Flow: O(N) recursive prompt recharge computation
    - Stability Gate: MLP-attention divergence for context/parametric gating
    - GQA Support: Grouped Query Attention head expansion for Llama-3

Backend Selection:
    1. If AG_SAR_USE_TORCH=1 environment variable is set, forces PyTorch fallback
    2. Otherwise, attempts to import Triton and use GPU-optimized kernels
    3. Falls back to PyTorch if Triton import fails (macOS, Windows, older GPUs)

    The backend choice is transparent to callers - all functions have identical
    signatures and semantics regardless of backend.

Performance Characteristics:
    - Triton backend: ~2-3x faster on NVIDIA GPUs with CUDA 12+
    - PyTorch backend: Works everywhere, benefits from torch.compile on Ampere+
    - All operations are O(N) memory to enable streaming inference

Environment Variables:
    AG_SAR_USE_TORCH: Set to "1" to force PyTorch backend (skip Triton)

Public API:
    - centrality_kernel: Attention-weighted centrality computation
    - compute_authority_flow: Recursive authority with prompt recharge
    - compute_authority_flow_vectorized: Single-pass vectorized approximation
    - compute_mlp_divergence: MLP-attention cosine divergence
    - compute_stability_gate: Exponential gate from divergence
    - align_gqa_heads: Expand KV heads to match query heads (GQA)
    - get_gqa_config: Extract GQA parameters from model config
"""

import os
import warnings

# Backend selection flag
_TRITON_AVAILABLE = False
_FORCE_TORCH = os.environ.get("AG_SAR_USE_TORCH", "0") == "1"

if _FORCE_TORCH:
    warnings.warn(
        "AG_SAR_USE_TORCH=1 is set. Using PyTorch fallback instead of Triton kernels.",
        UserWarning,
    )
else:
    try:
        import triton
        from .triton_kernels import centrality_kernel
        _TRITON_AVAILABLE = True
    except ImportError:
        pass  # Fall back to PyTorch silently

if not _TRITON_AVAILABLE:
    from .torch_functional import centrality_kernel_fallback as centrality_kernel

# Always export pure PyTorch functions for direct use
from .torch_functional import (
    # GQA Support for Llama-3 and similar architectures
    align_gqa_heads,
    get_gqa_config,
    # Authority Flow computation
    compute_authority_flow,
    compute_authority_flow_vectorized,
    # MLP Divergence and Stability Gate
    compute_mlp_divergence,
    compute_stability_gate,
)

__all__ = [
    # Backend indicator
    "centrality_kernel",
    "_TRITON_AVAILABLE",
    # GQA Support
    "align_gqa_heads",
    "get_gqa_config",
    # Authority Flow
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    # MLP Divergence and Stability Gate
    "compute_mlp_divergence",
    "compute_stability_gate",
]
