"""
AG-SAR Low-Level Operations.

Backend abstraction: Triton GPU kernels when available, PyTorch fallback otherwise.
"""

import os

_TRITON_AVAILABLE = False
_GATE_TRITON_AVAILABLE = False
_AUTHORITY_TRITON_AVAILABLE = False
_FORCE_TORCH = os.environ.get("AG_SAR_USE_TORCH", "0") == "1"

if not _FORCE_TORCH:
    try:
        import triton
        _TRITON_AVAILABLE = True
    except ImportError:
        pass

    try:
        from .triton_gate import fused_stability_gate
        _GATE_TRITON_AVAILABLE = True
    except ImportError:
        pass

    try:
        from .triton_authority import compute_flash_authority_v3
        _AUTHORITY_TRITON_AVAILABLE = True
    except ImportError:
        pass

if not _GATE_TRITON_AVAILABLE:
    from .torch_functional import compute_stability_gate as fused_stability_gate

# FlashAuthority fallback (returns None to signal unavailability)
if not _AUTHORITY_TRITON_AVAILABLE:
    compute_flash_authority_v3 = None

from .torch_functional import (
    align_gqa_heads,
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_authority_flow_streaming,
    compute_authority_flow_recursive,
    compute_authority_from_qk,
    compute_logit_divergence_jsd,
    compute_mlp_divergence,
    compute_stability_gate,
    compute_structural_gain,
)

__all__ = [
    "_TRITON_AVAILABLE",
    "_AUTHORITY_TRITON_AVAILABLE",
    "align_gqa_heads",
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    "compute_authority_flow_streaming",
    "compute_authority_flow_recursive",
    "compute_authority_from_qk",
    "compute_flash_authority_v3",
    "compute_logit_divergence_jsd",
    "compute_mlp_divergence",
    "compute_stability_gate",
    "compute_structural_gain",
    "fused_stability_gate",
]
