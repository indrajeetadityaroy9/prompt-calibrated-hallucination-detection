"""
AG-SAR Utilities.

- tensor: H100 optimizations, TF32 setup, numerical stability, attention masking
"""

from .tensor import (
    # H100/Hopper Optimizations
    enable_tf32,
    enable_h100_optimizations,
    is_tf32_enabled,
    is_h100,
    get_optimal_dtype,
    optimize_for_inference,
    # Numerical Utilities
    safe_normalize,
    # Attention Utilities
    apply_attention_mask,
    # Model Utilities
    get_model_dtype,
    get_model_device,
)

__all__ = [
    # H100 Optimizations
    "enable_tf32",
    "enable_h100_optimizations",
    "is_tf32_enabled",
    "is_h100",
    "get_optimal_dtype",
    "optimize_for_inference",
    # Numerical
    "safe_normalize",
    # Attention
    "apply_attention_mask",
    # Model
    "get_model_dtype",
    "get_model_device",
]
