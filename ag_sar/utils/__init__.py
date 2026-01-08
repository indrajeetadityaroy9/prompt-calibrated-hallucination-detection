"""
AG-SAR Utilities Module.

This module provides hardware optimization functions, numerical stability utilities,
and model inspection helpers used throughout the AG-SAR pipeline.

Hardware Optimization:
    - enable_tf32: Enables TensorFloat-32 for ~3x matmul speedup on Ampere/Hopper
    - enable_h100_optimizations: Full H100 optimization suite (TF32 + Flash SDP)
    - get_optimal_dtype: Returns BFloat16 for Ampere+, Float16 for older GPUs
    - optimize_for_inference: Applies all inference optimizations to a model

Numerical Stability:
    - safe_normalize: L2 normalization with epsilon to prevent division by zero

Attention Utilities:
    - apply_attention_mask: Applies attention mask to attention weights

Model Inspection:
    - get_model_dtype: Extract dtype from model parameters
    - get_model_device: Extract device from model parameters

Performance Notes:
    TF32 (TensorFloat-32) provides ~3x speedup for FP32 operations on Ampere/Hopper
    tensor cores by using 19-bit mantissa (vs 23-bit FP32) with FP32 dynamic range.
    This is transparent to user code but may cause minor numerical differences.

    BFloat16 is preferred over Float16 for Ampere+ GPUs because it has FP32-equivalent
    dynamic range, avoiding overflow issues common with Float16 in attention softmax.
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
    # Numerical Stability
    "safe_normalize",
    # Attention Utilities
    "apply_attention_mask",
    # Model Inspection
    "get_model_dtype",
    "get_model_device",
]
