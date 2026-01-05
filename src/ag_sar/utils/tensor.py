"""
Tensor Utilities for AG-SAR.

Handles H100/Hopper-specific optimizations, TF32 acceleration,
numerical stability, and attention masking.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════════
#                          H100/HOPPER OPTIMIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def enable_tf32() -> None:
    """
    Enable TF32 precision for ~3x matmul speedup on H100/A100.

    TF32 uses 19-bit precision (8 exponent + 10 mantissa + 1 sign)
    for matrix multiplications while keeping full FP32 for accumulation.

    Called automatically at module import.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')


def enable_h100_optimizations() -> None:
    """
    Enable all Hopper-specific optimizations.

    Call this at the very top of __init__.py for maximum performance.

    Enables:
        - TF32 for 3x faster FP32 matrix operations
        - cuDNN benchmark mode for optimized convolution algorithms
        - Flash SDP for memory-efficient attention (when available)
    """
    if not torch.cuda.is_available():
        return

    # TF32: 3x faster on H100 for FP32 math operations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # cuDNN: Auto-tune for best convolution algorithms
    torch.backends.cudnn.benchmark = True

    # Enable Flash SDP when available (requires PyTorch 2.0+)
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

    # Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def is_tf32_enabled() -> bool:
    """Check if TF32 is enabled."""
    return (
        torch.cuda.is_available() and
        torch.backends.cuda.matmul.allow_tf32 and
        torch.backends.cudnn.allow_tf32
    )


def is_h100() -> bool:
    """
    Check if running on H100 (Hopper) GPU.

    Returns:
        True if GPU compute capability >= 9.0 (Hopper)
    """
    if not torch.cuda.is_available():
        return False

    major, minor = torch.cuda.get_device_capability()
    return major >= 9  # Hopper is compute capability 9.0


def get_optimal_dtype() -> torch.dtype:
    """
    Get optimal dtype for current hardware.

    Returns:
        - bfloat16 for H100/A100 (best balance of speed and precision)
        - float32 for CPU or older GPUs
    """
    if not torch.cuda.is_available():
        return torch.float32

    # Check for bfloat16 support (Ampere+)
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere (A100) or newer (H100)
        return torch.bfloat16

    return torch.float32


def optimize_for_inference() -> None:
    """
    Set PyTorch to inference-optimized mode.

    Disables gradient computation and enables inference optimizations.
    Call before running uncertainty quantification.
    """
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        # Synchronize and clear cache before inference
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
#                           NUMERICAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Numerically stable normalization (sum to 1).

    Args:
        tensor: Input tensor
        dim: Dimension to normalize
        eps: Numerical stability constant

    Returns:
        Normalized tensor summing to 1 along dim
    """
    total = tensor.sum(dim=dim, keepdim=True).clamp(min=eps)
    return tensor / total


# ═══════════════════════════════════════════════════════════════════════════════
#                           ATTENTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def apply_attention_mask(
    attention: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    Apply attention mask by zeroing padded rows/columns.

    CRITICAL: Must apply BEFORE centrality computation to prevent
    padding tokens from participating in power iteration.

    Args:
        attention: (B, H, S, S) or (B, S, S) attention weights
        attention_mask: (B, S) valid token mask (1 = valid, 0 = padding)
        normalize: Re-normalize rows after masking

    Returns:
        Masked attention tensor
    """
    # Expand mask for broadcasting
    if attention.dim() == 4:
        # (B, H, S, S)
        row_mask = attention_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
        col_mask = attention_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, S)
    else:
        # (B, S, S)
        row_mask = attention_mask.unsqueeze(-1)  # (B, S, 1)
        col_mask = attention_mask.unsqueeze(1)   # (B, 1, S)

    # Zero out padded rows and columns
    masked = attention * row_mask.float() * col_mask.float()

    if normalize:
        # Re-normalize rows to sum to 1
        row_sum = masked.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        masked = masked / row_sum

    return masked


# ═══════════════════════════════════════════════════════════════════════════════
#                           MODEL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Extract model parameter dtype."""
    return next(model.parameters()).dtype


def get_model_device(model: nn.Module) -> torch.device:
    """Extract model device."""
    return next(model.parameters()).device
