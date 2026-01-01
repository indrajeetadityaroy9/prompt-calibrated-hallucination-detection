"""Utility functions for AG-SAR pipeline."""

from typing import Optional
import torch
import torch.nn as nn


def enable_tf32() -> None:
    """
    Enable TF32 for matmul operations on H100/A100 Tensor Cores.

    TF32 provides significant speedup with minimal precision loss
    for matrix multiplications. Recommended for H100 GPUs.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_model_dtype(model: nn.Module) -> torch.dtype:
    """Get the dtype of model parameters."""
    return next(model.parameters()).dtype


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of model parameters."""
    return next(model.parameters()).device


def apply_attention_mask(
    attention: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize: bool = False
) -> torch.Tensor:
    """
    Zero out rows and columns corresponding to padding tokens.

    CRITICAL: Must apply BEFORE centrality computation to avoid
    padding tokens acting as attention sinks.

    Auto-detects 3D (batch, seq, seq) or 4D (batch, heads, seq, seq) tensors.

    Args:
        attention: Attention tensor of shape (B, N, N) or (B, H, N, N)
        attention_mask: (batch, seq) with 1 for valid tokens, 0 for padding
        normalize: Whether to re-normalize rows after masking

    Returns:
        Masked attention tensor with padding rows/columns zeroed
    """
    if attention.dim() == 4:
        # (batch, heads, seq, seq)
        row_mask = attention_mask.unsqueeze(1).unsqueeze(-1).float()
        col_mask = attention_mask.unsqueeze(1).unsqueeze(-2).float()
    else:
        # (batch, seq, seq)
        row_mask = attention_mask.unsqueeze(-1).float()
        col_mask = attention_mask.unsqueeze(-2).float()

    masked = attention * row_mask * col_mask

    if normalize:
        row_sum = masked.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        masked = masked / row_sum

    return masked


def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Normalize tensor along dimension with numerical stability.

    Args:
        tensor: Input tensor to normalize
        dim: Dimension along which to normalize (default: -1)
        eps: Small constant to avoid division by zero

    Returns:
        Normalized tensor that sums to 1 along the specified dimension
    """
    total = tensor.sum(dim=dim, keepdim=True).clamp(min=eps)
    return tensor / total
