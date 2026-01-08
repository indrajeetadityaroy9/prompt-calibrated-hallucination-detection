"""
Tensor Utilities - H100 optimizations and numerical helpers.
"""

import torch
import torch.nn as nn


def enable_tf32() -> None:
    """Enable TF32 for ~3x matmul speedup on Ampere/Hopper."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')


def enable_h100_optimizations() -> None:
    """Enable all Hopper-specific optimizations."""
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)


def is_tf32_enabled() -> bool:
    return (torch.cuda.is_available() and
            torch.backends.cuda.matmul.allow_tf32 and
            torch.backends.cudnn.allow_tf32)


def is_h100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def get_optimal_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float32


def optimize_for_inference() -> None:
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Numerically stable normalization (sum to 1)."""
    total = tensor.sum(dim=dim, keepdim=True).clamp(min=eps)
    return tensor / total


def apply_attention_mask(
    attention: torch.Tensor,
    attention_mask: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Apply attention mask by zeroing padded rows/columns."""
    if attention.dim() == 4:
        row_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
        col_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    else:
        row_mask = attention_mask.unsqueeze(-1)
        col_mask = attention_mask.unsqueeze(1)

    masked = attention * row_mask.float() * col_mask.float()

    if normalize:
        row_sum = masked.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        masked = masked / row_sum
    return masked


def get_model_dtype(model: nn.Module) -> torch.dtype:
    return next(model.parameters()).dtype


def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
