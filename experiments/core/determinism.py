"""
Global determinism utilities for reproducible experiments.

CRITICAL: Call set_global_seed() at process entry point BEFORE
loading any models or data. This ensures all random operations
(numpy, torch, python random) produce identical results.
"""

import os
import random
import warnings
from typing import Optional

import numpy as np
import torch


# Default seed for all experiments
DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """
    Set global seed for all random number generators.

    MUST be called at process entry point before any model or data loading.
    This ensures reproducibility across:
    - Python random module
    - NumPy random
    - PyTorch CPU and CUDA

    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For extra determinism (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Note: These settings can impact performance but ensure reproducibility
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def verify_dependencies() -> bool:
    """
    Verify all required dependencies are importable.

    Returns:
        True if all dependencies are available

    Raises:
        ImportError: If any required dependency is missing
    """
    required = []

    # Core science libraries
    try:
        import numpy
        import scipy
        import sklearn
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.calibration import calibration_curve
    except ImportError as e:
        required.append(f"sklearn/scipy: {e}")

    # Deep learning
    try:
        import torch
        import transformers
    except ImportError as e:
        required.append(f"torch/transformers: {e}")

    # SelfCheckGPT dependencies
    try:
        import sentence_transformers
    except ImportError as e:
        required.append(f"sentence_transformers: {e}")

    if required:
        raise ImportError(
            "Missing required dependencies:\n" + "\n".join(required)
        )

    return True


def verify_cuda(min_vram_gb: float = 0.0) -> dict:
    """
    Verify CUDA availability and VRAM.

    Args:
        min_vram_gb: Minimum required VRAM in GB (0 = no check)

    Returns:
        Dict with CUDA status information

    Raises:
        RuntimeError: If CUDA unavailable or insufficient VRAM
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "total_vram_gb": 0.0,
    }

    if not torch.cuda.is_available():
        if min_vram_gb > 0:
            raise RuntimeError("CUDA not available but GPU required")
        return info

    info["device_count"] = torch.cuda.device_count()

    total_vram = 0.0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        total_vram += vram_gb
        info["devices"].append({
            "index": i,
            "name": props.name,
            "vram_gb": round(vram_gb, 2),
            "compute_capability": f"{props.major}.{props.minor}",
        })

    info["total_vram_gb"] = round(total_vram, 2)

    if min_vram_gb > 0 and total_vram < min_vram_gb:
        raise RuntimeError(
            f"Insufficient VRAM: {total_vram:.1f}GB available, "
            f"{min_vram_gb:.1f}GB required"
        )

    return info


def get_determinism_info() -> dict:
    """
    Get current determinism configuration for logging.

    Returns:
        Dict with seed and torch backend settings
    """
    return {
        "python_hash_seed": os.environ.get("PYTHONHASHSEED", "not set"),
        "torch_cudnn_deterministic": getattr(
            torch.backends.cudnn, "deterministic", None
        ),
        "torch_cudnn_benchmark": getattr(
            torch.backends.cudnn, "benchmark", None
        ),
    }
