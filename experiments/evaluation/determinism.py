"""
Global determinism utilities for reproducible experiments.

CRITICAL: Call set_global_seed() at process entry point BEFORE
loading any models or data. This ensures all random operations
(numpy, torch, python random) produce identical results.
"""

import os
import random

import numpy as np
import torch


# Default seed for all experiments
DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """
    Set global seed for all random number generators.

    MUST be called at process entry point before any model or data loading.
    This ensures reproducibility across:
    - Python random module
    - NumPy random
    - PyTorch CPU and CUDA

    Args:
        seed: Random seed (default: 42)
        deterministic: If True, enable full deterministic mode including
                       cuDNN settings. This ensures bit-exact reproducibility
                       but may reduce performance by 10-20%.

    Note:
        When deterministic=True, also sets:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        - torch.use_deterministic_algorithms(True) where supported
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # For extra determinism (may impact performance)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Full deterministic mode for bit-exact reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms globally (PyTorch 1.8+)
        # Use warn_only=True (PyTorch 1.11+) to avoid errors from operations
        # that don't have deterministic implementations (e.g., scatter, Triton kernels)
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                # Try warn_only mode first (PyTorch 1.11+)
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch without warn_only parameter - skip to avoid runtime errors
                import warnings
                warnings.warn(
                    "PyTorch version does not support warn_only mode for deterministic algorithms. "
                    "Skipping torch.use_deterministic_algorithms() to avoid runtime errors.",
                    RuntimeWarning,
                )


def get_current_seed() -> int:
    """
    Get the current seed from environment.

    Returns:
        Seed value from PYTHONHASHSEED or DEFAULT_SEED if not set.
    """
    return int(os.environ.get("PYTHONHASHSEED", DEFAULT_SEED))
