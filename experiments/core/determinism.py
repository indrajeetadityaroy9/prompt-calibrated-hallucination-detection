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
