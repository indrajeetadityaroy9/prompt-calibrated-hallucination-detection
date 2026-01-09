"""
Determinism utilities for reproducible experiments.

Sets random seeds across Python, NumPy, and PyTorch to ensure reproducibility.
"""

import os
import random

import numpy as np
import torch


DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (may reduce performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
