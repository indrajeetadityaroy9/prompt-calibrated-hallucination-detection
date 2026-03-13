"""
Context Utilization Score (CUS) — Lookback Ratio Bimodality signal for AG-SAR.

CUS = 1 - Otsu coefficient of lookback ratio vector.
Range [0,1], higher = more unimodal = riskier.
"""

import numpy as np
from torch import Tensor

from ..numerics import otsu_coefficient


def compute_cus(attention_slices: dict[int, Tensor], prompt_len: int) -> float:
    """CUS = 1 - otsu_coefficient(lookback_ratios).

    Collects per-head lookback ratios (context attention mass) across all layers,
    then measures bimodality via Otsu inter-class variance ratio.
    """
    lr_values = []
    for attn in attention_slices.values():
        for h in range(attn.shape[0]):
            lr_values.append(attn[h, :prompt_len].float().sum().item())

    return float(np.clip(1.0 - otsu_coefficient(lr_values), 0.0, 1.0))
