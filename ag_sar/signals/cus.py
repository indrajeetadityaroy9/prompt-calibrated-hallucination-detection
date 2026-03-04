"""
Context Utilization Score (CUS) — Lookback Ratio Bimodality signal for AG-SAR.

CUS = 1 - Otsu coefficient of lookback ratio vector.
Range [0,1], higher = more unimodal = riskier.
"""

import numpy as np
from torch import Tensor
from typing import Dict

from ..numerics import otsu_threshold, EPS


def compute_cus(attention_slices: Dict[int, Tensor], prompt_len: int) -> float:
    """CUS = 1 - otsu_coefficient(lookback_ratios).

    Collects per-head lookback ratios (context attention mass) across all layers,
    then measures bimodality via Otsu inter-class variance ratio.
    """
    lr_values = []
    for attn in attention_slices.values():
        for h in range(attn.shape[0]):
            lr_values.append(attn[h, :prompt_len].float().sum().item())

    lr = np.array(lr_values)
    total_var = float(np.var(lr))
    threshold = otsu_threshold(lr)
    mask_low = lr <= threshold
    mask_high = lr > threshold
    w1 = float(mask_low.mean())
    w2 = float(mask_high.mean())
    inter_var = w1 * w2 * (float(lr[mask_low].mean()) - float(lr[mask_high].mean())) ** 2
    return float(1.0 - inter_var / (total_var + EPS))
