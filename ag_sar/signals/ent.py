"""
Attention Entropy Dispersion (ENT) — head specialization bimodality signal.

ENT = 1 - Otsu coefficient of per-head normalized entropies.
Range [0,1], higher = less bimodal = riskier.
"""

import math

import numpy as np
from torch import Tensor

from ..numerics import EPS, otsu_coefficient


def compute_ent(attention_slices: dict[int, Tensor], seq_len: int) -> float:
    """ENT = 1 - otsu_coefficient(per-head normalized entropies).

    Collects per-head Shannon entropy of attention distributions across all layers,
    normalizes by log2(seq_len) to bound in [0, 1], then measures bimodality
    via Otsu inter-class variance ratio.
    """
    log2_n = math.log2(seq_len)
    entropies = []
    for attn in attention_slices.values():
        for h in range(attn.shape[0]):
            a = attn[h].float().clamp(min=EPS)
            H = -(a * a.log2()).sum().item() / log2_n
            entropies.append(H)

    return float(np.clip(1.0 - otsu_coefficient(entropies), 0.0, 1.0))
