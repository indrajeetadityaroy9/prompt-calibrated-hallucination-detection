"""
Attention Entropy Dispersion (ENT) — head specialization bimodality signal.

ENT = 1 - Otsu coefficient of per-head normalized entropies.
Range [0,1], higher = less bimodal = riskier.
"""

import math

import numpy as np
from torch import Tensor

from ..numerics import EPS, otsu_coefficient


def compute_ent(attn_tensor: Tensor, seq_len: int) -> float:
    """ENT = 1 - otsu_coefficient(per-head normalized entropies).

    Args:
        attn_tensor: Stacked attention weights, shape (n_layers, n_heads, seq_len).
        seq_len: Current sequence length for entropy normalization.
    """
    log2_n = math.log2(seq_len)
    a = attn_tensor.float().clamp(min=EPS)
    H = -(a * a.log2()).sum(dim=-1) / log2_n  # (n_layers, n_heads)
    return float(np.clip(1.0 - otsu_coefficient(H.reshape(-1).cpu().numpy()), 0.0, 1.0))
