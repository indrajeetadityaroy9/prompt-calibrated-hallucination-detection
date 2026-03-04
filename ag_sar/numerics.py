"""Numerical utilities: JSD, effective rank, Otsu threshold."""

import math
import numpy as np
import torch
from torch import Tensor

EPS = 1e-10


def jsd(p: Tensor, q: Tensor) -> float:
    """JSD(P||Q) in bits, [0, 1]. M = 0.5*(P+Q)."""
    m = 0.5 * (p + q)
    kl_pm = (torch.xlogy(p, p) - torch.xlogy(p, m)).sum()
    kl_qm = (torch.xlogy(q, q) - torch.xlogy(q, m)).sum()
    return float((0.5 * kl_pm + 0.5 * kl_qm).item() / math.log(2))


def effective_rank(S: Tensor) -> int:
    """Effective rank via Shannon entropy of normalized singular values.

    erank(S) = exp(H(p)) where p_i = s_i / sum(s_j).
    Continuous, parameter-free, in [1, len(S)].
    """
    S = S.float().clamp(min=EPS)
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return max(1, round(H.exp().item()))


def otsu_threshold(values) -> float:
    """Optimal bimodal threshold: argmax sigma_b^2(t). Otsu (1979)."""
    values = np.asarray(values, dtype=float)
    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    best_threshold = sorted_vals[0]
    best_variance = -1.0

    for i in range(1, n):
        w0 = i / n
        w1 = 1.0 - w0
        mu0 = sorted_vals[:i].mean()
        mu1 = sorted_vals[i:].mean()
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = 0.5 * (sorted_vals[i - 1] + sorted_vals[i])

    return float(best_threshold)
