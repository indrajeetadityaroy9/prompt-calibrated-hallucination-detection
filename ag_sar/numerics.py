"""Numerical utilities: JSD, effective rank, Otsu threshold, Tracy-Widom CDF."""

import math
import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor

EPS = 1e-10

# Tracy-Widom beta=1 exact moments (Tracy & Widom 1994, Bornemann 2010)
_TW1_MU = -1.2065335745820
_TW1_SIGMA = 1.2680340580149
_TW1_SKEW = 0.29346452408


def tracy_widom_cdf(s: float) -> float:
    """CDF of the Tracy-Widom beta=1 distribution via Cornish-Fisher expansion.

    Uses exact TW1 moments with Cornish-Fisher skewness correction to map
    through the Gaussian CDF. Accuracy < 0.005 for |standardized z| < 3,
    which covers the operating range of AG-SAR's spectral phase-transition score.

    Johnstone (2001): (lambda_max - mu_n) / sigma_n  ->  TW_1
    """
    z = (s - _TW1_MU) / _TW1_SIGMA
    # Cornish-Fisher correction for positive skewness (Cornish & Fisher, 1938)
    z_cf = z - (_TW1_SKEW / 6.0) * (z * z - 1.0)
    return float(norm.cdf(z_cf))


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


def otsu_threshold(values: np.ndarray | list) -> float:
    """Optimal bimodal threshold: argmax sigma_b^2(t). Otsu (1979). O(n)."""
    values = np.asarray(values, dtype=float)
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n < 2:
        return float(sorted_vals[0]) if n == 1 else 0.0

    cumsum = np.cumsum(sorted_vals)
    total_sum = cumsum[-1]

    best_threshold = sorted_vals[0]
    best_variance = -1.0

    for i in range(1, n):
        w0 = i / n
        w1 = 1.0 - w0
        mu0 = cumsum[i - 1] / i
        mu1 = (total_sum - cumsum[i - 1]) / (n - i)
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = 0.5 * (sorted_vals[i - 1] + sorted_vals[i])

    return float(best_threshold)
