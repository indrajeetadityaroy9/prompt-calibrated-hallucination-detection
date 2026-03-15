"""Numerical utilities: effective rank, Otsu threshold, Tracy-Widom CDF."""

import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor

# Numerical stability constant derived from float32 machine precision.
# float32 eps ≈ 1.19e-7; squaring gives a value that is:
#   (a) negligible relative to any float32 computation (eps² << eps)
#   (b) safely invertible within float32 range (1/eps² ≈ 7e13 << 3.4e38)
EPS = float(torch.finfo(torch.float32).eps ** 2)

# Tracy-Widom beta=1 exact moments (Tracy & Widom 1994, Bornemann 2010).
# These are mathematical constants of the TW₁ distribution, not parameters.
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
    # Cornish-Fisher third-cumulant correction (Cornish & Fisher, 1938):
    # z_cf = z - (skew / 3!) * (z² - 1)
    z_cf = z - (_TW1_SKEW / 6) * (z * z - 1.0)
    return float(norm.cdf(z_cf))


def effective_rank(S: Tensor) -> int:
    """Effective rank via Shannon entropy of normalized singular values.

    erank(S) = exp(H(p)) where p_i = s_i / sum(s_j).
    Continuous, parameter-free, in [1, len(S)].
    """
    S = S.float().clamp(min=EPS)
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return round(H.exp().item())


def _otsu_internals(values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Shared Otsu computation: returns (best_idx, sorted_vals, between_var, total_var)."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    cumsum = np.cumsum(sorted_vals)
    total_sum = cumsum[-1]

    indices = np.arange(1, n)
    w0 = indices / n
    w1 = 1.0 - w0
    mu0 = cumsum[:-1] / indices
    mu1 = (total_sum - cumsum[:-1]) / (n - indices)
    between_var = w0 * w1 * (mu0 - mu1) ** 2

    best_idx = int(np.argmax(between_var))
    return best_idx, sorted_vals, between_var, float(np.var(values))


def otsu_threshold(values: np.ndarray | list) -> float:
    """Optimal bimodal threshold: argmax sigma_b^2(t). Otsu (1979). Vectorized."""
    values = np.asarray(values, dtype=float)
    best_idx, sorted_vals, _, _ = _otsu_internals(values)
    return float(0.5 * (sorted_vals[best_idx] + sorted_vals[best_idx + 1]))


def otsu_coefficient(values: np.ndarray | list) -> float:
    """Otsu bimodality coefficient: max(sigma_b^2) / sigma_total^2. Range [0, 1]."""
    values = np.asarray(values, dtype=float)
    best_idx, _, between_var, total_var = _otsu_internals(values)
    return float(between_var[best_idx] / (total_var + EPS))
