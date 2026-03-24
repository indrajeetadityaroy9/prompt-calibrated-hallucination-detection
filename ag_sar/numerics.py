import numpy as np
import torch
from scipy.stats import norm
from torch import Tensor

EPS = float(torch.finfo(torch.float32).eps ** 2)

_TW1_MU = -1.2065335745820
_TW1_SIGMA = 1.2680340580149
_TW1_SKEW = 0.29346452408


def tracy_widom_cdf(s: float) -> float:
    z = (s - _TW1_MU) / _TW1_SIGMA
    z_cf = z - (_TW1_SKEW / 6) * (z * z - 1.0)
    return float(norm.cdf(z_cf))


def effective_rank(S: Tensor) -> int:
    S = S.float().clamp(min=EPS)
    p = S / S.sum()
    H = -(p * p.log()).sum()
    return round(H.exp().item())


def _otsu_internals(values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, float]:
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
    values = np.asarray(values, dtype=float)
    best_idx, sorted_vals, _, _ = _otsu_internals(values)
    return float(0.5 * (sorted_vals[best_idx] + sorted_vals[best_idx + 1]))


def otsu_coefficient(values: np.ndarray | list) -> float:
    values = np.asarray(values, dtype=float)
    best_idx, _, between_var, total_var = _otsu_internals(values)
    return float(between_var[best_idx] / (total_var + EPS))
