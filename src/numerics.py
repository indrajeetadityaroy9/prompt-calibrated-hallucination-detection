import numpy as np
import torch
from torch import Tensor

EPS = float(torch.finfo(torch.float32).eps ** 2)


def marchenko_pastur_edge(sigma2: float, gamma: float) -> float:
    return sigma2 * (1.0 + gamma ** 0.5) ** 2


def information_flow_regularity(fi_profile: Tensor) -> float:
    fi = fi_profile.float().clamp(min=EPS)
    l1 = fi.sum()
    tv = (fi[1:] - fi[:-1]).abs().sum()
    return float(l1 / (l1 + tv))


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


def otsu_coefficient(values: np.ndarray | list) -> float:
    values = np.asarray(values, dtype=float)
    best_idx, _, between_var, total_var = _otsu_internals(values)
    return float(between_var[best_idx] / (total_var + EPS))
