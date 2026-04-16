import math

import numpy as np
import torch
from torch import Tensor


def marchenko_pastur_edge(sigma2: float, gamma: float) -> float:
    return sigma2 * (1.0 + gamma ** 0.5) ** 2


def information_flow_regularity(fi_profile: Tensor) -> float:
    fi = fi_profile.float().clamp(min=torch.finfo(torch.float32).tiny)
    l1 = fi.sum()
    return float(l1 / (l1 + (fi[1:] - fi[:-1]).abs().sum()))


def effective_rank(S: Tensor) -> int:
    p = S.float().clamp(min=torch.finfo(torch.float32).tiny)
    p = p / p.sum()
    return math.ceil((-(p * p.log()).sum()).exp().item())


def _otsu_internals(values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, float]:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    indices = np.arange(1, n)
    w0 = indices / n
    mu0 = cumsum[:-1] / indices
    mu1 = (cumsum[-1] - cumsum[:-1]) / (n - indices)
    between_var = w0 * (1.0 - w0) * (mu0 - mu1) ** 2
    return int(np.argmax(between_var)), sorted_vals, between_var, float(np.var(values))


def otsu_coefficient(values: np.ndarray | list) -> float:
    values = np.asarray(values, dtype=float)
    best_idx, _, between_var, total_var = _otsu_internals(values)
    return float(between_var[best_idx] / (total_var + np.finfo(values.dtype).eps))
