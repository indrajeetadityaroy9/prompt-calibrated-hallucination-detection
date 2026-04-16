import math

import numpy as np
import torch
from torch import Tensor


def marchenko_pastur_edge(sigma2: float, gamma: float) -> float:
    return sigma2 * (1.0 + gamma ** 0.5) ** 2


def information_flow_regularity(fi_profile: Tensor) -> float:
    fi = fi_profile.float()
    l1 = fi.sum()
    return float(l1 / (l1 + (fi[1:] - fi[:-1]).abs().sum()))


def effective_rank(S: Tensor) -> int:
    p = S.float() / S.float().sum()
    return math.ceil((-torch.xlogy(p, p).sum()).exp().item())


def otsu(values: np.ndarray | list) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    idx = np.arange(1, n)
    w0 = idx / n
    mu0 = cumsum[:-1] / idx
    mu1 = (cumsum[-1] - cumsum[:-1]) / (n - idx)
    between = w0 * (1.0 - w0) * (mu0 - mu1) ** 2
    best = int(np.argmax(between))
    return 0.5 * (sorted_vals[best] + sorted_vals[best + 1]), float(between[best] / np.var(values))
