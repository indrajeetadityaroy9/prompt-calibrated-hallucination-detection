from dataclasses import dataclass

import numpy as np
from sklearn.covariance import ledoit_wolf

from src.config import RiskySpan


@dataclass
class CalibrationStats:
    mu: np.ndarray
    precision: np.ndarray
    tau: float
    h: float


def _cusum(dists: np.ndarray, tau: float) -> np.ndarray:
    out = np.empty(len(dists))
    c = 0.0
    for i, d in enumerate(dists):
        c = max(0.0, c + d - tau)
        out[i] = c
    return out


def calibrate_cusum(signal_matrix: np.ndarray) -> CalibrationStats:
    mu = signal_matrix.mean(axis=0)
    precision = np.linalg.inv(ledoit_wolf(signal_matrix)[0])
    diff = signal_matrix - mu
    dists = np.sum(diff @ precision * diff, axis=1)
    tau = float(dists.mean())
    cusum = _cusum(dists, tau)
    return CalibrationStats(mu=mu, precision=precision, tau=tau, h=float(cusum.max()))


def compute_cusum_risks(signal_matrix: np.ndarray, stats: CalibrationStats):
    diff = signal_matrix - stats.mu
    dists = np.sum(diff @ stats.precision * diff, axis=1)
    cusum = _cusum(dists, stats.tau)
    token_risks = cusum / (cusum + stats.h)
    above = cusum > stats.h
    n = len(dists)
    changes = np.diff(above.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [n]])
    spans = [RiskySpan(int(s), int(e), float(cusum[s:e].max())) for s, e in zip(starts, ends)]
    return token_risks.tolist(), cusum.tolist(), float(token_risks.max()), bool(cusum.max() > stats.h), spans
