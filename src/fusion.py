from dataclasses import dataclass

import numpy as np
from sklearn.covariance import ledoit_wolf

from src.config import RiskySpan
from src.numerics import EPS


@dataclass
class CalibrationStats:
    mu: np.ndarray
    precision: np.ndarray
    tau: float
    h: float


def _cusum(dists: np.ndarray, tau: float) -> np.ndarray:
    cusum = np.zeros(len(dists))
    for i in range(len(dists)):
        cusum[i] = max(0.0, (cusum[i - 1] if i > 0 else 0.0) + dists[i] - tau)
    return cusum


def calibrate_cusum(signal_matrix: np.ndarray) -> CalibrationStats:
    mu = signal_matrix.mean(axis=0)
    cov, _ = ledoit_wolf(signal_matrix)
    precision = np.linalg.inv(cov)
    diff = signal_matrix - mu
    dists = np.sum((diff @ precision) * diff, axis=1)
    tau = float(np.mean(dists))
    cusum = _cusum(dists, tau)
    return CalibrationStats(mu=mu, precision=precision, tau=tau, h=max(float(np.max(cusum)), float(EPS)))


def compute_cusum_risks(signal_matrix: np.ndarray, stats: CalibrationStats):
    diff = signal_matrix - stats.mu
    dists = np.sum((diff @ stats.precision) * diff, axis=1)
    n = len(dists)
    cusum = _cusum(dists, stats.tau)
    token_risks = cusum / (cusum + stats.h)
    above = cusum > stats.h
    spans = []
    if np.any(above):
        changes = np.diff(above.astype(int))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [n]])
        for s, e in zip(starts, ends):
            spans.append(RiskySpan(
                start=int(s), end=int(e), peak_cusum=float(cusum[s:e].max()),
            ))
    return (
        token_risks.tolist(),
        cusum.tolist(),
        float(np.max(token_risks)),
        bool(np.max(cusum) > stats.h),
        spans,
    )
