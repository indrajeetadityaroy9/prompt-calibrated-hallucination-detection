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


def _mahal_dists(signal_matrix: np.ndarray, mu: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = signal_matrix - mu
    return np.sum((diff @ precision) * diff, axis=1)


def _cusum(dists: np.ndarray, tau: float) -> np.ndarray:
    cusum = np.zeros(len(dists))
    for i in range(len(dists)):
        cusum[i] = max(0.0, (cusum[i - 1] if i > 0 else 0.0) + dists[i] - tau)
    return cusum


def calibrate_cusum(signal_matrix: np.ndarray) -> CalibrationStats:
    mu = signal_matrix.mean(axis=0)
    precision = np.linalg.inv(ledoit_wolf(signal_matrix)[0])
    dists = _mahal_dists(signal_matrix, mu, precision)
    tau = float(np.mean(dists))
    return CalibrationStats(mu=mu, precision=precision, tau=tau, h=max(float(np.max(_cusum(dists, tau))), float(EPS)))


def compute_cusum_risks(signal_matrix: np.ndarray, stats: CalibrationStats):
    dists = _mahal_dists(signal_matrix, stats.mu, stats.precision)
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
        spans = [RiskySpan(start=int(s), end=int(e), peak_cusum=float(cusum[s:e].max())) for s, e in zip(starts, ends)]
    return token_risks.tolist(), cusum.tolist(), float(np.max(token_risks)), bool(np.max(cusum) > stats.h), spans
