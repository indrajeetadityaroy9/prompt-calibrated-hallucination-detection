"""
Split conformal prediction for hallucination risk scores.

Provides finite-sample coverage guarantees for the detection threshold,
replacing the fixed response_flag_threshold with a statistically principled
calibrated threshold.

Reference: "Mitigating LLM Hallucinations via Conformal Abstention" (arXiv:2405.01563)
"""

from typing import Tuple

import numpy as np


class ConformalCalibrator:
    """Split conformal prediction for hallucination risk scores."""

    def __init__(self, alpha: float = 0.10):
        """
        Initialize conformal calibrator.

        Args:
            alpha: Target miscoverage rate (0.10 = 90% coverage guarantee).
        """
        self.alpha = alpha
        self._threshold: float = None

    def calibrate(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute conformal quantile from calibration data.

        Uses non-hallucinated examples (label=0) to set the threshold:
        q_hat = quantile of scores[label==0] at level ceil((1-alpha)(n+1))/n.

        Args:
            scores: Risk scores from calibration set.
            labels: Ground truth labels (1=hallucination, 0=faithful).

        Returns:
            Calibrated threshold q_hat.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)

        # Non-conformity scores from faithful examples
        faithful_scores = scores[labels == 0]
        n = len(faithful_scores)

        if n == 0:
            self._threshold = 0.5
            return self._threshold

        # Finite-sample corrected quantile level
        q_level = min(1.0, np.ceil((1 - self.alpha) * (n + 1)) / n)

        self._threshold = float(np.quantile(faithful_scores, q_level))
        return self._threshold

    @property
    def threshold(self) -> float:
        """Return calibrated threshold (raises if not calibrated)."""
        if self._threshold is None:
            raise RuntimeError("Call calibrate() before accessing threshold.")
        return self._threshold

    def predict(self, score: float) -> Tuple[bool, float]:
        """
        Predict whether a score indicates hallucination.

        Args:
            score: Risk score for a single example.

        Returns:
            (is_flagged, threshold) tuple.
        """
        return score >= self.threshold, self.threshold

    def predict_batch(self, scores: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Batch prediction.

        Args:
            scores: Array of risk scores.

        Returns:
            (flags_array, threshold) tuple.
        """
        scores = np.asarray(scores)
        return scores >= self.threshold, self.threshold
