"""
Metrics calculation with bootstrap confidence intervals.

Implements AUROC, AUPRC, F1, and other classification metrics
with proper statistical uncertainty quantification.
"""

from typing import Dict, List, Tuple, Callable, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


class MetricsCalculator:
    """
    Compute classification metrics with bootstrap confidence intervals.

    Provides rigorous statistical uncertainty for all metrics,
    essential for publication-quality results.

    Example:
        >>> calc = MetricsCalculator(bootstrap_samples=1000, confidence_level=0.95)
        >>> metrics, ci_bounds = calc.compute_all(labels, scores, ["auroc", "auprc"])
        >>> print(f"AUROC: {metrics['auroc']:.4f} [{ci_bounds['auroc'][0]:.4f}, {ci_bounds['auroc'][1]:.4f}]")
    """

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ):
        """
        Initialize metrics calculator.

        Args:
            bootstrap_samples: Number of bootstrap resamples for CI
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            seed: Random seed for reproducibility
        """
        self.n_bootstrap = bootstrap_samples
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(seed)

        # Define metric functions
        self._metric_fns: Dict[str, Callable] = {
            "auroc": self._auroc,
            "auprc": self._auprc,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "accuracy": self._accuracy,
        }

    def _auroc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUROC."""
        if len(np.unique(y_true)) < 2:
            return 0.5  # Undefined for single class
        return roc_auc_score(y_true, y_scores)

    def _auprc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute AUPRC (Average Precision)."""
        if len(np.unique(y_true)) < 2:
            return np.mean(y_true)  # Baseline for single class
        return average_precision_score(y_true, y_scores)

    def _f1(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute F1 at median threshold."""
        y_pred = (y_scores > np.median(y_scores)).astype(int)
        return f1_score(y_true, y_pred, zero_division=0)

    def _precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute precision at median threshold."""
        y_pred = (y_scores > np.median(y_scores)).astype(int)
        return precision_score(y_true, y_pred, zero_division=0)

    def _recall(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute recall at median threshold."""
        y_pred = (y_scores > np.median(y_scores)).astype(int)
        return recall_score(y_true, y_pred, zero_division=0)

    def _accuracy(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute accuracy at median threshold."""
        y_pred = (y_scores > np.median(y_scores)).astype(int)
        return accuracy_score(y_true, y_pred)

    def compute_all(
        self,
        labels: List[int],
        scores: List[float],
        metric_names: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
        """
        Compute all requested metrics with confidence intervals.

        Args:
            labels: Ground truth labels (0/1)
            scores: Predicted uncertainty scores
            metric_names: List of metrics to compute

        Returns:
            Tuple of (metrics_dict, ci_bounds_dict)
            - metrics_dict: {metric_name: point_estimate}
            - ci_bounds_dict: {metric_name: (lower_bound, upper_bound)}
        """
        labels = np.array(labels)
        scores = np.array(scores)

        if len(labels) == 0:
            return {}, {}

        metrics = {}
        ci_bounds = {}

        for name in metric_names:
            if name not in self._metric_fns:
                continue

            metric_fn = self._metric_fns[name]

            try:
                point, lo, hi = self._bootstrap_ci(labels, scores, metric_fn)
                metrics[name] = point
                ci_bounds[name] = (lo, hi)
            except Exception:
                # Fallback for edge cases
                metrics[name] = 0.0
                ci_bounds[name] = (0.0, 0.0)

        return metrics, ci_bounds

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric_fn: Callable,
    ) -> Tuple[float, float, float]:
        """
        Compute bootstrap confidence interval for a metric.

        Args:
            y_true: Ground truth labels
            y_scores: Predicted scores
            metric_fn: Function (y_true, y_scores) -> float

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        n = len(y_true)

        # Point estimate
        try:
            point = metric_fn(y_true, y_scores)
        except Exception:
            point = 0.0

        # Bootstrap resampling
        bootstrap_scores = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            try:
                score = metric_fn(y_true[idx], y_scores[idx])
                bootstrap_scores.append(score)
            except Exception:
                # Skip failed samples (e.g., single class in resample)
                continue

        if not bootstrap_scores:
            return point, point, point

        # Percentile method for CI
        alpha = 1 - self.confidence_level
        lo = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        hi = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

        return point, lo, hi

    def compute_confident_subset(
        self,
        labels: List[int],
        scores: List[float],
        confidences: List[Optional[float]],
        threshold: float,
        metric_names: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]], int]:
        """
        Compute metrics on confident subset (model confidence > threshold).

        Args:
            labels: Ground truth labels
            scores: Uncertainty scores
            confidences: Model confidence scores (may contain None)
            threshold: Confidence threshold
            metric_names: Metrics to compute

        Returns:
            Tuple of (metrics, ci_bounds, n_confident)
        """
        # Filter to confident samples
        indices = [
            i
            for i, conf in enumerate(confidences)
            if conf is not None and conf > threshold
        ]

        if len(indices) < 10:
            return {}, {}, len(indices)

        conf_labels = [labels[i] for i in indices]
        conf_scores = [scores[i] for i in indices]

        metrics, ci_bounds = self.compute_all(conf_labels, conf_scores, metric_names)

        return metrics, ci_bounds, len(indices)
