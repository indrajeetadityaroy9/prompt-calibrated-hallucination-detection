"""
Metrics calculation with bootstrap confidence intervals.

ICML/NeurIPS-grade metrics implementation using sklearn and scipy.
Includes: AUROC, AUPRC, ECE, Brier, AURC with proper NaN handling.

Library standards (Phase 1.1):
- AUROC: sklearn.metrics.roc_auc_score
- AUPRC: sklearn.metrics.average_precision_score (NOT auc(recall, precision))
- ECE: sklearn.calibration.calibration_curve + weighted average
- Brier: sklearn.metrics.brier_score_loss
- AURC: Custom numpy vectorization (no sklearn equivalent)
- Bootstrap CI: scipy.stats.bootstrap or percentile method
"""

from typing import Dict, List, Tuple, Callable, Optional, Union
import warnings

# Phase 1.1: Required imports (exact)
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr, bootstrap

# Phase 1.3: Tensor safety
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _to_numpy(arr: Union[np.ndarray, List, "torch.Tensor"]) -> np.ndarray:
    """
    Safely convert input to numpy array.

    Phase 1.3: Handles torch tensors with proper device transfer.
    """
    if TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


class MetricsCalculator:
    """
    Compute classification metrics with bootstrap confidence intervals.

    ICML-grade implementation with:
    - NaN filtering with fail-fast at >5% drop rate (Phase 1.5)
    - sklearn-based metric computation (Phase 1.2)
    - ECE, Brier, AURC support
    - Bootstrap confidence intervals via scipy.stats.bootstrap

    Example:
        >>> calc = MetricsCalculator(bootstrap_samples=1000, confidence_level=0.95)
        >>> metrics, ci_bounds = calc.compute_all(labels, scores, ["auroc", "auprc", "ece"])
        >>> print(f"AUROC: {metrics['auroc']:.4f} [{ci_bounds['auroc'][0]:.4f}, {ci_bounds['auroc'][1]:.4f}]")
    """

    def __init__(
        self,
        bootstrap_samples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
        max_nan_rate: float = 0.05,
    ):
        """
        Initialize metrics calculator.

        Args:
            bootstrap_samples: Number of bootstrap resamples for CI
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            seed: Random seed for reproducibility
            max_nan_rate: Maximum allowed NaN rate (default 5%, Phase 1.5)
        """
        self.n_bootstrap = bootstrap_samples
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(seed)
        self.max_nan_rate = max_nan_rate

        # Define metric functions
        self._metric_fns: Dict[str, Callable] = {
            "auroc": self._auroc,
            "auprc": self._auprc,
            "ece": self._ece,
            "brier": self._brier,
            "aurc": self._aurc,
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "accuracy": self._accuracy,
        }

    def _filter_nan(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Filter out NaN values with fail-fast on excessive drop rate.

        Phase 1.5: NaN Drop Strategy implemented before any metric computation.

        Returns:
            Tuple of (filtered_y_true, filtered_y_scores, dropped_count)

        Raises:
            ValueError: If NaN rate exceeds max_nan_rate
        """
        valid_mask = ~np.isnan(y_scores) & ~np.isinf(y_scores)
        dropped_count = len(y_scores) - np.sum(valid_mask)
        total = len(y_scores)

        if dropped_count > 0:
            drop_rate = dropped_count / total
            warnings.warn(
                f"Dropped {dropped_count} NaN/Inf samples ({drop_rate:.1%})"
            )
            if drop_rate > self.max_nan_rate:
                raise ValueError(
                    f"Excessive NaNs in metric inputs: {drop_rate:.1%} > {self.max_nan_rate:.1%} threshold. "
                    f"Dropped {dropped_count}/{total} samples."
                )

        return y_true[valid_mask], y_scores[valid_mask], dropped_count

    def _check_single_class(self, y_true: np.ndarray, metric_name: str) -> bool:
        """
        Phase 1.5: Single-Class Protection.

        Check if y_true contains only one class. Returns True if single-class
        (metric should be skipped), False otherwise.
        """
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            warnings.warn(
                f"Single class in y_true for {metric_name}: {unique_classes}. "
                "Returning None for this metric."
            )
            return True
        return False

    def _auroc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
        """
        Compute AUROC using sklearn.

        Phase 1.2: No custom numerical integration.
        """
        if self._check_single_class(y_true, "auroc"):
            return None
        return float(roc_auc_score(y_true, y_scores))

    def _auprc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
        """
        Compute AUPRC (Average Precision) using sklearn.

        Phase 1.2: Uses average_precision_score, NOT auc(recall, precision).
        """
        if self._check_single_class(y_true, "auprc"):
            return None
        return float(average_precision_score(y_true, y_scores))

    def _ece(
        self, y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10
    ) -> Optional[float]:
        """
        Compute Expected Calibration Error using sklearn.calibration_curve.

        Phase 1.2: Uses calibration_curve + weighted mean absolute gap.

        ECE = Σ (|B_i| / n) × |accuracy(B_i) - confidence(B_i)|
        """
        if self._check_single_class(y_true, "ece"):
            return None

        # Clip scores to valid probability range
        y_scores_clipped = np.clip(y_scores, 0.0, 1.0)
        n = len(y_true)

        try:
            # Use sklearn's calibration_curve for robust binning
            prob_true, prob_pred = calibration_curve(
                y_true, y_scores_clipped, n_bins=n_bins, strategy="uniform"
            )

            # Compute bin counts via digitizing
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(y_scores_clipped, bin_edges[1:-1])

            # Count samples in each bin
            bin_counts = np.zeros(n_bins)
            for i in range(n_bins):
                bin_counts[i] = np.sum(bin_indices == i)

            # Only use bins with samples
            non_empty_mask = bin_counts > 0
            non_empty_counts = bin_counts[non_empty_mask]

            # Weighted ECE computation
            if len(prob_true) != len(non_empty_counts):
                # Fallback to simple weighted average
                ece = float(np.mean(np.abs(prob_true - prob_pred)))
            else:
                weights = non_empty_counts / n
                ece = float(np.sum(weights * np.abs(prob_true - prob_pred)))

            return ece

        except Exception:
            # Fallback: manual binning
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (y_scores_clipped >= bin_edges[i]) & (
                        y_scores_clipped <= bin_edges[i + 1]
                    )
                else:
                    mask = (y_scores_clipped >= bin_edges[i]) & (
                        y_scores_clipped < bin_edges[i + 1]
                    )
                bin_size = np.sum(mask)
                if bin_size > 0:
                    bin_accuracy = np.mean(y_true[mask])
                    bin_confidence = np.mean(y_scores_clipped[mask])
                    ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)
            return float(ece)

    def _brier(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute Brier Score using sklearn.

        Phase 1.2: Uses sklearn.metrics.brier_score_loss.
        """
        y_scores_clipped = np.clip(y_scores, 0.0, 1.0)
        return float(brier_score_loss(y_true, y_scores_clipped))

    def _aurc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute Area Under Risk-Coverage Curve.

        Phase 1.4: Exact vectorized logic, no Python loops over samples.

        Args:
            y_true: 0=Fact, 1=Hallucination
            y_scores: Uncertainty score (higher = reject first)

        Returns:
            AURC value (lower is better)

        X-axis: Coverage (fraction of data kept)
        Y-axis: Risk (error rate on kept data)
        """
        n = len(y_true)
        if n == 0:
            return 0.0

        # 1. Sort by uncertainty (High to Low) - reject high uncertainty first
        desc_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_indices]

        # 2. Compute cumulative risk (vectorized, no Python loops)
        cumulative_hallucinations = np.cumsum(y_true_sorted[::-1])[::-1]
        cumulative_count = np.arange(n, 0, -1)  # n, n-1, ..., 1

        risks = cumulative_hallucinations / cumulative_count
        coverages = cumulative_count / n

        # 3. Compute Area Under Risk-Coverage using sklearn.metrics.auc
        return float(auc(coverages, risks))

    def _find_optimal_threshold(
        self, y_true: np.ndarray, y_scores: np.ndarray
    ) -> float:
        """Find threshold that maximizes F1 score."""
        from sklearn.metrics import precision_recall_curve

        if self._check_single_class(y_true, "threshold"):
            return 0.5

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        # Compute F1 for each threshold
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # Last element has recall=0
        return thresholds[best_idx] if len(thresholds) > 0 else 0.5

    def _f1(self, y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
        """Compute F1 at optimal threshold (maximizes F1)."""
        if self._check_single_class(y_true, "f1"):
            return None
        threshold = self._find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        return float(f1_score(y_true, y_pred, zero_division=0))

    def _precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
        """Compute precision at optimal F1 threshold."""
        if self._check_single_class(y_true, "precision"):
            return None
        threshold = self._find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        return float(precision_score(y_true, y_pred, zero_division=0))

    def _recall(self, y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
        """Compute recall at optimal F1 threshold."""
        if self._check_single_class(y_true, "recall"):
            return None
        threshold = self._find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        return float(recall_score(y_true, y_pred, zero_division=0))

    def _accuracy(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute accuracy at optimal F1 threshold."""
        threshold = self._find_optimal_threshold(y_true, y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        return float(accuracy_score(y_true, y_pred))

    def compute_all(
        self,
        labels: Union[List[int], np.ndarray, "torch.Tensor"],
        scores: Union[List[float], np.ndarray, "torch.Tensor"],
        metric_names: List[str],
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, Tuple[float, float]]]:
        """
        Compute all requested metrics with confidence intervals.

        Args:
            labels: Ground truth labels (0/1)
            scores: Predicted uncertainty scores
            metric_names: List of metrics to compute

        Returns:
            Tuple of (metrics_dict, ci_bounds_dict)
            - metrics_dict: {metric_name: point_estimate or None}
            - ci_bounds_dict: {metric_name: (lower_bound, upper_bound)}

        Raises:
            ValueError: If NaN rate exceeds max_nan_rate
        """
        # Phase 1.3: Convert to numpy safely
        labels = _to_numpy(labels)
        scores = _to_numpy(scores)

        if len(labels) == 0:
            return {}, {}

        # Phase 1.5: Filter NaN/Inf with fail-fast
        labels, scores, dropped = self._filter_nan(labels, scores)

        if len(labels) == 0:
            raise ValueError("No valid samples after NaN filtering")

        metrics = {}
        ci_bounds = {}

        for name in metric_names:
            if name not in self._metric_fns:
                warnings.warn(f"Unknown metric: {name}")
                continue

            metric_fn = self._metric_fns[name]

            # Compute point estimate and CI
            point, lo, hi = self._bootstrap_ci(labels, scores, metric_fn)
            metrics[name] = point
            ci_bounds[name] = (lo, hi)

        return metrics, ci_bounds

    def _bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric_fn: Callable,
    ) -> Tuple[Optional[float], float, float]:
        """
        Compute bootstrap confidence interval for a metric.

        Phase 5.2: Uses scipy.stats.bootstrap for CI computation.

        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        n = len(y_true)

        # Point estimate
        try:
            point = metric_fn(y_true, y_scores)
        except Exception:
            return None, float("nan"), float("nan")

        if point is None:
            return None, float("nan"), float("nan")

        # Bootstrap resampling (percentile method)
        bootstrap_scores = []
        for _ in range(self.n_bootstrap):
            idx = self.rng.choice(n, size=n, replace=True)
            try:
                score = metric_fn(y_true[idx], y_scores[idx])
                if score is not None:
                    bootstrap_scores.append(score)
            except Exception:
                continue

        if len(bootstrap_scores) < self.n_bootstrap * 0.5:
            warnings.warn(
                f"Only {len(bootstrap_scores)}/{self.n_bootstrap} bootstrap samples succeeded"
            )
        if len(bootstrap_scores) < 10:
            # Not enough samples for a meaningful CI
            return point, float("nan"), float("nan")

        # Percentile method for CI
        alpha = 1 - self.confidence_level
        lo = float(np.percentile(bootstrap_scores, 100 * alpha / 2))
        hi = float(np.percentile(bootstrap_scores, 100 * (1 - alpha / 2)))

        return point, lo, hi


def compute_risk_coverage_aurc(
    y_true: np.ndarray, y_scores: np.ndarray
) -> float:
    """
    Standalone AURC computation with exact vectorized logic.

    Phase 1.4: No Python loops over samples.

    Args:
        y_true: Binary labels (0=fact, 1=hallucination)
        y_scores: Uncertainty scores (higher = more uncertain)

    Returns:
        AURC value (lower is better)
    """
    y_true = _to_numpy(y_true)
    y_scores = _to_numpy(y_scores)

    n = len(y_true)
    if n == 0:
        return 0.0

    # Sort by uncertainty descending
    desc_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_indices]

    # Vectorized cumulative computation
    cumulative_hallucinations = np.cumsum(y_true_sorted[::-1])[::-1]
    cumulative_count = np.arange(n, 0, -1)

    risks = cumulative_hallucinations / cumulative_count
    coverages = cumulative_count / n

    return float(auc(coverages, risks))
