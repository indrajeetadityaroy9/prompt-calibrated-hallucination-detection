"""
Base utilities for metrics computation.

Provides common functionality used across AUROC, calibration,
and other evaluation metrics.
"""

from typing import List, Tuple
import numpy as np


def compute_binary_metrics(
    labels: List[bool],
    scores: List[float],
    threshold: float
) -> Tuple[int, int, int, int]:
    """
    Compute binary classification metrics at a given threshold.

    Args:
        labels: Ground truth binary labels
        scores: Predicted scores (higher = positive class)
        threshold: Classification threshold

    Returns:
        Tuple of (TP, FP, TN, FN) counts
    """
    predictions = [s >= threshold for s in scores]
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    return tp, fp, tn, fn


def compute_precision_recall(
    labels: List[bool],
    scores: List[float],
    threshold: float
) -> Tuple[float, float]:
    """
    Compute precision and recall at a given threshold.

    Args:
        labels: Ground truth binary labels
        scores: Predicted scores
        threshold: Classification threshold

    Returns:
        Tuple of (precision, recall)
    """
    tp, fp, _, fn = compute_binary_metrics(labels, scores, threshold)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def compute_f1(
    labels: List[bool],
    scores: List[float],
    threshold: float
) -> float:
    """
    Compute F1 score at a given threshold.

    Args:
        labels: Ground truth binary labels
        scores: Predicted scores
        threshold: Classification threshold

    Returns:
        F1 score
    """
    precision, recall = compute_precision_recall(labels, scores, threshold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bins(
    scores: List[float],
    num_bins: int = 10
) -> List[Tuple[float, float]]:
    """
    Create evenly spaced bins for calibration/threshold analysis.

    Args:
        scores: Score values to bin
        num_bins: Number of bins

    Returns:
        List of (lower, upper) bin boundaries
    """
    min_score, max_score = min(scores), max(scores)
    bin_width = (max_score - min_score) / num_bins
    return [
        (min_score + i * bin_width, min_score + (i + 1) * bin_width)
        for i in range(num_bins)
    ]


def compute_thresholds(
    scores: List[float],
    num_thresholds: int = 100
) -> np.ndarray:
    """
    Generate thresholds spanning the score range.

    Args:
        scores: Score values
        num_thresholds: Number of thresholds

    Returns:
        Array of threshold values
    """
    min_s, max_s = min(scores), max(scores)
    # Add small margin to include endpoints
    margin = (max_s - min_s) * 0.01
    return np.linspace(min_s - margin, max_s + margin, num_thresholds)


def compute_roc_curve(
    labels: List[bool],
    scores: List[float],
    num_thresholds: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve (FPR, TPR) at multiple thresholds.

    Args:
        labels: Ground truth binary labels
        scores: Predicted scores
        num_thresholds: Number of threshold points

    Returns:
        Tuple of (fpr, tpr, thresholds) arrays
    """
    thresholds = compute_thresholds(scores, num_thresholds)
    fpr_list = []
    tpr_list = []

    for thresh in thresholds:
        tp, fp, tn, fn = compute_binary_metrics(labels, scores, thresh)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def normalize_scores(
    scores: List[float],
    method: str = 'minmax'
) -> List[float]:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Input scores
        method: 'minmax' or 'zscore'

    Returns:
        Normalized scores in [0, 1]
    """
    arr = np.array(scores)

    if method == 'minmax':
        min_s, max_s = arr.min(), arr.max()
        if max_s - min_s > 0:
            normalized = (arr - min_s) / (max_s - min_s)
        else:
            normalized = np.zeros_like(arr)
        return normalized.tolist()

    elif method == 'zscore':
        mean = arr.mean()
        std = arr.std() + 1e-10
        zscore = (arr - mean) / std
        # Convert to [0, 1] using sigmoid
        normalized = 1 / (1 + np.exp(-zscore))
        return normalized.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")


def invert_scores(scores: List[float]) -> List[float]:
    """
    Invert scores (for converting uncertainty to confidence).

    Higher input -> lower output (after normalization).

    Args:
        scores: Input scores

    Returns:
        Inverted scores
    """
    # Normalize first
    normalized = normalize_scores(scores, method='minmax')
    return [1.0 - s for s in normalized]
