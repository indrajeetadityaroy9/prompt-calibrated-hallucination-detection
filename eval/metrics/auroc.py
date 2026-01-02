"""
AUROC and AUPRC computation for hallucination detection.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def compute_auroc(
    labels: List[bool],
    scores: List[float],
    higher_score_is_positive: bool = True
) -> float:
    """
    Compute Area Under ROC Curve.

    Label Convention:
        - labels[i] = True means sample i IS a hallucination (positive class)
        - labels[i] = False means sample i is factual (negative class)
        - Higher AUROC means better hallucination detection

    Note: TruthfulQA dataset uses opposite convention (label=True means factual).
        The caller (exp3_auroc.py) inverts via: is_hallucination = not sample.label

    Args:
        labels: Ground truth labels (True = hallucination, False = factual)
        scores: Uncertainty scores from method
        higher_score_is_positive: If True, higher score means more uncertain

    Returns:
        AUROC score in [0, 1]
    """
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)

    if not higher_score_is_positive:
        y_score = -y_score

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)


def compute_auprc(
    labels: List[bool],
    scores: List[float],
    higher_score_is_positive: bool = True
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    More appropriate when classes are imbalanced.

    Args:
        labels: Ground truth labels
        scores: Uncertainty scores
        higher_score_is_positive: Score interpretation

    Returns:
        AUPRC score in [0, 1]
    """
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)

    if not higher_score_is_positive:
        y_score = -y_score

    return average_precision_score(y_true, y_score)


def compute_roc_curve(
    labels: List[bool],
    scores: List[float],
    higher_score_is_positive: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve points.

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)

    if not higher_score_is_positive:
        y_score = -y_score

    return roc_curve(y_true, y_score)


def compute_precision_recall_curve(
    labels: List[bool],
    scores: List[float],
    higher_score_is_positive: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve points.

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    y_true = np.array([1 if l else 0 for l in labels])
    y_score = np.array(scores)

    if not higher_score_is_positive:
        y_score = -y_score

    return precision_recall_curve(y_true, y_score)


def plot_roc_curve(
    results: Dict[str, Tuple[List[bool], List[float]]],
    title: str = "ROC Curve - Hallucination Detection",
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multiple methods.

    Args:
        results: Dict mapping method name to (labels, scores) tuple
        title: Plot title
        save_path: Path to save figure (None = display)
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (method, (labels, scores)) in enumerate(results.items()):
        fpr, tpr, _ = compute_roc_curve(labels, scores)
        auroc = compute_auroc(labels, scores)

        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            label=f'{method} (AUROC = {auroc:.3f})',
            linewidth=2
        )

    plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_optimal_threshold(
    labels: List[bool],
    scores: List[float],
    metric: str = 'youden'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        labels: Ground truth labels
        scores: Uncertainty scores
        metric: 'youden' (maximize TPR - FPR) or 'f1' (maximize F1)

    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    fpr, tpr, thresholds = compute_roc_curve(labels, scores)

    if metric == 'youden':
        # Youden's J statistic: maximize (TPR - FPR)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        return thresholds[optimal_idx], j_scores[optimal_idx]

    elif metric == 'f1':
        # F1 score
        precision, recall, thresholds_pr = compute_precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last (always 0)
        return thresholds_pr[optimal_idx], f1_scores[optimal_idx]

    else:
        raise ValueError(f"Unknown metric: {metric}")
