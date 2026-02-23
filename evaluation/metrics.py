"""
Evaluation metrics for hallucination detection.

Computes AUROC, AUPRC, TPR@FPR, Calibration (ECE/Brier), and Selective Prediction.
"""

from typing import List, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    auroc: float
    auprc: float
    f1: float

    # Safety metrics
    tpr_at_5_fpr: float

    # Counts
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Calibration
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0

    # Selective prediction / Risk-Coverage
    aurc: float = 0.0
    e_aurc: float = 0.0
    risk_at_90_coverage: float = 0.0


def compute_tpr_at_fpr(scores: List[float], labels: List[int], target_fpr: float = 0.05) -> float:
    """Compute TPR at a specific FPR threshold."""
    scores = np.array(scores)
    labels = np.array(labels)

    n_neg = np.sum(labels == 0)
    n_pos = np.sum(labels == 1)

    if n_neg == 0 or n_pos == 0: return 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores)

    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0

    idx = valid_indices[-1]
    return float(tpr[idx])

def compute_calibration_error(scores: List[float], labels: List[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (scores >= bin_lower) & (scores <= bin_upper)
        else:
            in_bin = (scores >= bin_lower) & (scores < bin_upper)

        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_accuracy = np.mean(labels[in_bin])
            bin_confidence = np.mean(scores[in_bin])
            ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)

    return float(ece)

def compute_metrics(
    scores: List[float],
    labels: List[int],
    threshold: float = 0.5,
) -> MetricsResult:
    """Compute all evaluation metrics."""
    # Basic ranking metrics
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    # Safety metric
    tpr_5 = compute_tpr_at_fpr(scores, labels, 0.05)

    # Calibration metrics
    ece = compute_calibration_error(scores, labels)
    brier = float(brier_score_loss(labels, scores))

    # Selective prediction metrics
    aurc_val = compute_aurc(scores, labels)
    e_aurc_val = compute_e_aurc(scores, labels)
    risk_90 = compute_risk_at_coverage(scores, labels, 0.9)

    # Classification metrics at threshold
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return MetricsResult(
        auroc=auroc,
        auprc=auprc,
        f1=f1,
        tpr_at_5_fpr=tpr_5,
        expected_calibration_error=ece,
        brier_score=brier,
        aurc=aurc_val,
        e_aurc=e_aurc_val,
        risk_at_90_coverage=risk_90,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )


def compute_aurc(scores: List[float], labels: List[int]) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).
    Lower is better — indicates better uncertainty estimation.
    Reference: Geifman & El-Yaniv (2017)
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n = len(scores)

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    cumulative_errors = np.cumsum(sorted_labels)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)

    aurc = np.trapezoid(risks, coverages)

    return float(aurc)


def compute_e_aurc(scores: List[float], labels: List[int]) -> float:
    """Compute Excess AURC (E-AURC = AURC - AURC_optimal)."""
    labels = np.array(labels)
    n = len(labels)
    n_errors = int(np.sum(labels))

    if n_errors == 0 or n_errors == n:
        return 0.0

    aurc = compute_aurc(scores, labels)

    k_start = n - n_errors + 1
    ks = np.arange(k_start, n + 1)
    aurc_optimal = float(np.sum((ks - (n - n_errors)) / ks) / n)

    return float(max(0.0, aurc - aurc_optimal))


def compute_risk_at_coverage(
    scores: List[float],
    labels: List[int],
    target_coverage: float = 0.9,
) -> float:
    """Compute risk (error rate) at a specific coverage level."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    k = max(1, int(target_coverage * n))
    k = min(k, n)

    risk = np.sum(sorted_labels[:k]) / k

    return float(risk)


def bootstrap_auroc_ci(
    scores: List[float],
    labels: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for AUROC."""
    if len(set(labels)) < 2:
        return (0.0, 1.0)

    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    rng = np.random.RandomState(seed)
    aurocs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        if len(set(boot_labels)) < 2:
            continue

        auroc = float(roc_auc_score(boot_labels, boot_scores))
        aurocs.append(auroc)

    if not aurocs:
        return (0.0, 1.0)

    alpha = 1 - confidence
    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))
