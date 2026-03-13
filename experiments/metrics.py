"""
Evaluation metrics for hallucination detection.

Computes AUROC, AUPRC, TPR@FPR, Calibration (ECE/Brier), and Selective Prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    f1_score as sklearn_f1_score,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    auroc: float
    auprc: float
    f1: float
    tpr_at_5_fpr: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    expected_calibration_error: float = 0.0
    brier_score: float = 0.0
    aurc: float = 0.0
    e_aurc: float = 0.0
    risk_at_90_coverage: float = 0.0


def compute_tpr_at_fpr(scores: list[float], labels: list[int], target_fpr: float = 0.05) -> float:
    """Compute TPR at a specific FPR threshold via linear interpolation."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def compute_calibration_error(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n = len(scores_arr)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (scores_arr >= bin_boundaries[i]) & (scores_arr <= bin_boundaries[i + 1])
        else:
            in_bin = (scores_arr >= bin_boundaries[i]) & (scores_arr < bin_boundaries[i + 1])

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_accuracy = np.mean(labels_arr[in_bin])
            bin_confidence = np.mean(scores_arr[in_bin])
            ece += (bin_size / n) * np.abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_metrics(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> MetricsResult:
    """Compute all evaluation metrics."""
    if len(set(labels)) < 2:
        raise ValueError(
            f"Cannot compute metrics with a single class (found only label={labels[0]}). "
            "Need both positive and negative examples."
        )
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    tpr_5 = compute_tpr_at_fpr(scores, labels, 0.05)
    ece = compute_calibration_error(scores, labels)
    brier = float(brier_score_loss(labels, scores))
    aurc_val = compute_aurc(scores, labels)
    e_aurc_val = compute_e_aurc(scores, labels)
    risk_90 = compute_risk_at_coverage(scores, labels, 0.9)

    preds = [1 if s >= threshold else 0 for s in scores]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1 = float(sklearn_f1_score(labels, preds, zero_division=0.0))

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


def compute_aurc(scores: list[float], labels: list[int]) -> float:
    """Area Under Risk-Coverage curve (AURC). Geifman & El-Yaniv (2017)."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    cumulative_errors = np.cumsum(sorted_labels)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)

    return float(np.trapezoid(risks, coverages))


def compute_e_aurc(scores: list[float], labels: list[int]) -> float:
    """Excess AURC (E-AURC = AURC - AURC_optimal)."""
    labels = np.array(labels)
    n = len(labels)
    n_errors = int(np.sum(labels))

    aurc = compute_aurc(scores, labels)

    k_start = n - n_errors + 1
    ks = np.arange(k_start, n + 1)
    aurc_optimal = float(np.sum((ks - (n - n_errors)) / ks) / n)

    return max(0.0, aurc - aurc_optimal)


def compute_risk_at_coverage(
    scores: list[float],
    labels: list[int],
    target_coverage: float = 0.9,
) -> float:
    """Compute risk (error rate) at a specific coverage level."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    k = min(max(1, int(target_coverage * n)), n)
    return float(np.sum(sorted_labels[:k]) / k)


def bootstrap_auroc_ci(
    scores: list[float],
    labels: list[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for AUROC."""
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    rng = np.random.default_rng(seed)
    aurocs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n)
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        if len(set(boot_labels)) < 2:
            continue

        aurocs.append(float(roc_auc_score(boot_labels, boot_scores)))

    alpha = 1 - confidence
    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))
