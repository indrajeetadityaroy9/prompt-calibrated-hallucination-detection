from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap
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


def compute_tpr_at_fpr(scores: list[float], labels: list[int], *, target_fpr: float = 0.05) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))


def compute_calibration_error(scores: list[float], labels: list[int], *, n_bins: int = 10) -> float:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    n = len(scores_arr)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores_arr, bin_edges[1:-1]), 0, n_bins - 1)

    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    bin_acc_sums = np.bincount(bin_idx, weights=labels_arr, minlength=n_bins)
    bin_conf_sums = np.bincount(bin_idx, weights=scores_arr, minlength=n_bins)

    nonempty = bin_counts > 0
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    np.divide(bin_acc_sums, bin_counts, out=bin_acc, where=nonempty)
    np.divide(bin_conf_sums, bin_counts, out=bin_conf, where=nonempty)

    return float(np.sum((bin_counts / n) * np.abs(bin_acc - bin_conf)))


def compute_metrics(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.5,
) -> MetricsResult:
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
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    k = int(target_coverage * n)
    return float(np.sum(sorted_labels[:k]) / k)


def bootstrap_auroc_ci(
    scores: list[float],
    labels: list[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    def auroc_statistic(indices):
        return roc_auc_score(labels_arr[indices.astype(int)], scores_arr[indices.astype(int)])

    result = bootstrap(
        (np.arange(len(scores_arr)),),
        auroc_statistic,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        random_state=np.random.default_rng(seed),
        method="BCa",
    )
    return (float(result.confidence_interval.low), float(result.confidence_interval.high))
