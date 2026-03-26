from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import bootstrap
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score as sklearn_f1_score,
    roc_auc_score,
    roc_curve,
)
from torchmetrics.functional.classification import (
    binary_calibration_error,
    binary_specificity_at_sensitivity,
)


@dataclass
class MetricsResult:
    auroc: float
    auprc: float
    fpr_at_95_tpr: float
    f1_optimal: float
    optimal_threshold: float
    expected_calibration_error: float
    brier_score: float
    aurc: float
    e_aurc: float


def compute_metrics(scores: list[float], labels: list[int]) -> MetricsResult:
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)
    scores_t = torch.tensor(scores_arr, dtype=torch.float32)
    labels_t = torch.tensor(labels_arr, dtype=torch.long)

    auroc = float(roc_auc_score(labels_arr, scores_arr))
    auprc = float(average_precision_score(labels_arr, scores_arr))
    brier = float(brier_score_loss(labels_arr, scores_arr))

    specificity, _ = binary_specificity_at_sensitivity(scores_t, labels_t, min_sensitivity=0.95)
    fpr_at_95 = float(1.0 - specificity.item())

    fpr, tpr, thresholds = roc_curve(labels_arr, scores_arr)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])
    preds = (scores_arr >= optimal_threshold).astype(int)
    f1_optimal = float(sklearn_f1_score(labels_arr, preds, zero_division=0.0))

    ece = float(binary_calibration_error(scores_t, labels_t, n_bins=10, norm="l1"))

    sorted_idx = np.argsort(scores_arr)
    sorted_labels = labels_arr[sorted_idx]
    n = len(scores_arr)
    coverages = np.arange(1, n + 1) / n
    risks = np.cumsum(sorted_labels) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risks, coverages))

    n_errors = int(labels_arr.sum())
    k_start = n - n_errors + 1
    ks = np.arange(k_start, n + 1)
    aurc_optimal = float(np.sum((ks - (n - n_errors)) / ks) / n)
    e_aurc = float(aurc - aurc_optimal)

    return MetricsResult(
        auroc=auroc,
        auprc=auprc,
        fpr_at_95_tpr=fpr_at_95,
        f1_optimal=f1_optimal,
        optimal_threshold=optimal_threshold,
        expected_calibration_error=ece,
        brier_score=brier,
        aurc=aurc,
        e_aurc=e_aurc,
    )


def bootstrap_auroc_ci(
    scores: list[float],
    labels: list[int],
    *,
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
