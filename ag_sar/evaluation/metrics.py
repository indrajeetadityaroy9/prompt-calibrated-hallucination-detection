"""
Evaluation metrics for hallucination detection.

Computes AUROC, AUPRC, TPR@FPR, Calibration (ECE/Brier), and Correlations.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    brier_score_loss,
    precision_recall_curve,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    auroc: float
    auprc: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    threshold: float

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

    # Correlation with ground truth (0/1)
    pearson: float = 0.0
    spearman: float = 0.0

    # Selective prediction / Risk-Coverage
    aurc: float = 0.0
    e_aurc: float = 0.0
    risk_at_90_coverage: float = 0.0


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Compute Area Under ROC Curve (Trapezoidal)."""
    return float(roc_auc_score(labels, scores))

def compute_auprc(scores: List[float], labels: List[int]) -> float:
    """Compute Area Under Precision-Recall Curve."""
    return float(average_precision_score(labels, scores))

def compute_tpr_at_fpr(scores: List[float], labels: List[int], target_fpr: float = 0.05) -> float:
    """
    Compute TPR at a specific FPR threshold.
    Crucial for 'Safety' (how many hallucinations do we catch if we accept 5% false alarms?).
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n_neg = np.sum(labels == 0)
    n_pos = np.sum(labels == 1)

    if n_neg == 0 or n_pos == 0: return 0.0

    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find index where FPR <= target_fpr
    # fpr is increasing. We want max tpr s.t. fpr <= target
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return 0.0

    idx = valid_indices[-1]
    return float(tpr[idx])

def compute_brier_score(scores: List[float], labels: List[int]) -> float:
    """
    Compute Brier Score (Mean Squared Error of probabilities).
    Lower is better.
    """
    return float(brier_score_loss(labels, scores))

def compute_calibration_error(scores: List[float], labels: List[int], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b, acc is accuracy, conf is mean confidence.
    """
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    # Create bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include lower bound, exclude upper (except for last bin which includes 1.0)
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

def compute_correlation(scores: List[float], labels: List[int]) -> Tuple[float, float]:
    """Compute Pearson and Spearman correlation with labels."""
    if len(set(labels)) < 2: return (0.0, 0.0)

    p_corr, _ = pearsonr(scores, labels)
    s_corr, _ = spearmanr(scores, labels)

    # Handle NaN
    if np.isnan(p_corr): p_corr = 0.0
    if np.isnan(s_corr): s_corr = 0.0

    return float(p_corr), float(s_corr)

def find_optimal_threshold(scores: List[float], labels: List[int], metric: str = "f1") -> float:
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)

    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5

def compute_metrics(
    scores: List[float],
    labels: List[int],
    threshold: float = 0.5,
    include_selective: bool = True,
) -> MetricsResult:
    """
    Compute all evaluation metrics.

    Args:
        scores: Predicted risk/confidence scores (higher = more likely hallucination)
        labels: Ground truth labels (1 = hallucination, 0 = faithful)
        threshold: Classification threshold (default 0.5)
        include_selective: Whether to compute AURC/E-AURC/Risk@90 (slower)

    Returns:
        MetricsResult with all computed metrics
    """
    # Basic ranking metrics
    auroc = compute_auroc(scores, labels)
    auprc = compute_auprc(scores, labels)

    # Safety metric
    tpr_5 = compute_tpr_at_fpr(scores, labels, 0.05)

    # Calibration metrics
    ece = compute_calibration_error(scores, labels)
    brier = compute_brier_score(scores, labels)

    # Correlation metrics
    pearson, spearman = compute_correlation(scores, labels)

    # Selective prediction metrics (AURC, E-AURC, Risk@90)
    aurc_val = 0.0
    e_aurc_val = 0.0
    risk_90 = 0.0
    if include_selective:
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
    accuracy = (tp + tn) / len(labels)

    return MetricsResult(
        auroc=auroc,
        auprc=auprc,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        threshold=threshold,
        tpr_at_5_fpr=tpr_5,
        expected_calibration_error=ece,
        brier_score=brier,
        pearson=pearson,
        spearman=spearman,
        aurc=aurc_val,
        e_aurc=e_aurc_val,
        risk_at_90_coverage=risk_90,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
    )

def compute_span_metrics(
    predicted_spans: List[Tuple[int, int]],
    gold_spans: List[Tuple[int, int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute span-level metrics with IoU matching."""
    def compute_iou(span1, span2):
        start1, end1 = span1
        start2, end2 = span2
        intersection = max(0, min(end1, end2) - max(start1, start2))
        union = max(end1, end2) - min(start1, start2)
        return intersection / union if union > 0 else 0

    matched_pred = set()
    matched_gold = set()

    for i, pred in enumerate(predicted_spans):
        for j, gold in enumerate(gold_spans):
            if j in matched_gold: continue
            if compute_iou(pred, gold) >= iou_threshold:
                matched_pred.add(i)
                matched_gold.add(j)
                break

    tp = len(matched_pred)
    fp = len(predicted_spans) - tp
    fn = len(gold_spans) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "matched_spans": tp,
        "total_predicted": len(predicted_spans),
        "total_gold": len(gold_spans),
    }


# =============================================================================
# Risk-Coverage Metrics (Selective Prediction)
# =============================================================================

def compute_risk_coverage(
    scores: List[float],
    labels: List[int],
    n_points: int = 100,
) -> List[Dict[str, float]]:
    """
    Compute Risk-Coverage curve data.

    Risk-Coverage evaluates selective prediction: as we increase the coverage
    (fraction of samples we choose to predict on), how does the risk (error rate) change?

    Lower scores = more confident = predict first.
    Risk = error rate on predicted samples.

    Args:
        scores: Confidence/risk scores (lower = more confident for abstention)
        labels: Ground truth labels (1 = positive/hallucination)
        n_points: Number of coverage points to compute

    Returns:
        List of dicts with 'coverage' and 'risk' keys
    """
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    # Sort by score ascending (lower score = more confident = predict first)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # Compute cumulative error (assuming prediction=0, error when label=1)
    # For hallucination detection: score predicts "is hallucination"
    # Risk = fraction of hallucinations among covered samples
    cumulative_positives = np.cumsum(sorted_labels)

    results = []
    coverages = np.linspace(0, 1, n_points + 1)[1:]  # Skip 0

    for cov in coverages:
        k = max(1, int(cov * n))
        risk = cumulative_positives[k - 1] / k if k > 0 else 0.0
        results.append({
            "coverage": float(cov),
            "risk": float(risk),
            "n_samples": k,
        })

    return results


def compute_aurc(scores: List[float], labels: List[int]) -> float:
    """
    Compute Area Under Risk-Coverage curve (AURC).

    AURC measures the quality of selective prediction.
    Lower is better - indicates better uncertainty estimation.

    Reference: Geifman & El-Yaniv (2017) "Selective Classification for Deep Neural Networks"

    Args:
        scores: Risk/uncertainty scores (higher = more uncertain)
        labels: Ground truth (1 = error/hallucination, 0 = correct/faithful)

    Returns:
        AURC value (lower is better)
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n = len(scores)

    # Sort by score ascending (lower score = more confident)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # Compute cumulative risk at each coverage level
    # Risk at coverage k/n = (sum of errors in first k samples) / k
    cumulative_errors = np.cumsum(sorted_labels)
    coverages = np.arange(1, n + 1) / n
    risks = cumulative_errors / np.arange(1, n + 1)

    # AURC = integral of risk over coverage using trapezoidal rule
    # Add (0, risk[0]) as starting point
    aurc = np.trapz(risks, coverages)

    return float(aurc)


def compute_e_aurc(scores: List[float], labels: List[int]) -> float:
    """
    Compute Excess AURC (E-AURC).

    E-AURC = AURC - AURC_optimal

    where AURC_optimal is achieved by an oracle that perfectly separates errors.
    E-AURC isolates the quality of uncertainty estimation from the base error rate.

    Reference: Geifman & El-Yaniv (2017)

    Args:
        scores: Risk/uncertainty scores
        labels: Ground truth labels

    Returns:
        E-AURC value (lower is better, 0 is optimal)
    """
    scores = np.array(scores)
    labels = np.array(labels)

    n = len(labels)
    n_errors = np.sum(labels)

    if n_errors == 0 or n_errors == n:
        return 0.0  # Perfect or all-wrong, no excess possible

    # Compute actual AURC
    aurc = compute_aurc(scores, labels)

    # Compute optimal AURC (oracle ranking: all correct first, then errors)
    # Optimal risk-coverage: risk=0 until coverage=(n-n_errors)/n, then increases
    # AURC_optimal = integral from (n-n_errors)/n to 1 of the rising risk

    # With oracle ordering: first (n - n_errors) samples are correct (risk=0)
    # then n_errors samples are errors (risk increases from 0 to n_errors/n)

    # Optimal AURC can be computed as: (n_errors/n) * (1 + n_errors) / (2*n)
    # Simplified: n_errors * (n_errors + 1) / (2 * n^2)
    error_rate = n_errors / n
    aurc_optimal = error_rate * (1 + 1/n) / 2

    # More precise formula: integrate risk from coverage = (n-n_errors)/n to 1
    # At coverage c where c > (n-n_errors)/n:
    # risk(c) = (c*n - (n-n_errors)) / (c*n) = 1 - (n-n_errors)/(c*n)

    # Actually, simpler exact formula:
    # AURC_opt = (1/n) * sum_{k=n-n_errors+1}^{n} (k - (n-n_errors)) / k
    k_start = n - n_errors + 1
    if k_start <= n:
        ks = np.arange(k_start, n + 1)
        risks_opt = (ks - (n - n_errors)) / ks
        aurc_optimal = np.sum(risks_opt) / n
    else:
        aurc_optimal = 0.0

    e_aurc = aurc - aurc_optimal
    return float(max(0.0, e_aurc))  # Ensure non-negative


def compute_risk_at_coverage(
    scores: List[float],
    labels: List[int],
    target_coverage: float = 0.9,
) -> float:
    """
    Compute risk (error rate) at a specific coverage level.

    Args:
        scores: Risk/uncertainty scores (higher = more uncertain)
        labels: Ground truth labels (1 = error)
        target_coverage: Target coverage (e.g., 0.9 for 90%)

    Returns:
        Risk (error rate) at the target coverage
    """
    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    # Sort by score ascending (lower = more confident)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    # Number of samples at target coverage
    k = max(1, int(target_coverage * n))
    k = min(k, n)

    # Risk = error rate in first k samples
    risk = np.sum(sorted_labels[:k]) / k

    return float(risk)


# =============================================================================
# Bootstrap Confidence Intervals
# =============================================================================

def bootstrap_auroc_ci(
    scores: List[float],
    labels: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for AUROC.

    Args:
        scores: Predicted scores
        labels: Ground truth labels
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(set(labels)) < 2:
        return (0.0, 1.0)

    scores = np.array(scores)
    labels = np.array(labels)
    n = len(scores)

    rng = np.random.RandomState(seed)
    aurocs = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        # Need both classes in bootstrap sample
        if len(set(boot_labels)) < 2:
            continue

        auroc = compute_auroc(boot_scores.tolist(), boot_labels.tolist())
        aurocs.append(auroc)

    if not aurocs:
        return (0.0, 1.0)

    alpha = 1 - confidence
    lower = np.percentile(aurocs, 100 * alpha / 2)
    upper = np.percentile(aurocs, 100 * (1 - alpha / 2))

    return (float(lower), float(upper))
