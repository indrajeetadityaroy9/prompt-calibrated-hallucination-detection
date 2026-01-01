"""
Calibration metrics for uncertainty estimation.

ECE (Expected Calibration Error) measures how well confidence
scores match actual accuracy.
"""

from typing import List, Tuple, Optional
import numpy as np


def compute_ece(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error.

    ECE = Σ (|bin| / n) * |accuracy(bin) - confidence(bin)|

    A perfectly calibrated model has ECE = 0.

    Args:
        confidences: Confidence scores in [0, 1]
        accuracies: Binary accuracy labels
        num_bins: Number of bins for calibration

    Returns:
        ECE score in [0, 1] (lower is better)
    """
    confidences = np.array(confidences)
    accuracies = np.array([1 if a else 0 for a in accuracies])

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.sum() / total_samples

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return ece


def compute_mce(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error.

    MCE = max over bins of |accuracy(bin) - confidence(bin)|

    Args:
        confidences: Confidence scores
        accuracies: Binary accuracy labels
        num_bins: Number of bins

    Returns:
        MCE score in [0, 1]
    """
    confidences = np.array(confidences)
    accuracies = np.array([1 if a else 0 for a in accuracies])

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    max_error = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            error = abs(avg_accuracy - avg_confidence)
            max_error = max(max_error, error)

    return max_error


def get_calibration_curve(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data.

    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_confidences)
    """
    confidences = np.array(confidences)
    accuracies = np.array([1 if a else 0 for a in accuracies])

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_confs = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        bin_center = (bin_lower + bin_upper) / 2

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            bin_centers.append(bin_center)
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())

    return np.array(bin_centers), np.array(bin_accs), np.array(bin_confs)


def plot_reliability_diagram(
    confidences: List[float],
    accuracies: List[bool],
    num_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[str] = None
):
    """
    Plot reliability diagram (calibration curve).

    A perfectly calibrated model lies on the diagonal.

    Args:
        confidences: Confidence scores
        accuracies: Binary accuracy labels
        num_bins: Number of bins
        title: Plot title
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt

    bin_centers, bin_accs, bin_confs = get_calibration_curve(
        confidences, accuracies, num_bins
    )
    ece = compute_ece(confidences, accuracies, num_bins)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # Calibration curve
    ax.bar(
        bin_centers, bin_accs,
        width=1.0/num_bins,
        alpha=0.7,
        edgecolor='black',
        label=f'Model (ECE = {ece:.3f})'
    )

    # Gap visualization
    for bc, ba, bconf in zip(bin_centers, bin_accs, bin_confs):
        if ba < bconf:
            ax.fill_between([bc - 0.5/num_bins, bc + 0.5/num_bins],
                          ba, bconf, alpha=0.3, color='red')
        else:
            ax.fill_between([bc - 0.5/num_bins, bc + 0.5/num_bins],
                          bconf, ba, alpha=0.3, color='green')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def uncertainty_to_confidence(
    uncertainties: List[float],
    method: str = 'normalize'
) -> List[float]:
    """
    Convert uncertainty scores to confidence scores in [0, 1].

    Args:
        uncertainties: Uncertainty scores (higher = more uncertain)
        method: 'normalize' or 'sigmoid'

    Returns:
        Confidence scores in [0, 1] (higher = more confident)
    """
    uncertainties = np.array(uncertainties)

    if method == 'normalize':
        # Linear normalization
        min_u, max_u = uncertainties.min(), uncertainties.max()
        if max_u - min_u > 0:
            normalized = (uncertainties - min_u) / (max_u - min_u)
        else:
            normalized = np.zeros_like(uncertainties)
        return (1 - normalized).tolist()

    elif method == 'sigmoid':
        # Sigmoid transformation
        center = uncertainties.mean()
        scale = uncertainties.std() + 1e-10
        normalized = (uncertainties - center) / scale
        confidences = 1 / (1 + np.exp(normalized))
        return confidences.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")
