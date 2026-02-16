"""
Numerical utilities for safe computation of softmax, JSD, and similarity.
"""

import torch
from torch import Tensor
import math

# Canonical numerical stability constant — single source of truth.
# Import this in signal/aggregation modules instead of defining locally.
EPS = 1e-10


def safe_softmax(logits: Tensor, dim: int = -1, eps: float = 1e-10) -> Tensor:
    """
    Numerically stable softmax that handles extreme values.

    Subtracts max before exp to prevent overflow.

    Args:
        logits: Input logits tensor
        dim: Dimension to apply softmax over
        eps: Small constant added to prevent log(0)

    Returns:
        Probability distribution (sums to 1 along dim)
    """
    # Subtract max for numerical stability
    logits_max = logits.max(dim=dim, keepdim=True).values
    logits_shifted = logits - logits_max

    # Clamp to prevent exp overflow
    logits_clamped = torch.clamp(logits_shifted, min=-100, max=0)

    exp_logits = torch.exp(logits_clamped)
    probs = exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + eps)

    return probs


def safe_log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """
    Numerically stable log-softmax.

    Uses log-sum-exp trick for stability.

    Args:
        logits: Input logits tensor
        dim: Dimension to apply log-softmax over

    Returns:
        Log probabilities
    """
    logits_max = logits.max(dim=dim, keepdim=True).values
    logits_shifted = logits - logits_max

    # log_softmax = logits - log(sum(exp(logits)))
    # = logits - max - log(sum(exp(logits - max)))
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=dim, keepdim=True) + 1e-10)

    return logits_shifted - log_sum_exp


def safe_jsd(p: Tensor, q: Tensor, eps: float = 1e-10) -> float:
    """
    Compute Jensen-Shannon Divergence between two probability distributions.

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    where M = 0.5 * (P + Q)

    Returns JSD in BITS (divided by ln(2)), bounded [0, 1].

    Args:
        p: First probability distribution (1D tensor)
        q: Second probability distribution (1D tensor)
        eps: Small constant to prevent log(0)

    Returns:
        JSD in bits, bounded [0, 1]
    """
    # Ensure valid probability distributions
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()

    # Mixture distribution
    m = 0.5 * (p + q)

    # KL divergences
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum()
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum()

    # JSD
    jsd_nats = 0.5 * kl_pm + 0.5 * kl_qm

    # Convert to bits and clamp to [0, 1]
    jsd_bits = jsd_nats.item() / math.log(2)
    jsd_bits = max(0.0, min(1.0, jsd_bits))

    return jsd_bits


def otsu_threshold(values) -> float:
    """
    Optimal bimodal threshold maximizing between-class variance.

    sigma_b^2(t) = w0 * w1 * (mu0 - mu1)^2

    Zero free parameters. Optimal for bimodal distributions.

    Reference: Otsu (1979) "A Threshold Selection Method from
    Gray-Level Histograms"

    Args:
        values: 1D array-like of values to threshold

    Returns:
        Optimal threshold value
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float(values[0]) if len(values) == 1 else 0.0

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    best_threshold = sorted_vals[0]
    best_variance = -1.0

    for i in range(1, n):
        w0 = i / n
        w1 = 1.0 - w0
        mu0 = sorted_vals[:i].mean()
        mu1 = sorted_vals[i:].mean()
        variance = w0 * w1 * (mu0 - mu1) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = 0.5 * (sorted_vals[i - 1] + sorted_vals[i])

    return float(best_threshold)


def mad_sigma(values) -> float:
    """
    Robust standard deviation estimate via Median Absolute Deviation.

    sigma_MAD = 1.4826 * median(|x_i - median(x)|)

    The constant 1.4826 makes this a consistent estimator of sigma
    for Gaussian distributions, while being robust to outliers.

    Reference: Rousseeuw & Croux (1993) "Alternatives to the Median
    Absolute Deviation"

    Args:
        values: 1D array-like of values

    Returns:
        Robust sigma estimate (0.0 if fewer than 2 values)
    """
    import numpy as np
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return 0.0
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return 1.4826 * float(mad)


