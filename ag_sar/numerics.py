"""
Numerical utilities for safe computation of softmax, JSD, and similarity.
"""

import torch
from torch import Tensor
import math

# Canonical numerical stability constant — single source of truth.
# Import this in signal/aggregation modules instead of defining locally.
EPS = 1e-10



# Dtype-aware lower bounds for softmax clamping. Derived from
# math.log(torch.finfo(dtype).tiny), rounded conservatively.
# Below these values, exp() produces 0.0 — clamping is numerically lossless.
_SOFTMAX_MIN = {torch.float32: -88.0, torch.float16: -15.0, torch.bfloat16: -88.0}


def safe_softmax(logits: Tensor, dim: int = -1, eps: float = 1e-10) -> Tensor:
    """Numerically stable softmax: max-subtraction + dtype-aware clamping."""
    # Subtract max for numerical stability
    logits_max = logits.max(dim=dim, keepdim=True).values
    logits_shifted = logits - logits_max

    # Dtype-aware clamp to prevent exp overflow
    min_val = _SOFTMAX_MIN.get(logits_shifted.dtype, -88.0)
    logits_clamped = torch.clamp(logits_shifted, min=min_val, max=0)

    exp_logits = torch.exp(logits_clamped)
    probs = exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + eps)

    return probs


def safe_log_softmax(logits: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable log-softmax via log-sum-exp trick."""
    logits_max = logits.max(dim=dim, keepdim=True).values
    logits_shifted = logits - logits_max

    # log_softmax = logits - log(sum(exp(logits)))
    # = logits - max - log(sum(exp(logits - max)))
    log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=dim, keepdim=True) + 1e-10)

    return logits_shifted - log_sum_exp


def safe_jsd(p: Tensor, q: Tensor, eps: float = 1e-10) -> float:
    """JSD(P||Q) in bits, bounded [0, 1]. M = 0.5*(P+Q)."""
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


# ── Adaptive Nucleus ─────────────────────────────────────────────

# Base nucleus mass, embedded here (private). Callers use entropy_adaptive_nucleus().
_BASE_NUCLEUS_MASS = 0.95


def entropy_adaptive_nucleus(probs: Tensor) -> float:
    """Adaptive nucleus mass: mass = base + (1 - base) * (1 - H/H_max).

    Boundary conditions (verified):
    - Uniform (H = ln(V)):  mass = 0.95 + 0.05 * 0 = 0.95
    - Delta   (H = 0):      mass = 0.95 + 0.05 * 1 = 1.00
    - Intermediate: linear interpolation between 0.95 and 1.0

    For peaked distributions (low entropy), the candidate set grows slightly
    (mass closer to 1.0), ensuring rare but important tokens are included.
    For flat distributions, standard 0.95 applies.
    """
    V = probs.shape[-1]
    if V <= 1:
        return 1.0
    ln_V = math.log(V)
    p_clamped = probs.clamp(min=EPS)
    H = -(p_clamped * p_clamped.log()).sum().item()
    normalized_entropy = min(1.0, H / ln_V)
    return _BASE_NUCLEUS_MASS + (1.0 - _BASE_NUCLEUS_MASS) * (1.0 - normalized_entropy)


def otsu_threshold(values) -> float:
    """Optimal bimodal threshold: argmax sigma_b^2(t) = w0*w1*(mu0-mu1)^2. Otsu (1979)."""
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


