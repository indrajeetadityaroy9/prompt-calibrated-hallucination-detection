"""
Numerical utilities for safe computation of softmax, JSD, and similarity.
"""

from typing import Optional, Tuple
import torch
from torch import Tensor
import math


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


def max_cosine_similarity(query: Tensor, keys: Tensor, eps: float = 1e-8) -> float:
    """
    Compute maximum cosine similarity between query and multiple keys.

    Args:
        query: Query vector [dim]
        keys: Key vectors [num_keys, dim]
        eps: Small constant for numerical stability

    Returns:
        Maximum cosine similarity in [-1, 1]
    """
    query_norm = query / (torch.norm(query) + eps)
    keys_norm = keys / (torch.norm(keys, dim=-1, keepdim=True) + eps)

    similarities = torch.matmul(keys_norm, query_norm)
    return similarities.max().item()
