"""
Token Entropy and Varentropy for uncertainty quantification.

Varentropy (variance of entropy) detects oscillating confidence - a hallucination signal.
"""

from typing import Optional, Tuple
import torch


def compute_token_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token predictive entropy: H = -sum(p * log p)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    raw_entropy = -torch.sum(probs * log_probs, dim=-1)

    # Align: entropy[i] = uncertainty about token[i]
    entropy = torch.zeros_like(raw_entropy)
    entropy[:, 1:] = raw_entropy[:, :-1]

    if attention_mask is not None:
        entropy = entropy * attention_mask.float()
    return entropy


def compute_varentropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute varentropy (variance of entropy) per token.

    High varentropy = oscillating confidence = hallucination signal.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
    surprisal = -log_probs
    squared_deviation = (surprisal - entropy) ** 2
    raw_varentropy = torch.sum(probs * squared_deviation, dim=-1)

    varentropy = torch.zeros_like(raw_varentropy)
    varentropy[:, 1:] = raw_varentropy[:, :-1]

    if attention_mask is not None:
        varentropy = varentropy * attention_mask.float()
    return varentropy
