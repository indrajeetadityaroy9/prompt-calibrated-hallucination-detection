"""
Token Entropy Computation.

Basic per-token predictive entropy, useful as a baseline metric.
"""

from typing import Optional
import torch


def compute_token_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute per-token predictive entropy.

    H(t_i) = -sum_v p(v|context) * log p(v|context)

    ALIGNMENT: In autoregressive models, logits[i] predicts token[i+1].
    We shift the entropy so entropy[i] represents uncertainty about token[i].

    Args:
        logits: (batch, seq, vocab) model output logits
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature for calibration

    Returns:
        entropy: (batch, seq) entropy per token position
    """
    scaled_logits = logits / temperature
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)

    # Raw entropy (about next token prediction)
    raw_entropy = -torch.sum(probs * log_probs, dim=-1)

    # Align: entropy[i] = uncertainty about generating token[i]
    entropy = torch.zeros_like(raw_entropy)
    entropy[:, 1:] = raw_entropy[:, :-1]

    if attention_mask is not None:
        entropy = entropy * attention_mask.float()

    return entropy
