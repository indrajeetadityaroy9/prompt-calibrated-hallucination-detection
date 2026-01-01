"""
Uncertainty quantification via Graph-Shifted Entropy (GSE).

Module: uncertainty.py

Implements the core uncertainty metric that leverages attention graph
structure to focus on semantically relevant tokens.

Paper Reference:
    Section 3.3: Graph-Shifted Entropy Computation
    Core formula: GSE(T) = Σ_i H(t_i) × R̃(t_i)

Where:
- H(t_i) = per-token predictive entropy
- R̃(t_i) = normalized sink-aware relevance

Key Functions:
    - compute_token_entropy(): Per-token predictive entropy H(t_i)
    - compute_graph_shifted_entropy(): Main GSE computation
    - detect_hallucination(): Threshold-based detection
    - compute_per_token_uncertainty(): Token-level contributions

GSE down-weights uncertainty contributions from:
- Leaf nodes (irrelevant tokens with low relevance)
- Attention sinks (high attention but low semantic value)
"""

from typing import Optional, Tuple
import torch


def compute_token_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute per-token predictive entropy.

    H(t_i) = -Σ_v p(v|context) × log p(v|context)

    Higher entropy = more uncertainty about the predicted token.

    Args:
        logits: (batch, seq, vocab) model output logits
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature for calibration

    Returns:
        entropy: (batch, seq) entropy per token position
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Compute log probabilities (numerically stable via log_softmax)
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)

    # Entropy: H = -Σ p × log(p)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Apply mask if provided
    if attention_mask is not None:
        entropy = entropy * attention_mask.float()

    return entropy


def normalize_relevance(
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Normalize relevance scores to sum to 1 (within valid tokens).

    R̃(t_i) = R(t_i) / Σ_j R(t_j)

    Args:
        relevance: (batch, seq) raw relevance scores
        attention_mask: (batch, seq) valid token mask
        eps: Numerical stability constant

    Returns:
        normalized: (batch, seq) normalized relevance summing to 1
    """
    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    total = relevance.sum(dim=-1, keepdim=True).clamp(min=eps)
    normalized = relevance / total

    return normalized


def compute_graph_shifted_entropy(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Graph-Shifted Entropy (GSE).

    GSE(T) = Σ_i H(t_i) × R̃(t_i)

    This shifts focus from all tokens to only semantically relevant ones,
    naturally down-weighting uncertainty from:
    - Leaf nodes (low relevance tokens)
    - Attention sinks (high attention, low value norm tokens)

    Args:
        token_entropy: (batch, seq) per-token entropy
        relevance: (batch, seq) sink-aware relevance scores
        attention_mask: (batch, seq) valid token mask

    Returns:
        gse: (batch,) Graph-Shifted Entropy per sequence
    """
    # Normalize relevance to get weights
    normalized_relevance = normalize_relevance(relevance, attention_mask)

    # Apply mask to entropy if provided
    if attention_mask is not None:
        token_entropy = token_entropy * attention_mask.float()

    # Weighted sum: GSE = Σ(H × R̃)
    gse = (token_entropy * normalized_relevance).sum(dim=-1)

    return gse


def detect_hallucination(
    gse: torch.Tensor,
    threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect hallucination based on GSE threshold.

    Higher GSE indicates more uncertainty on relevant tokens,
    suggesting the model is less confident and more likely hallucinating.

    Args:
        gse: (batch,) Graph-Shifted Entropy scores
        threshold: Hallucination threshold

    Returns:
        is_hallucination: (batch,) boolean tensor, True if likely hallucinating
        confidence: (batch,) confidence scores (how far from threshold)
    """
    is_hallucination = gse > threshold

    # Confidence: sigmoid of distance from threshold
    # Values close to threshold -> ~0.5 confidence
    # Values far above threshold -> ~1.0 confidence
    # Values far below threshold -> ~0.0 confidence
    confidence = torch.sigmoid(gse - threshold)

    return is_hallucination, confidence


def compute_per_token_uncertainty(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-token uncertainty contribution to GSE.

    Useful for identifying which tokens contribute most to overall uncertainty.

    Args:
        token_entropy: (batch, seq) per-token entropy
        relevance: (batch, seq) sink-aware relevance scores
        attention_mask: (batch, seq) valid token mask

    Returns:
        uncertainty_contribution: (batch, seq) per-token GSE contribution
    """
    normalized_relevance = normalize_relevance(relevance, attention_mask)
    contribution = token_entropy * normalized_relevance

    if attention_mask is not None:
        contribution = contribution * attention_mask.float()

    return contribution
