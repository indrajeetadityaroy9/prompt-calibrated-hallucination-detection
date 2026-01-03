"""
Spectral Surprisal Measures.

Implements Manifold-Consistent Spectral Surprisal (MC-SS):
    MC-SS = S_bounded + lambda * (1 - v_norm)

Key insight: Uses ADDITIVE formulation (not multiplicative) to catch
"Confident Lies" - tokens with low surprisal but low centrality.

Also provides Graph-Shifted Surprisal (GSS) for forced response evaluation.
"""

from typing import Optional
import torch


def compute_token_surprisal(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute per-token surprisal (negative log probability).

    Surprisal(t_i) = -log P(t_i | t_1, ..., t_{i-1})

    This is the correct metric for FORCED responses where we want to measure
    how surprised the model was by each specific token.

    Args:
        logits: (batch, seq, vocab) model output logits
        input_ids: (batch, seq) actual token IDs
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature

    Returns:
        surprisal: (batch, seq) surprisal per token
    """
    # Shift logits and labels: logits[i] predicts input_ids[i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Per-token NLL
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # Reshape to (batch, seq-1)
    nll = nll.view(shift_labels.shape)

    # Pad to match original sequence length
    batch_size = logits.size(0)
    padding = torch.zeros((batch_size, 1), device=logits.device, dtype=logits.dtype)
    surprisal = torch.cat([padding, nll], dim=1)

    if attention_mask is not None:
        surprisal = surprisal * attention_mask.float()

    return surprisal


def compute_bounded_surprisal(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    beta: float = 5.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute soft-clamped bounded surprisal for MC-SS.

    S_bounded(x_t) = tanh(-log P(x_t | x_{<t}) / beta)

    This bounds surprisal to [0, 1], preventing extreme values from
    dominating the MC-SS metric.

    Args:
        logits: (batch, seq, vocab) model output logits
        input_ids: (batch, seq) actual token IDs
        beta: Softness parameter (higher = softer clamp)
        attention_mask: (batch, seq) valid token mask

    Returns:
        bounded_surprisal: (batch, seq) in range [0, 1]
    """
    raw_surprisal = compute_token_surprisal(logits, input_ids, attention_mask=None)
    bounded = torch.tanh(raw_surprisal / beta)

    if attention_mask is not None:
        bounded = bounded * attention_mask.float()

    return bounded


def compute_manifold_consistent_spectral_surprisal(
    bounded_surprisal: torch.Tensor,
    centrality: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    penalty_weight: float = 1.0,
) -> torch.Tensor:
    """
    Compute Manifold-Consistent Spectral Surprisal (MC-SS).

    MC-SS = Mean( S_bounded + lambda * (1 - v_norm) )

    CRITICAL: Uses ADDITIVE formulation to catch "Confident Lies":
        - Confident Fact: 0.1 + 0 = 0.1 (Low)
        - Confident Lie: 0.1 + 0.98 = 1.08 (High)
        - Confused Lie: 0.9 + 0.98 = 1.88 (Very High)

    CRITICAL: Uses MAX-normalization (not L1) for centrality to preserve
    discriminative power between grounded/ungrounded tokens.

    Args:
        bounded_surprisal: (batch, seq) soft-clamped surprisal in [0, 1]
        centrality: (batch, seq) Hebbian-filtered centrality
        attention_mask: (batch, seq) valid token mask
        penalty_weight: lambda weight for penalty term

    Returns:
        mcss: (batch,) MC-SS score per sequence
    """
    # Mask before finding max
    if attention_mask is not None:
        centrality = centrality * attention_mask.float()

    # MAX-normalize centrality (NOT L1!)
    v_max = centrality.max(dim=-1, keepdim=True).values + 1e-10
    v_norm = centrality / v_max

    # Structure penalty (inverted centrality)
    penalty = 1.0 - v_norm

    # ADDITIVE fusion
    score_token = bounded_surprisal + (penalty_weight * penalty)

    # Average over valid tokens
    if attention_mask is not None:
        valid_tokens = attention_mask.sum(dim=-1).clamp(min=1)
        mcss = (score_token * attention_mask.float()).sum(dim=-1) / valid_tokens
    else:
        mcss = score_token.mean(dim=-1)

    return mcss


def compute_graph_shifted_surprisal(
    surprisal: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Graph-Shifted Surprisal (GSS).

    GSS(T) = sum_i S(t_i) * R~(t_i)

    This is the CORRECT metric for hallucination detection on forced responses.
    It combines likelihood signal (surprisal) with topology signal (relevance).

    "I don't care that you are surprised by 'the'. I only care if you are
    surprised by 'Paris'." - The Graph provides the filter.

    Args:
        surprisal: (batch, seq) per-token surprisal
        relevance: (batch, seq) sink-aware relevance scores
        attention_mask: (batch, seq) valid token mask

    Returns:
        gss: (batch,) Graph-Shifted Surprisal per sequence
    """
    # Normalize relevance
    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    total = relevance.sum(dim=-1, keepdim=True).clamp(min=1e-10)
    normalized_relevance = relevance / total

    if attention_mask is not None:
        surprisal = surprisal * attention_mask.float()

    gss = (surprisal * normalized_relevance).sum(dim=-1)
    return gss
