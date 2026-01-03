"""
Entropy-Based Uncertainty Measures.

Implements Graph-Shifted Entropy (GSE), the core AG-SAR metric:
    GSE(T) = sum_i H(t_i) * R~(t_i)

Where:
- H(t_i) = per-token predictive entropy
- R~(t_i) = normalized sink-aware relevance
"""

from typing import Optional, Tuple, Dict, Callable
import torch


# Compiled function cache
_COMPILED_CACHE: Dict[str, Callable] = {}


def _get_compile_options() -> dict:
    """Get torch.compile options optimized for inference."""
    return {
        'mode': 'reduce-overhead',
        'fullgraph': True,
        'dynamic': False,
    }


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


def normalize_relevance(
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Normalize relevance scores to sum to 1.

    R~(t_i) = R(t_i) / sum_j R(t_j)

    Args:
        relevance: (batch, seq) raw relevance scores
        attention_mask: (batch, seq) valid token mask
        eps: Numerical stability constant

    Returns:
        normalized: (batch, seq) normalized relevance
    """
    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    total = relevance.sum(dim=-1, keepdim=True).clamp(min=eps)
    return relevance / total


def compute_graph_shifted_entropy(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Graph-Shifted Entropy (GSE).

    GSE(T) = sum_i H(t_i) * R~(t_i)

    This shifts focus from all tokens to only semantically relevant ones,
    naturally down-weighting uncertainty from leaf nodes and attention sinks.

    Args:
        token_entropy: (batch, seq) per-token entropy
        relevance: (batch, seq) sink-aware relevance scores
        attention_mask: (batch, seq) valid token mask

    Returns:
        gse: (batch,) Graph-Shifted Entropy per sequence
    """
    normalized_relevance = normalize_relevance(relevance, attention_mask)

    if attention_mask is not None:
        token_entropy = token_entropy * attention_mask.float()

    gse = (token_entropy * normalized_relevance).sum(dim=-1)
    return gse


def detect_hallucination(
    gse: torch.Tensor,
    threshold: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Detect hallucination based on GSE threshold.

    Higher GSE = more uncertainty on relevant tokens = likely hallucinating.

    Args:
        gse: (batch,) Graph-Shifted Entropy scores
        threshold: Hallucination threshold

    Returns:
        is_hallucination: (batch,) boolean tensor
        confidence: (batch,) confidence scores
    """
    is_hallucination = gse > threshold
    confidence = torch.sigmoid(gse - threshold)
    return is_hallucination, confidence


def compute_per_token_uncertainty(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-token uncertainty contribution to GSE.

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


# =============================================================================
# Compiled versions for hot paths
# =============================================================================

def _entropy_kernel(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Core entropy computation (compile-friendly)."""
    scaled_logits = logits / temperature
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def _gse_kernel(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """Core GSE computation (compile-friendly)."""
    total = relevance.sum(dim=-1, keepdim=True).clamp(min=eps)
    normalized_relevance = relevance / total
    return (token_entropy * normalized_relevance).sum(dim=-1)


def get_compiled_entropy() -> Callable:
    """Get torch.compiled version of entropy kernel."""
    if 'entropy' not in _COMPILED_CACHE:
        try:
            _COMPILED_CACHE['entropy'] = torch.compile(
                _entropy_kernel, **_get_compile_options()
            )
        except Exception:
            _COMPILED_CACHE['entropy'] = _entropy_kernel
    return _COMPILED_CACHE['entropy']


def get_compiled_gse() -> Callable:
    """Get torch.compiled version of GSE kernel."""
    if 'gse' not in _COMPILED_CACHE:
        try:
            _COMPILED_CACHE['gse'] = torch.compile(
                _gse_kernel, **_get_compile_options()
            )
        except Exception:
            _COMPILED_CACHE['gse'] = _gse_kernel
    return _COMPILED_CACHE['gse']


def compute_token_entropy_compiled(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compiled version of compute_token_entropy."""
    entropy_fn = get_compiled_entropy()
    raw_entropy = entropy_fn(logits, temperature)

    # Alignment
    entropy = torch.zeros_like(raw_entropy)
    entropy[:, 1:] = raw_entropy[:, :-1]

    if attention_mask is not None:
        entropy = entropy * attention_mask.float()

    return entropy


def compute_gse_compiled(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compiled version of compute_graph_shifted_entropy."""
    if attention_mask is not None:
        token_entropy = token_entropy * attention_mask.float()
        relevance = relevance * attention_mask.float()

    gse_fn = get_compiled_gse()
    return gse_fn(token_entropy, relevance)


def clear_compile_cache() -> None:
    """Clear the compiled function cache."""
    _COMPILED_CACHE.clear()
