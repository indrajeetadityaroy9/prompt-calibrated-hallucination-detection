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

Performance:
    - Hot paths are torch.compiled for reduced Python overhead
    - Use get_compiled_functions() to access pre-compiled versions
"""

from typing import Optional, Tuple, Dict, Callable
import torch

# Cache for compiled functions
_COMPILED_CACHE: Dict[str, Callable] = {}


def _get_compile_options() -> dict:
    """Get torch.compile options optimized for inference."""
    return {
        'mode': 'reduce-overhead',  # Minimize Python overhead
        'fullgraph': True,  # Capture full graph for max optimization
        'dynamic': False,  # Static shapes for better optimization
    }


def compute_token_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    align_with_tokens: bool = True
) -> torch.Tensor:
    """
    Compute per-token predictive entropy.

    H(t_i) = -Σ_v p(v|context) × log p(v|context)

    Higher entropy = more uncertainty about the predicted token.

    CRITICAL ALIGNMENT FIX:
        In autoregressive models, logits[i] predicts token[i+1].
        When align_with_tokens=True (default), we shift the entropy so that
        entropy[i] represents the uncertainty about generating token[i],
        NOT about predicting token[i+1].

        This aligns with standard "Predictive Entropy" definitions (Kuhn et al.)
        and ensures H(t_i) × R(t_i) correctly weights each token's uncertainty.

    Args:
        logits: (batch, seq, vocab) model output logits
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature for calibration
        align_with_tokens: If True (default), shift entropy so entropy[i]
                          represents uncertainty about token[i]. If False,
                          use legacy behavior where entropy[i] is about token[i+1].

    Returns:
        entropy: (batch, seq) entropy per token position, aligned with tokens
    """
    # Apply temperature scaling
    scaled_logits = logits / temperature

    # Compute log probabilities (numerically stable via log_softmax)
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)

    # Entropy: H = -Σ p × log(p)
    # At this point, entropy[i] = uncertainty about predicting token[i+1]
    raw_entropy = -torch.sum(probs * log_probs, dim=-1)

    if align_with_tokens:
        # ALIGNMENT FIX: Shift entropy so entropy[i] = uncertainty about token[i]
        # logits[i] predicts token[i+1], so entropy[i] is about token[i+1]
        # We want entropy[i] to be about token[i]
        # Therefore: aligned_entropy[i] = raw_entropy[i-1] for i > 0
        #            aligned_entropy[0] = 0 (no prediction for first token)
        batch_size, seq_len = raw_entropy.shape
        entropy = torch.zeros_like(raw_entropy)
        # Shift: entropy[1:] = raw_entropy[:-1]
        # entropy[i] = raw_entropy[i-1] = uncertainty about token[i]
        entropy[:, 1:] = raw_entropy[:, :-1]
        # First token has no predictive entropy (it's the input, not generated)
        entropy[:, 0] = 0.0
    else:
        # Legacy behavior (misaligned - kept for backwards compatibility)
        entropy = raw_entropy

    # Apply mask if provided
    if attention_mask is not None:
        entropy = entropy * attention_mask.float()

    return entropy


def compute_token_surprisal(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute per-token surprisal (negative log probability of actual token).

    Surprisal(t_i) = -log P(t_i | t_1, ..., t_{i-1})

    This is the correct metric for FORCED responses (evaluation mode) where
    we want to measure how surprised the model was by each specific token,
    not just the overall uncertainty.

    For hallucination detection on forced responses:
    - Low surprisal = model expected this token = likely factual
    - High surprisal = model didn't expect this = likely hallucination

    Args:
        logits: (batch, seq, vocab) model output logits
        input_ids: (batch, seq) actual token IDs
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature for calibration

    Returns:
        surprisal: (batch, seq) surprisal per token position
                   surprisal[i] = -log P(token[i] | context before i)
    """
    # Use CrossEntropyLoss for efficient, numerically stable computation
    # Shift logits and labels to align: logits[i] predicts input_ids[i+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Compute per-token NLL
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nll = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    # Reshape back to (batch, seq-1)
    nll = nll.view(shift_labels.shape)

    # Pad with 0 at start to match original sequence length
    # surprisal[0] = 0 (first token has no prediction)
    batch_size = logits.size(0)
    padding = torch.zeros((batch_size, 1), device=logits.device, dtype=logits.dtype)
    surprisal = torch.cat([padding, nll], dim=1)

    # Apply mask if provided
    if attention_mask is not None:
        surprisal = surprisal * attention_mask.float()

    return surprisal


def compute_graph_shifted_surprisal(
    surprisal: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Graph-Shifted Surprisal (GSS).

    GSS(T) = Σ_i S(t_i) × R̃(t_i)

    Where:
    - S(t_i) = -log P(t_i | context) = token surprisal (NLL)
    - R̃(t_i) = normalized sink-aware relevance

    This is the CORRECT metric for hallucination detection on forced responses.
    It combines:
    1. Likelihood signal: How surprised was the model by this specific token?
    2. Topology signal: How central/relevant is this token in the attention graph?

    The key insight: Standard perplexity weights all tokens equally (1/N).
    GSS weights by structural centrality, focusing on semantically important tokens.

    "I don't care that you are surprised by 'the'. I only care if you are
    surprised by 'Paris'." - The Graph provides the filter.

    Args:
        surprisal: (batch, seq) per-token surprisal (NLL)
        relevance: (batch, seq) sink-aware relevance scores
        attention_mask: (batch, seq) valid token mask

    Returns:
        gss: (batch,) Graph-Shifted Surprisal per sequence
    """
    # Normalize relevance to get weights
    normalized_relevance = normalize_relevance(relevance, attention_mask)

    # Apply mask to surprisal if provided
    if attention_mask is not None:
        surprisal = surprisal * attention_mask.float()

    # Weighted sum: GSS = Σ(S × R̃)
    gss = (surprisal * normalized_relevance).sum(dim=-1)

    return gss


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


# =============================================================================
# Compiled versions for hot paths (torch.compile)
# =============================================================================

def _entropy_kernel(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Core entropy computation without masking (compile-friendly).

    Separated from compute_token_entropy for better torch.compile optimization.
    Returns RAW entropy (not aligned) - alignment is done after.
    """
    scaled_logits = logits / temperature
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=-1)


def _align_entropy_with_tokens(entropy: torch.Tensor) -> torch.Tensor:
    """
    Align entropy so entropy[i] represents uncertainty about token[i].

    In autoregressive models, logits[i] predicts token[i+1], so raw entropy[i]
    is about token[i+1]. This function shifts to align with token positions.

    Args:
        entropy: (batch, seq) raw entropy where entropy[i] is about token[i+1]

    Returns:
        aligned: (batch, seq) entropy where entropy[i] is about token[i]
    """
    aligned = torch.zeros_like(entropy)
    aligned[:, 1:] = entropy[:, :-1]
    return aligned


def _gse_kernel(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Core GSE computation without masking (compile-friendly).

    Separated from compute_graph_shifted_entropy for better torch.compile optimization.
    """
    # Normalize relevance
    total = relevance.sum(dim=-1, keepdim=True).clamp(min=eps)
    normalized_relevance = relevance / total

    # Weighted sum
    return (token_entropy * normalized_relevance).sum(dim=-1)


def get_compiled_entropy() -> Callable:
    """
    Get torch.compiled version of entropy kernel.

    Returns:
        Compiled function for entropy computation.
        First call triggers compilation, subsequent calls use cached version.
    """
    if 'entropy' not in _COMPILED_CACHE:
        try:
            _COMPILED_CACHE['entropy'] = torch.compile(
                _entropy_kernel,
                **_get_compile_options()
            )
        except Exception:
            # Fallback to non-compiled version if compilation fails
            _COMPILED_CACHE['entropy'] = _entropy_kernel
    return _COMPILED_CACHE['entropy']


def get_compiled_gse() -> Callable:
    """
    Get torch.compiled version of GSE kernel.

    Returns:
        Compiled function for GSE computation.
        First call triggers compilation, subsequent calls use cached version.
    """
    if 'gse' not in _COMPILED_CACHE:
        try:
            _COMPILED_CACHE['gse'] = torch.compile(
                _gse_kernel,
                **_get_compile_options()
            )
        except Exception:
            # Fallback to non-compiled version if compilation fails
            _COMPILED_CACHE['gse'] = _gse_kernel
    return _COMPILED_CACHE['gse']


def compute_token_entropy_compiled(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    align_with_tokens: bool = True
) -> torch.Tensor:
    """
    Compiled version of compute_token_entropy.

    Uses torch.compile for reduced Python overhead on the hot path.

    Args:
        logits: (batch, seq, vocab) model output logits
        attention_mask: (batch, seq) valid token mask
        temperature: Softmax temperature for calibration
        align_with_tokens: If True (default), align entropy with token positions

    Returns:
        entropy: (batch, seq) entropy per token position
    """
    entropy_fn = get_compiled_entropy()
    raw_entropy = entropy_fn(logits, temperature)

    if align_with_tokens:
        # Apply alignment fix: entropy[i] should be about token[i]
        entropy = _align_entropy_with_tokens(raw_entropy)
    else:
        entropy = raw_entropy

    if attention_mask is not None:
        entropy = entropy * attention_mask.float()

    return entropy


def compute_gse_compiled(
    token_entropy: torch.Tensor,
    relevance: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compiled version of compute_graph_shifted_entropy.

    Uses torch.compile for reduced Python overhead on the hot path.
    """
    # Apply masks before compiled kernel
    if attention_mask is not None:
        token_entropy = token_entropy * attention_mask.float()
        relevance = relevance * attention_mask.float()

    gse_fn = get_compiled_gse()
    return gse_fn(token_entropy, relevance)


def clear_compile_cache() -> None:
    """Clear the compiled function cache (useful for testing)."""
    _COMPILED_CACHE.clear()
