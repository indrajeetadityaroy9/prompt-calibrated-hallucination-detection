"""
Authority Flow Measures (v3.1/v3.2).

Implements signal provenance tracking for hallucination detection:
- Register Filter: Kurtosis-based token classification
- Authority Flow: Recursive prompt recharge + generation flow
- MLP Divergence: Measures MLP override of attention signal

Key Insight: "Confident Lies are smooth on the attention graph"
- Authority Flow tracks WHERE signal comes from (context vs. parametric)
- MLP Divergence tracks WHAT the MLP does to attention output
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from ..ops import (
    EMAState,
    fisher_kurtosis,
    welford_update,
    compute_authority_flow as _compute_authority_flow_raw,
    compute_authority_flow_vectorized,
    compute_spectral_roughness,
    compute_mlp_divergence as _compute_mlp_divergence_raw,
)


def compute_register_mask(
    value_vectors: torch.Tensor,
    ema_state: Optional[EMAState] = None,
    kurtosis_threshold: float = 2.0,
    sink_token_count: int = 4,
    ema_decay: float = 0.995,
    update_ema: bool = True,
) -> Tuple[torch.Tensor, Optional[EMAState]]:
    """
    Compute Register Mask M(t) for filtering sinks and registers.

    Formula: M(t) = (t > sink_count) * Sigmoid(-Z(t) + tau)
    where Z(t) = (Kurt(v_t) - mu_EMA) / sigma_EMA

    Low-kurtosis tokens (registers/sinks) -> low mask value
    High-kurtosis tokens (semantic) -> high mask value

    Args:
        value_vectors: (B, S, D) value vectors per token
        ema_state: Previous EMA state for online adaptation
        kurtosis_threshold: tau threshold for sigmoid gate
        sink_token_count: First N tokens to mask as sinks
        ema_decay: Decay factor for EMA update
        update_ema: Whether to update EMA state

    Returns:
        mask: (B, S) register mask in [0, 1]
        updated_ema_state: Updated EMA state
    """
    from ..ops import compute_register_mask as _compute_register_mask
    return _compute_register_mask(
        value_vectors, ema_state, kurtosis_threshold,
        sink_token_count, ema_decay, update_ema
    )


def compute_authority_score(
    attention_weights: torch.Tensor,
    prompt_length: int,
    register_mask: Optional[torch.Tensor] = None,
    roughness: Optional[torch.Tensor] = None,
    lambda_roughness: float = 10.0,
    previous_authority: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_vectorized: bool = False,
    subject_boost: float = 0.0,
    subject_token_count: int = 5,
) -> torch.Tensor:
    """
    Compute Authority Score with optional spectral roughness penalty.

    Unified interface for v3.1 Authority Flow:
        A(t) = [sum_Prompt A_{t,j}] + [sum_Gen A_{t,j} * A(j)] * M(t)
        A_final(t) = A(t) / (1 + lambda * roughness(t))

    v4.0 Subject Anchor (for WikiBio-style context-free generation):
        When prompt_length is 0, first N tokens serve as subject anchor.
        Attention to subject tokens is boosted by subject_boost factor.

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        register_mask: (B, S) register mask M(t) in [0, 1]
        roughness: (B, S) spectral roughness or MLP divergence
        lambda_roughness: Penalty weight for roughness
        previous_authority: (B, S) authority from previous step (streaming)
        attention_mask: (B, S) padding mask
        use_vectorized: Use vectorized approximation (faster, less accurate)
        subject_boost: Multiplier for attention to subject tokens (0.0 = disabled)
        subject_token_count: Number of tokens to treat as subject

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    if use_vectorized:
        authority = compute_authority_flow_vectorized(
            attention_weights, prompt_length, register_mask, attention_mask,
            subject_boost=subject_boost, subject_token_count=subject_token_count
        )
    else:
        authority = _compute_authority_flow_raw(
            attention_weights, prompt_length, register_mask,
            previous_authority, attention_mask
        )

    # Apply roughness penalty if provided
    if roughness is not None and lambda_roughness > 0:
        authority = authority / (1.0 + lambda_roughness * roughness)
        authority = authority.clamp(0.0, 1.0)

    return authority


def compute_mlp_divergence(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MLP Divergence for hallucination detection.

    v3.2 Hypothesis: When hallucinating, MLP overrides attention signal
    with parametric memory, causing divergence between h_attn and h_block.

    Formula: delta(t) = 1 - CosineSim(h_attn, h_block)

    Args:
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        attention_mask: (B, S) optional padding mask

    Returns:
        divergence: (B, S) MLP divergence per token [0, 2]
            - 0 = perfect alignment (attention and MLP agree)
            - 1 = orthogonal
            - 2 = opposite directions
    """
    return _compute_mlp_divergence_raw(h_attn, h_block, attention_mask)


def compute_gated_authority(
    attention_weights: torch.Tensor,
    prompt_length: int,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    logits: torch.Tensor,
    register_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    stability_sensitivity: float = 10.0,
    parametric_weight: float = 0.5,
) -> torch.Tensor:
    """
    Context-Dependent Gated Authority (Unified RAG + Free Gen).

    Master Equation:
        A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Confidence(t) × parametric_weight

    Where:
    - Gate(t) = exp(-sensitivity × divergence(t)) = stability gate
    - Flow(t) = Authority Flow (prompt provenance tracking)
    - Confidence(t) = max(softmax(logits)) (parametric certainty)

    Physical Interpretation:
    - In RAG: MLP validates context → Gate ≈ 1.0 → Trust Flow
    - In Free Gen: MLP injects parametric → Gate ≈ 0.0 → Trust Confidence

    This unifies hallucination detection across all generation modes:
    - RAG with context: High authority = grounded in context
    - Free generation: High authority = confident in parametric memory
    - Mixed: Smooth interpolation based on MLP stability

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        logits: (B, S, V) model output logits for confidence
        register_mask: (B, S) register mask M(t) in [0, 1]
        attention_mask: (B, S) padding mask
        stability_sensitivity: Controls gate sharpness (default 10.0)
        parametric_weight: Weight for confidence when ignoring context (default 0.5)

    Returns:
        authority: (B, S) gated authority scores in [0, 1]

    Example:
        >>> # RAG scenario: model attends to context, MLP validates
        >>> auth = compute_gated_authority(attn, 50, h_attn, h_block, logits)
        >>> # auth[:, 50:] ≈ authority_flow (context-grounded)

        >>> # Free Gen: model ignores context, MLP provides knowledge
        >>> auth = compute_gated_authority(attn, 0, h_attn, h_block, logits)
        >>> # auth ≈ model_confidence (parametric-grounded)
    """
    from ..ops import compute_stability_gate

    # 1. Compute base Authority Flow
    flow = compute_authority_flow_vectorized(
        attention_weights, prompt_length, register_mask, attention_mask
    )

    # 2. Compute Stability Gate (Conductivity)
    # Gate = exp(-sensitivity × divergence)
    # High gate = MLP agrees with attention (stable)
    # Low gate = MLP overrides attention (parametric injection)
    gate = compute_stability_gate(h_attn, h_block, stability_sensitivity)

    # 3. Compute Parametric Confidence
    # Confidence = max probability (model's certainty in top prediction)
    probs = F.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values  # (B, S)

    # 4. Master Equation: Gated interpolation
    # A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Confidence(t) × weight
    #
    # When Gate ≈ 1 (RAG, context-driven):
    #   A(t) ≈ Flow(t) → Trust authority flow from context
    #
    # When Gate ≈ 0 (Free Gen, parametric-driven):
    #   A(t) ≈ Confidence(t) × weight → Trust model's parametric certainty
    authority = gate * flow + (1.0 - gate) * confidence * parametric_weight

    # Apply attention mask
    if attention_mask is not None:
        authority = authority * attention_mask.float()

    # Clamp to [0, 1]
    authority = authority.clamp(0.0, 1.0)

    return authority


def compute_semantic_authority(
    attention_weights: torch.Tensor,
    prompt_length: int,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    register_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    stability_sensitivity: float = 1.0,
    parametric_weight: float = 0.5,
    dispersion_k: int = 5,
    dispersion_sensitivity: float = 5.0,
) -> torch.Tensor:
    """
    Semantic Dispersion Authority (Consistency over Confidence).

    Replaces raw confidence with semantic consistency of top-k predictions.

    Master Equation:
        A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × parametric_weight

    Where:
    - Gate(t) = exp(-sensitivity × divergence(t)) = stability gate
    - Flow(t) = Authority Flow (prompt provenance tracking)
    - Trust(t) = 1 - Dispersion(t) × dispersion_sensitivity (semantic consistency)

    Key Insight:
    - Raw Confidence: "I am 99% sure it's 'Paris'" (could be a confident lie)
    - High Dispersion: "Top-5 are 'Paris', 'London', 'Rome'" (semantically confused = hallucination)
    - Low Dispersion: "Top-5 are 'US', 'USA', 'America'" (synonyms = grounded)

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (unembedding)
        register_mask: (B, S) register mask M(t) in [0, 1]
        attention_mask: (B, S) padding mask
        stability_sensitivity: Controls gate sharpness (default 1.0)
        parametric_weight: Weight for trust when ignoring context (default 0.5)
        dispersion_k: Number of top tokens for dispersion (default 5)
        dispersion_sensitivity: Scale factor for dispersion penalty (default 5.0)

    Returns:
        authority: (B, S) semantic authority scores in [0, 1]
    """
    from ..ops import compute_stability_gate
    from .semantics import compute_semantic_trust

    # 1. Compute base Authority Flow
    flow = compute_authority_flow_vectorized(
        attention_weights, prompt_length, register_mask, attention_mask
    )

    # 2. Compute Stability Gate (Conductivity)
    gate = compute_stability_gate(h_attn, h_block, stability_sensitivity)

    # 3. Compute Semantic Trust (replaces raw confidence)
    # Trust = 1 - (Dispersion × sensitivity)
    # High dispersion (confused between unrelated tokens) = Low trust
    # Low dispersion (alternatives are synonyms) = High trust
    trust = compute_semantic_trust(
        logits, embed_matrix, k=dispersion_k, sensitivity=dispersion_sensitivity
    )

    # 4. Master Equation: Gated interpolation with semantic trust
    # A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × weight
    #
    # When Gate ≈ 1 (RAG, context-driven):
    #   A(t) ≈ Flow(t) → Trust authority flow from context
    #
    # When Gate ≈ 0 (Free Gen, parametric-driven):
    #   A(t) ≈ Trust(t) × weight → Trust semantic consistency
    authority = gate * flow + (1.0 - gate) * trust * parametric_weight

    # Apply attention mask
    if attention_mask is not None:
        authority = authority * attention_mask.float()

    # Clamp to [0, 1]
    authority = authority.clamp(0.0, 1.0)

    return authority
