"""
Authority Flow Measures (v8.0 Gold Master).

Implements signal provenance tracking for hallucination detection:
- Authority Flow: Recursive prompt recharge + generation flow
- Gated Authority: Unified RAG + Free Gen with stability gating
- Semantic Authority: Semantic dispersion for confident lie detection

Key Insight: "Confident Lies are smooth on the attention graph"
- Authority Flow tracks WHERE signal comes from (context vs. parametric)
- Stability Gate measures MLP agreement with attention output
"""

from typing import Optional
import torch
import torch.nn.functional as F

from ..ops import (
    compute_authority_flow as _compute_authority_flow_raw,
    compute_authority_flow_vectorized,
    compute_mlp_divergence as _compute_mlp_divergence_raw,
)


def compute_authority_score(
    attention_weights: torch.Tensor,
    prompt_length: int,
    previous_authority: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_vectorized: bool = False,
    subject_boost: float = 0.0,
    subject_token_count: int = 5,
) -> torch.Tensor:
    """
    Compute Authority Score for hallucination detection.

    Authority Flow:
        A(t) = [sum_Prompt A_{t,j}] + [sum_Gen A_{t,j} * A(j)]

    Subject Anchor (for context-free generation):
        When prompt_length is small, first N tokens serve as subject anchor.
        Attention to subject tokens is boosted by subject_boost factor.

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
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
            attention_weights, prompt_length, None, attention_mask,
            subject_boost=subject_boost, subject_token_count=subject_token_count
        )
    else:
        authority = _compute_authority_flow_raw(
            attention_weights, prompt_length, None,
            previous_authority, attention_mask
        )

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
        attention_weights, prompt_length, None, attention_mask
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
    attention_mask: Optional[torch.Tensor] = None,
    stability_sensitivity: float = 1.0,
    parametric_weight: float = 0.5,
    dispersion_k: int = 5,
    dispersion_sensitivity: float = 5.0,
    dispersion_method: str = "top1_projection",
    nucleus_top_p: float = 0.95,
    # Intrinsic Detection (v12.0)
    intrinsic_trust: Optional[torch.Tensor] = None,
    # Gate Sharpening (v12.1 - Fixes Gate Leak)
    enable_gate_sharpening: bool = False,
    gate_sharpen_low: float = 0.2,
    gate_sharpen_high: float = 0.8,
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

    v12.0 Dynamic Blending (when intrinsic_trust is provided):
        trust_combined = Gate × JEPA_trust + (1-Gate) × intrinsic_trust
        A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × trust_combined × parametric_weight

    This enables:
    - High Gate (good RAG) → mostly JEPA trust (safe for novel data)
    - Low Gate (hallucination) → mostly intrinsic trust (catches lies)

    Key Insight:
    - Raw Confidence: "I am 99% sure it's 'Paris'" (could be a confident lie)
    - High Dispersion: "Top-5 are 'Paris', 'London', 'Rome'" (semantically confused = hallucination)
    - Low Dispersion: "Top-5 are 'US', 'USA', 'America'" (synonyms = grounded)

    Dispersion Methods:
    - "top1_projection": Distance from top-1 (best for QA/factual)
    - "centroid_variance": JEPA-style spread around centroid (Top-K)
    - "nucleus_variance": SOTA adaptive Top-P clustering (dynamic k)

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (unembedding)
        attention_mask: (B, S) padding mask
        stability_sensitivity: Controls gate sharpness (default 1.0)
        parametric_weight: Weight for trust when ignoring context (default 0.5)
        dispersion_k: Number of top tokens for dispersion (default 5)
        dispersion_sensitivity: Scale factor for dispersion penalty (default 5.0)
        dispersion_method: "top1_projection", "centroid_variance", or "nucleus_variance"
        nucleus_top_p: Cumulative probability threshold for nucleus_variance (default 0.95)
        intrinsic_trust: Optional (B, S) intrinsic trust from Truth Vector (v12.0)
        enable_gate_sharpening: Force gate to extremes (0 or 1) to prevent signal pollution
        gate_sharpen_low: Below this threshold, force gate to 0.0 (default 0.2)
        gate_sharpen_high: Above this threshold, force gate to 1.0 (default 0.8)

    Returns:
        authority: (B, S) semantic authority scores in [0, 1]
    """
    from ..ops import compute_stability_gate
    from .semantics import compute_semantic_trust

    # 1. Compute base Authority Flow
    flow = compute_authority_flow_vectorized(
        attention_weights, prompt_length, None, attention_mask
    )

    # 2. Compute Stability Gate (Conductivity)
    gate = compute_stability_gate(h_attn, h_block, stability_sensitivity)

    # 2.5 Gate Sharpening (v12.1 - Fixes Gate Leak)
    # Problem: On no-context scenarios (TruthfulQA), gate ~0.5 allows JEPA noise
    # to pollute the Truth Vector signal. Solution: binarize the gate.
    # - Gate < low_thresh → 0.0 (pure intrinsic trust)
    # - Gate > high_thresh → 1.0 (pure JEPA trust)
    if enable_gate_sharpening:
        gate = torch.where(gate < gate_sharpen_low, torch.zeros_like(gate), gate)
        gate = torch.where(gate > gate_sharpen_high, torch.ones_like(gate), gate)

    # 3. Compute Semantic Trust (JEPA trust from dispersion)
    # Trust = 1 - (Dispersion × sensitivity)
    # High dispersion (confused between unrelated tokens) = Low trust
    # Low dispersion (alternatives are synonyms) = High trust
    jepa_trust = compute_semantic_trust(
        logits, embed_matrix, k=dispersion_k, sensitivity=dispersion_sensitivity,
        method=dispersion_method, top_p=nucleus_top_p
    )

    # 3.5 Dynamic Blending with Intrinsic Trust (v12.0)
    # If intrinsic trust is provided, blend with JEPA trust using Gate
    # trust_combined = Gate × JEPA_trust + (1-Gate) × intrinsic_trust
    #
    # Scenarios:
    # - Good RAG (Gate=0.95): trust ≈ JEPA_trust (safe for novel data)
    # - Hallucination (Gate=0.10): trust ≈ intrinsic_trust (catches lies)
    if intrinsic_trust is not None:
        blend_alpha = gate.clamp(0.0, 1.0)
        trust = blend_alpha * jepa_trust + (1.0 - blend_alpha) * intrinsic_trust
    else:
        trust = jepa_trust

    # 4. Master Equation: Gated interpolation with (possibly blended) trust
    # A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × weight
    #
    # When Gate ≈ 1 (RAG, context-driven):
    #   A(t) ≈ Flow(t) → Trust authority flow from context
    #
    # When Gate ≈ 0 (Free Gen, parametric-driven):
    #   A(t) ≈ Trust(t) × weight → Trust (blended) semantic consistency
    authority = gate * flow + (1.0 - gate) * trust * parametric_weight

    # Apply attention mask
    if attention_mask is not None:
        authority = authority * attention_mask.float()

    # Clamp to [0, 1]
    authority = authority.clamp(0.0, 1.0)

    return authority
