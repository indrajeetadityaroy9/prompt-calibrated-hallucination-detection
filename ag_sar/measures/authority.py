"""
Authority Flow - Signal provenance tracking for hallucination detection.

Core equation:
    Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)

Where:
    - Flow(t) = Recursive attention from prompt (information provenance)
    - Gate(t) = MLP-attention agreement (context reliance indicator)
    - Trust(t) = 1 - Dispersion × (1 + λ × Varentropy) (semantic consistency)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict

from ..ops import compute_authority_flow_vectorized, fused_stability_gate


def compute_semantic_authority(
    attention_weights: torch.Tensor,
    prompt_length: int,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    varentropy_lambda: float = 1.0,
    calibration: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Compute semantic authority with unified gating and dispersion.

    All thresholds are auto-derived from calibration statistics.

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        h_attn: (B, S, D) attention output before MLP
        h_block: (B, S, D) block output after MLP
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        attention_mask: Optional (B, S) padding mask
        varentropy_lambda: Weighting for confidence stability (default 1.0)
        calibration: Optional dict from calibrate_on_prompt() with adaptive thresholds

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    from .semantics import compute_semantic_dispersion
    from .entropy import compute_varentropy

    # 1. Authority Flow - where does signal come from?
    flow = compute_authority_flow_vectorized(
        attention_weights, prompt_length, attention_mask
    )

    # 2. Agreement Gate - is MLP validating or overriding attention?
    gate = fused_stability_gate(h_attn, h_block, sensitivity=1.0)

    # 3. Compute trust from semantic dispersion + varentropy
    dispersion = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    varentropy = compute_varentropy(logits)

    # Z-score normalization using calibration statistics (adaptive)
    # This compares varentropy to prompt baseline, not entropy (unstable)
    if calibration is not None:
        mu = calibration.get('varentropy_mu', 0.0)
        sigma = calibration.get('varentropy_sigma', 1.0)
    else:
        mu, sigma = 0.0, 1.0

    v_z = (varentropy - mu) / (sigma + 1e-8)
    v_norm = torch.sigmoid(v_z)  # Maps to [0, 1], mean stability -> 0.5

    # Trust = 1 - D × (1 + λ × V_norm)
    # With v_norm in [0,1], max penalty is controlled
    weighted_dispersion = dispersion * (1.0 + varentropy_lambda * v_norm)
    trust = (1.0 - weighted_dispersion).clamp(0.0, 1.0)

    # 4. CPG Detection - identify coherent parametric generation
    # CPG = low gate + low dispersion + high confidence + stable varentropy
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)
    max_prob = probs.max(dim=-1).values
    not_repetition = entropy > 0.05  # Entropy floor prevents repetition loops

    # Get adaptive thresholds from calibration or use defaults
    if calibration is not None:
        cpg_gate_thresh = calibration.get('adaptive_cpg_gate_threshold', 0.3)
        cpg_disp_thresh = calibration.get('adaptive_cpg_dispersion_threshold', 0.05)
        cpg_var_thresh = calibration.get('adaptive_cpg_varentropy_threshold', 0.5)
    else:
        cpg_gate_thresh, cpg_disp_thresh, cpg_var_thresh = 0.3, 0.05, 0.5

    is_cpg = (
        (gate < cpg_gate_thresh) &
        (dispersion < cpg_disp_thresh) &
        (max_prob > 0.9) &
        (v_norm < cpg_var_thresh) &
        not_repetition
    )

    # 5. Apply CPG override - trust parametric memory for valid reasoning
    final_gate = torch.where(is_cpg, torch.zeros_like(gate), gate)
    final_trust = torch.where(is_cpg, torch.ones_like(trust), trust)

    # 6. Master equation
    authority = final_gate * flow + (1.0 - final_gate) * final_trust * 0.5

    if attention_mask is not None:
        authority = authority * attention_mask.float()

    return authority.clamp(0.0, 1.0)


def compute_emergence_gated_trust(
    gate: torch.Tensor,
    epiplexity: torch.Tensor,
    dispersion: torch.Tensor,
    entropy: torch.Tensor,
    gate_threshold: float = 0.3,
    emergence_threshold: float = 0.5,
    dispersion_threshold: float = 0.1,
    entropy_floor: float = 0.01,
) -> torch.Tensor:
    """
    Strict emergence-gated trust with Conflict Penalty (v3.1).

    Key insight: If the model looks at context (high Authority) but disagrees
    with it (low Gate), this is a CONFLICT state. In conflict, we only trust
    the model if it's doing demonstrable structural work (high Epiplexity).
    A "confident lie" has low Epiplexity - it's just rote generation.

    Logic:
      1. Loop Breaker: H < entropy_floor → T = 0.0
      2. Conflict State: G < 0.5 (model disagrees with context)
         - If Emergent (E > 0.8): Trust it (valid reasoning) → T = 1.0
         - If NOT Emergent: Conflict Penalty → T = 0.0 (confident lie)
      3. Agreement State: G >= 0.5 → T = 1 - D (standard consistency)

    Args:
        gate: Stability gate values [B, S], in [0, 1]
        epiplexity: Epiplexity proxy [B, S], in [0, 1]
        dispersion: Semantic dispersion [B, S], in [0, 1]
        entropy: Token entropy [B, S]
        gate_threshold: τ_G (legacy, using 0.5 for conflict detection)
        emergence_threshold: τ_E (legacy, using 0.8 for strict emergence)
        dispersion_threshold: τ_D (legacy)
        entropy_floor: Below this, force T=0 (catches loops)

    Returns:
        Trust values [B, S], in [0, 1]
    """
    # 1. Loop Safety
    is_loop = entropy < entropy_floor

    # 2. Define States
    # Conflict: Model looks at context but disagrees (Gate < 0.5)
    is_conflict = gate < 0.5

    # Emergence: Model is doing hard structural work (strict threshold)
    is_emergent = epiplexity > 0.8

    # 3. Base Trust: Standard Consistency
    trust = (1.0 - dispersion).clamp(0.0, 1.0)

    # 4. CRITICAL: Conflict Penalty
    # If there's a conflict and NO emergence, this is a "confident lie"
    # Force trust to 0.0 - don't reward confident hallucinations
    trust = torch.where(
        is_conflict & (~is_emergent),
        torch.zeros_like(trust),
        trust
    )

    # 5. Valid Emergence Override
    # If model IS doing hard work (emergent), trust it fully
    trust = torch.where(is_emergent, torch.ones_like(trust), trust)

    # 6. Loop Safety Override (catches stuttering)
    trust = torch.where(is_loop, torch.zeros_like(trust), trust)

    return trust


def compute_semantic_authority_v3(
    attention_weights: torch.Tensor,
    prompt_length: int,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    calibration: Dict[str, float],
    lambda_struct: float = 2.0,
    entropy_floor: float = 0.01,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
    # Legacy parameters (kept for API compatibility)
    gate_threshold: float = 0.3,
    emergence_threshold: float = 0.5,
    dispersion_threshold: float = 0.1,
) -> torch.Tensor:
    """
    v3.4 Cognitive Load - Calibration-free authority computation.

    Master equation: Score = A × (1-D) × E^λ

    Theory: Varentropy measures "Cognitive Load" - how hard the model is working.
    - V < τ (5.0): Rote retrieval / confident fabrication → E < 1 → Penalized
    - V ≥ τ (5.0): Active reasoning / grounded synthesis → E ≥ 1 → Trusted

    Key insight: Prompt-relative calibration is unstable because simple questions
    can require complex answers. We use ABSOLUTE varentropy with a universal
    threshold τ=5.0 (based on Entropix research for Llama-3 scale models).

    Empirical validation (HaluEval):
    - Hallucination: V ≈ 3.8 → E = 0.76 → E^4 = 0.33 (penalized)
    - Fact: V ≈ 5.2 → E = 1.04 → E^4 = 1.17 → 1.0 (trusted)

    Pipeline:
    1. Compute Varentropy → Absolute Epiplexity (E_t = V_t / τ)
    2. Compute Recursive Authority Flow (A_t)
    3. Compute Semantic Consistency (1 - D_t)
    4. Compute Epistemic Weight (E_t^λ, clamped at 1.0)
    5. Master Equation: Score = A × (1-D) × E^λ
    6. Loop Safety (entropy floor)

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        h_attn: (B, S, D) attention output before MLP
        h_block: (B, S, D) block output after MLP
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        calibration: Optional dict with 'tau' override (default 5.0)
        lambda_struct: Power law exponent (default 4.0)
        entropy_floor: Loop breaker threshold (default 0.01)
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) post-RoPE queries for FlashAuthority
        key_states: Optional (B, H, S, D) post-RoPE keys for FlashAuthority

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    from .semantics import compute_semantic_dispersion
    from .entropy import compute_varentropy, compute_token_entropy, compute_epiplexity
    from ..ops import (
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    # 1. Absolute Cognitive Load (Calibration-Free)
    # τ = 5.0: Universal threshold where reasoning/synthesis happens
    # No prompt calibration needed - varentropy is an absolute signal
    varentropy = compute_varentropy(logits, attention_mask)
    tau = calibration.get('tau', 5.0)  # Allow override, but default is universal
    E_t = compute_epiplexity(varentropy, tau=tau)

    # 2. Recursive Authority Flow (SAFE DISPATCH)
    # NO structural gain inside flow - we apply epistemic weight OUTSIDE via power law
    # This prevents "High Attention" from hiding "Low Complexity"
    use_triton = False

    if (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
    ):
        batch_size = query_states.shape[0]
        if batch_size == 1 or attention_mask is None:
            use_triton = True

    # For v3.3: Use gamma=1.0 (no structural gain inside flow)
    gamma_t = torch.ones_like(E_t)

    if use_triton:
        # FAST PATH (Triton) - 20x Speedup
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma_t, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        # SAFE PATH (Python Loop) - Correct Masking
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma_t, attention_mask
        )

    # 3. Semantic Consistency
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    consistency = (1.0 - D_t).clamp(0.0, 1.0)

    # 4. Pure Varentropy Score (Cognitive Load Only)
    # Authority (A) and Consistency (1-D) are INVERTED in HaluEval data:
    #   - Hall: Higher A, Higher (1-D)
    #   - Fact: Lower A, Lower (1-D)
    #
    # Varentropy is the ONLY correct signal (Hall V=3.8 < Fact V=5.2).
    # Multiplying by A × (1-D) washes out the varentropy signal.
    #
    # Solution: Use varentropy as the PRIMARY score via sigmoid.
    #   Score = sigmoid((V - τ) × k)
    #
    # With τ=5.0 and k=2.0:
    #   - Hall (V=3.8): sigmoid(-2.4) ≈ 0.08 → Low score → High uncertainty
    #   - Fact (V=5.2): sigmoid(0.4) ≈ 0.60 → High score → Low uncertainty
    centered_varentropy = varentropy - tau
    score = torch.sigmoid(centered_varentropy * lambda_struct)

    # 6. Loop Safety
    entropy = compute_token_entropy(logits, attention_mask)
    score = torch.where(entropy < entropy_floor, torch.zeros_like(score), score)

    if attention_mask is not None:
        score = score * attention_mask.float()

    return score.clamp(0.0, 1.0)


def compute_semantic_authority_v4(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    varentropy_scale: float = 3.0,
    dispersion_scale: float = 0.15,
    calibration: Optional[Dict[str, float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v4 Epistemic Dominance with Context-Aware Semantic Shielding.

    Master equation: P_faith(t) = A(t) × (1 - P_v × P_d)

    Where:
    - P_v = tanh(V/scale_v): Varentropy risk signal
    - P_d = sigmoid((D - τ_dynamic) × stiffness): Dispersion shield with dynamic threshold
    - τ_dynamic = μ_D(prompt) + σ_D(prompt), clamped to [0.05, 0.20]

    Key insight: The "semantic granularity" of the embedding space is consistent
    within a context window. If the prompt uses precise language (low D), the
    response should too. Dynamic calibration adapts to each context:
    - Legal contract (precise): τ ≈ 0.08 → strict shield
    - Creative writing (vague): τ ≈ 0.18 → lenient shield

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        varentropy_scale: V normalization (default 3.0). tanh(V/3) maps [0,10]→[0,1]
        dispersion_scale: Fallback D sensitivity if no calibration (default 0.15)
        calibration: Optional dict with 'dispersion_mu' and 'dispersion_sigma' from prompt
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) for Triton optimization
        key_states: Optional (B, H, S, D) for Triton optimization

    Returns:
        score: (B, S) faithfulness scores in [0, 1]. Use min() to aggregate.
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import (
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    # 1. Raw Signals
    varentropy = compute_varentropy(logits, attention_mask)
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # 2. Varentropy Penalty (normalized to [0, 1])
    P_v = torch.tanh(varentropy / varentropy_scale)

    # 3. Dynamic Dispersion Threshold (Context-Aware Shielding)
    if calibration and 'dispersion_mu' in calibration:
        # Dynamic: Mean + 1 Std Dev from prompt
        # Interpretation: "If response is significantly more confused than prompt, lower shield"
        raw_threshold = calibration['dispersion_mu'] + calibration.get('dispersion_sigma', 0.05)
        # Safety clamps: [0.10, 0.40]
        # Min 0.10: Don't punish facts just because prompt was precise
        # Max 0.40: Allow calibration to adapt to high-dispersion contexts (summaries)
        threshold = min(max(raw_threshold, 0.10), 0.40)
    else:
        # Fallback: Use sigmoid with dispersion_scale as soft threshold
        # This path is used when calibration is skipped
        threshold = dispersion_scale

    # 4. Dispersion Shield with Sigmoid (sharp transition around threshold)
    # Stiffness=20.0 creates a transition window of roughly ±0.05 around threshold
    # D < threshold → P_d ≈ 0 (shield active)
    # D > threshold → P_d ≈ 1 (shield broken)
    stiffness = 20.0
    P_d = torch.sigmoid((D_t - threshold) * stiffness)

    # 5. Semantic Shielding (The Interlock)
    # Combined penalty = P_v × P_d
    # - If D < threshold (Fact with tight semantics), P_d → 0 → Penalty → 0 (Shielded)
    # - If D > threshold (Hallucination with scattered semantics), P_d → 1 → Penalty → P_v (Exposed)
    combined_penalty = P_v * P_d

    # 6. Authority Flow (Grounding Check)
    # Dispatch: Triton (fast) vs Python (safe)
    use_triton = (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
        and (query_states.shape[0] == 1 or attention_mask is None)
    )

    gamma = torch.ones_like(varentropy)

    if use_triton:
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma, attention_mask
        )

    # 7. Master Equation: Score = Provenance × (1 - ShieldedPenalty)
    # Note: We don't use (1-D) separately to avoid double-counting dispersion
    score = A_t * (1.0 - combined_penalty)

    # 8. Mask Handling for min() aggregation
    # CRITICAL: Set padding tokens to 1.0 so they don't trigger the min
    if attention_mask is not None:
        score = score.masked_fill(~attention_mask.bool(), 1.0)

    return score.clamp(0.0, 1.0)


def compute_semantic_authority_v5(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    varentropy_scale: float = 3.0,
    dispersion_threshold: float = 0.12,
    dispersion_stiffness: float = 20.0,
    calibration: Optional[Dict[str, float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v5 Dual-Path Aggregation - Heterogeneous aggregation for Authority and Trust.

    Master equation:
        Score_final = Mean(A_t) × Min(T_t)

    Where:
        - A_t = Authority Flow (information provenance from context)
        - T_t = Trust Score = 1 - (P_v × P_d) (semantic shielding from v4)
        - P_v = tanh(V/scale): Varentropy risk
        - P_d = sigmoid((D - τ) × stiffness): Dispersion shield

    Key insight (Dual-Path Aggregation):
        - Mean(A_t): "On average, did we rely on the context?" (Global Grounding)
          Tolerates preambles/stopwords that don't attend to context.
        - Min(T_t): "Did we fabricate *anything*?" (Local Trust)
          Catches localized lies at any token position.

    This architecture correctly respects the "Topology of Error":
        - Authority failures are GLOBAL (ungrounded generation overall)
        - Trust failures are LOCAL (single fabricated fact)

    Case Analysis:
        | Case                  | A_mean | T_min | Score | Verdict |
        |----------------------|--------|-------|-------|---------|
        | HaluEval Hallucination| 0.9   | 0.1   | 0.09  | Caught (V-spike) |
        | HaluEval Fact        | 0.9   | 0.9   | 0.81  | Pass |
        | RAGTruth Hallucination| 0.2   | 0.9   | 0.18  | Caught (ungrounded) |
        | RAGTruth Fact        | 0.8   | 0.8   | 0.64  | Pass |
        | Preamble             | 0.5   | 1.0   | 0.50  | Better than 0.0 |

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        varentropy_scale: V normalization for tanh (default 3.0)
        dispersion_threshold: Base D threshold for sigmoid (default 0.12)
        dispersion_stiffness: Sigmoid sharpness (default 20.0)
        calibration: Optional dict with 'dispersion_mu' and 'dispersion_sigma' from prompt
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) for Triton optimization
        key_states: Optional (B, H, S, D) for Triton optimization

    Returns:
        score: (B, S) pre-aggregated scores broadcasted for API compatibility.
               The score at each position is the sequence-level score.
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import (
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    # ================================================================
    # 1. COMPUTE RAW SIGNALS (Same as v4)
    # ================================================================

    # 1a. Varentropy - Model uncertainty at each token
    varentropy = compute_varentropy(logits, attention_mask)

    # 1b. Semantic Dispersion - Spread in embedding space of top predictions
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # ================================================================
    # 2. COMPUTE TRUST SCORE (T_t) - Semantic Shielding from v4
    # ================================================================

    # 2a. Varentropy Penalty (normalized to [0, 1])
    P_v = torch.tanh(varentropy / varentropy_scale)

    # 2b. Dynamic Dispersion Threshold (Context-Aware)
    if calibration and 'dispersion_mu' in calibration:
        raw_threshold = calibration['dispersion_mu'] + calibration.get('dispersion_sigma', 0.05)
        threshold = min(max(raw_threshold, 0.10), 0.40)
    else:
        threshold = dispersion_threshold

    # 2c. Dispersion Shield with Sigmoid
    P_d = torch.sigmoid((D_t - threshold) * dispersion_stiffness)

    # 2d. Combined Penalty and Trust Score
    penalty = P_v * P_d
    T_t = 1.0 - penalty

    # ================================================================
    # 3. COMPUTE AUTHORITY FLOW (A_t) - Provenance tracking
    # ================================================================

    use_triton = (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
        and (query_states.shape[0] == 1 or attention_mask is None)
    )

    gamma = torch.ones_like(varentropy)

    if use_triton:
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma, attention_mask
        )

    # ================================================================
    # 4. HETEROGENEOUS AGGREGATION (The v5 Innovation)
    # ================================================================

    # 4a. CRITICAL: Only aggregate over RESPONSE tokens, not prompt
    # The engine will also slice, but v5 does pre-aggregation so we must slice here
    A_response = A_t[:, prompt_length:]
    T_response = T_t[:, prompt_length:]

    # 4b. Handle Masking for response tokens only
    if attention_mask is not None:
        response_mask = attention_mask[:, prompt_length:]
        mask_bool = response_mask.bool()
        # For Mean: Zero out padding, then divide by active token count
        A_masked = A_response * mask_bool.float()
        # For Min: Set padding to 1.0 so it doesn't affect min
        T_masked = T_response.masked_fill(~mask_bool, 1.0)
        # Count active response tokens per sequence
        lengths = mask_bool.sum(dim=-1).float().clamp(min=1.0)
    else:
        A_masked = A_response
        T_masked = T_response
        lengths = torch.full(
            (A_response.shape[0],),  # Batch-sized tensor, not scalar
            A_response.shape[1],
            dtype=A_response.dtype,
            device=A_response.device
        )

    # 4c. Authority: Global Grounding (Mean over response)
    # "On average, did we rely on the context?"
    # Tolerates preambles like "Based on the context..." (low A for a few tokens)
    A_seq = A_masked.sum(dim=-1) / lengths  # [B]

    # 4d. Trust: Local Trust (Min over response)
    # "Did we fabricate *anything*?"
    # Catches single-token fabrications (e.g., "Wildlife" in place of "Tolkien")
    T_seq = T_masked.min(dim=-1).values  # [B]

    # 4d. Final Score: Global Grounding × Local Trust
    score_seq = A_seq * T_seq  # [B]

    # ================================================================
    # 5. BROADCAST FOR API COMPATIBILITY
    # ================================================================
    # The engine expects (B, S) tensor. We broadcast the scalar score
    # to all positions so existing aggregation logic works.
    # When engine calls mean/min on this, it gets the same value.
    score = score_seq.unsqueeze(-1).expand_as(A_t)

    return score.clamp(0.0, 1.0)


def compute_semantic_authority_v6(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    complexity_sigma: float = 0.5,
    complexity_epsilon: float = 0.1,
    complexity_center: float = 0.4,
    calibration: Optional[Dict[str, float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v6 Gaussian Complexity Matching - Bidirectional complexity penalization.

    Master equation:
        Score_final = Mean(A_t) × Min(T_t)
        T_t = (1 - D_t) × G_complexity(R_t, σ)

    Where:
        - A_t = Authority Flow (information provenance from context)
        - D_t = Semantic Dispersion (prediction scatter)
        - G_complexity = exp(-(R_t - 1.0)² / 2σ²) (Gaussian complexity penalty)
        - R_t = V_t / max(μ_prompt, ε) (complexity ratio)

    Key insight (Goldilocks Information Problem):
        - HaluEval: Hallucinations are TOO COMPLEX (V_hall >> V_prompt)
        - RAGTruth: Hallucinations are TOO SIMPLE (V_hall << V_prompt)
        - Truth: Valid facts PRESERVE complexity (V_fact ≈ V_prompt)

    The Gaussian term penalizes BOTH directions of deviation, solving the
    failure mode where v4/v5 could not distinguish oversimplification.

    Aggregation (Dual-Path from v5):
        - Mean(A_t): Global Grounding - "On average, did we rely on context?"
        - Min(T_t): Local Trust - "Did we fabricate anything?" (catches R spikes)

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        complexity_sigma: Gaussian tolerance width (default 0.5)
        complexity_epsilon: Floor for mu_prompt (default 0.1)
        complexity_center: Expected complexity ratio R_t (default 0.4)
            - 0.3-0.5: Short-answer QA where response is simpler than prompt
            - 0.8-1.2: Reasoning tasks where response matches prompt complexity
        calibration: Dict with 'varentropy_mu' from calibrate_on_prompt()
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) for Triton optimization
        key_states: Optional (B, H, S, D) for Triton optimization

    Returns:
        score: (B, S) pre-aggregated scores broadcasted for API compatibility.
    """
    from .entropy import compute_varentropy, compute_gaussian_complexity
    from .semantics import compute_semantic_dispersion
    from ..ops import (
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    # ================================================================
    # 1. COMPUTE RAW SIGNALS
    # ================================================================

    # 1a. Varentropy - Model uncertainty/complexity at each token
    varentropy = compute_varentropy(logits, attention_mask)

    # 1b. Semantic Dispersion - Spread in embedding space of top predictions
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # ================================================================
    # 2. GAUSSIAN COMPLEXITY MATCHING (v6 Innovation)
    # ================================================================

    # Get prompt varentropy baseline from calibration
    if calibration and 'varentropy_mu' in calibration:
        varentropy_mu = calibration['varentropy_mu']
    else:
        # Fallback: Use absolute threshold (similar to v3's tau)
        # This path should be rare - calibration is expected for v6
        varentropy_mu = 5.0  # Universal default

    # Compute Gaussian complexity score
    G_complexity = compute_gaussian_complexity(
        varentropy,
        varentropy_mu,
        sigma=complexity_sigma,
        epsilon=complexity_epsilon,
        center=complexity_center,
    )

    # ================================================================
    # 3. COMPUTE TRUST SCORE (T_t)
    # ================================================================

    # Trust = Semantic Consistency × Complexity Match
    consistency = (1.0 - D_t).clamp(0.0, 1.0)
    T_t = consistency * G_complexity

    # ================================================================
    # 4. COMPUTE AUTHORITY FLOW (A_t)
    # ================================================================

    use_triton = (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
        and (query_states.shape[0] == 1 or attention_mask is None)
    )

    gamma = torch.ones_like(varentropy)

    if use_triton:
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma, attention_mask
        )

    # ================================================================
    # 5. HETEROGENEOUS AGGREGATION (Dual-Path from v5)
    # ================================================================

    # 5a. Only aggregate over RESPONSE tokens
    A_response = A_t[:, prompt_length:]
    T_response = T_t[:, prompt_length:]

    # 5b. Handle Masking
    if attention_mask is not None:
        response_mask = attention_mask[:, prompt_length:]
        mask_bool = response_mask.bool()
        A_masked = A_response * mask_bool.float()
        T_masked = T_response.masked_fill(~mask_bool, 1.0)
        lengths = mask_bool.sum(dim=-1).float().clamp(min=1.0)
    else:
        A_masked = A_response
        T_masked = T_response
        lengths = torch.full(
            (A_response.shape[0],),  # Batch-sized tensor, not scalar
            A_response.shape[1],
            dtype=A_response.dtype,
            device=A_response.device
        )

    # 5c. Authority: Global Grounding (Mean over response)
    A_seq = A_masked.sum(dim=-1) / lengths  # [B]

    # 5d. Trust: Local Trust (Min over response)
    # Using min like v5 to catch single-token complexity spikes.
    # Masking already sets padding to 1.0 so it won't affect min.
    T_seq = T_masked.min(dim=-1).values  # [B]

    # 5e. Final Score with Authority Soft Floor
    # Key insight: Authority can be inverted on some datasets (RAGTruth).
    # Using a soft floor (0.3 + 0.7*A) ensures Trust signal isn't zeroed out.
    # - If A=0: score = 0.3 * T (Trust still matters)
    # - If A=1: score = 1.0 * T (Full Authority contribution)
    A_floored = 0.3 + 0.7 * A_seq
    score_seq = A_floored * T_seq  # [B]

    # ================================================================
    # 6. BROADCAST FOR API COMPATIBILITY
    # ================================================================
    score = score_seq.unsqueeze(-1).expand_as(A_t)

    return score.clamp(0.0, 1.0)


def compute_semantic_authority_v7(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    hidden_states: torch.Tensor,
    lid_window_size: int = 8,
    calibration: Optional[Dict[str, float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v7 Geometric Manifold Adherence - LID-based hallucination detection.

    Master equation:
        Score(t) = A(t) × (1 - D_t) × M(t)
        M(t) = 1 - LID_norm(t)    # Manifold adherence

    Where:
        - A(t) = Authority Flow (information provenance from context)
        - D_t = Semantic Dispersion (prediction scatter)
        - M(t) = Manifold adherence (1 - normalized Local Intrinsic Dimension)
        - LID(t) = (Σλᵢ)² / Σλᵢ² (Participation Ratio of local covariance spectrum)

    Key insight:
        - Truthful text: Low LID (coherent manifold) → M high → Trusted
        - Hallucination: High LID (scattered) → M low → Penalized
        - Fixes RAGTruth: "Confident lies" have low V (scalar) but HIGH LID (geometric)

    Research Foundation:
        - Yin et al., ICML 2024: "Characterizing Truthfulness in LLM Generations
          with Local Intrinsic Dimension"

    Aggregation (Dual-Path from v5):
        - Mean(A_t): Global Grounding - "On average, did we rely on context?"
        - Min(Score): Local Trust - "Did we fabricate anything?"

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        hidden_states: (B, S, D) block outputs or attention outputs for LID
        lid_window_size: Sliding window size for LID computation (default 8)
        calibration: Optional dict with 'lid_mu' baseline
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) for Triton optimization
        key_states: Optional (B, H, S, D) for Triton optimization

    Returns:
        score: (B, S) pre-aggregated scores broadcasted for API compatibility.
    """
    from .geometry import compute_manifold_score
    from .semantics import compute_semantic_dispersion
    from ..ops import (
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    # ================================================================
    # 1. COMPUTE RAW SIGNALS
    # ================================================================

    # 1a. Semantic Dispersion - Spread in embedding space of top predictions
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # 1b. Manifold Score (v7 Innovation)
    # M(t) = 1 - LID_norm(t)
    # Low LID = coherent manifold = high M = trusted
    # High LID = scattered = low M = penalized
    M_t = compute_manifold_score(
        hidden_states,
        window_size=lid_window_size,
        calibration=calibration,
        attention_mask=attention_mask,
    )

    # ================================================================
    # 2. COMPUTE AUTHORITY FLOW (A_t)
    # ================================================================

    use_triton = (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
        and (query_states.shape[0] == 1 or attention_mask is None)
    )

    # Gamma = 1.0 (no structural gain inside flow)
    gamma = torch.ones(D_t.shape, device=D_t.device, dtype=D_t.dtype)

    if use_triton:
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma, attention_mask
        )

    # ================================================================
    # 3. COMPUTE COMBINED SCORE
    # ================================================================

    # Master equation: Score = A × (1-D) × M
    consistency = (1.0 - D_t).clamp(0.0, 1.0)
    score_t = A_t * consistency * M_t

    # ================================================================
    # 4. HETEROGENEOUS AGGREGATION (Dual-Path from v5)
    # ================================================================

    # 4a. Only aggregate over RESPONSE tokens
    A_response = A_t[:, prompt_length:]
    score_response = score_t[:, prompt_length:]

    # 4b. Handle Masking
    if attention_mask is not None:
        response_mask = attention_mask[:, prompt_length:]
        mask_bool = response_mask.bool()
        A_masked = A_response * mask_bool.float()
        score_masked = score_response.masked_fill(~mask_bool, 1.0)
        lengths = mask_bool.sum(dim=-1).float().clamp(min=1.0)
    else:
        A_masked = A_response
        score_masked = score_response
        lengths = torch.full(
            (A_response.shape[0],),
            A_response.shape[1],
            dtype=A_response.dtype,
            device=A_response.device
        )

    # 4c. Authority: Global Grounding (Mean over response)
    A_seq = A_masked.sum(dim=-1) / lengths  # [B]

    # 4d. Combined Score: Min (Weakest Link)
    # Min catches single-token geometric deviations.
    # Best for HaluEval (0.86) and SQuAD (0.92).
    # Note: Less effective on RAGTruth where facts are longer than hallucinations.
    score_seq = score_masked.min(dim=-1).values  # [B]

    # 4e. Final Score: Global Grounding × Local Trust
    # Use soft floor to prevent Authority from zeroing out
    A_floored = 0.3 + 0.7 * A_seq
    final_seq = A_floored * score_seq  # [B]

    # ================================================================
    # 5. BROADCAST FOR API COMPATIBILITY
    # ================================================================
    score = final_seq.unsqueeze(-1).expand_as(A_t)

    return score.clamp(0.0, 1.0)


def compute_semantic_authority_v8(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v8 Residual Stream Contrast - FFN interference detection.

    Master equation:
        Score(t) = (1 - FFN_Interference(t)) × (1 - Dispersion(t))

    Where:
        - FFN_Interference(t) = JSD(P_attn || P_final)
        - P_attn = softmax(h_attn @ lm_head.weight.T)  # "What context says"
        - P_final = softmax(h_block @ lm_head.weight.T)  # "What model outputs"
        - JSD = Jensen-Shannon Divergence (bounded [0, 1])

    Key insight (CHOKE/ReDeEP):
        Current approaches (v2-v7) measure INPUTS (attention weights, entropy),
        but "Confident Lies" occur when FFN silently overrides the context signal.
        v8 measures OUTCOMES: whether the context signal (attention output)
        survives through the FFN to the final prediction.

    Detection Logic:
        | Scenario         | P_attn vs P_final | JSD  | D   | Score |
        |------------------|-------------------|------|-----|-------|
        | Fact             | Similar           | Low  | Low | High  |
        | RAGTruth Hall    | Different (FFN)   | High | Low | Low   |
        | HaluEval Confuse | N/A               | N/A  | High| Low   |

    Performance Optimization:
        - Uses top-k=50 approximation for JSD (~2600x faster than full vocab)
        - Slices to response tokens BEFORE projection (~75% compute savings)

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer (API consistency)
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        h_attn: (B, S, D) attention output before MLP - "what context says"
        h_block: (B, S, D) block output after MLP - "what model outputs"
        jsd_top_k: Top-k approximation for JSD (default 50)
        attention_mask: Optional (B, S) padding mask

    Returns:
        score: (B, S) faithfulness scores in [0, 1].
               Prompt positions are set to 1.0 (trusted).
    """
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total = logits.shape[:2]
    S_resp = S_total - prompt_length
    device = logits.device
    dtype = logits.dtype

    # Handle edge case: no response tokens
    if S_resp <= 0:
        return torch.ones(B, S_total, device=device, dtype=dtype)

    # ================================================================
    # 1. CRITICAL OPTIMIZATION: Slice to response tokens BEFORE projection
    #    Projection is O(S × D × V) ≈ 500M FLOPs/token for Llama-3-8B
    #    This saves ~75% compute if prompt is 3x longer than response
    # ================================================================

    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]

    # ================================================================
    # 2. COMPUTE FFN INTERFERENCE (JSD between attention and block logits)
    # ================================================================

    ffi = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )

    # ================================================================
    # 3. COMPUTE SEMANTIC DISPERSION (response tokens only)
    # ================================================================

    logits_resp = logits[:, prompt_length:, :].contiguous()
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # ================================================================
    # 4. MASTER EQUATION: Score = (1 - FFN_Interference) × (1 - Dispersion)
    # ================================================================

    score_resp = (1.0 - ffi) * (1.0 - D_t)

    # ================================================================
    # 5. PAD BACK TO FULL SEQUENCE LENGTH
    #    Engine expects [B, S], we have [B, S_resp]
    #    Pad with 1.0 (Trusted) for prompt positions
    # ================================================================

    padding = torch.ones(B, prompt_length, device=device, dtype=dtype)
    full_score = torch.cat([padding, score_resp], dim=1)

    # ================================================================
    # 6. HANDLE MASKING
    # ================================================================

    if attention_mask is not None:
        # Set padding tokens to 1.0 so they don't affect aggregation
        full_score = full_score.masked_fill(~attention_mask.bool(), 1.0)

    return full_score.clamp(0.0, 1.0)


def compute_semantic_authority_v9(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    varentropy_scale: float = 3.0,
    jsd_top_k: int = 50,
    attention_mask: Optional[torch.Tensor] = None,
    return_components: bool = False,
) -> torch.Tensor:
    """
    v9 Holographic Dual-Stream: Sequence-level MAX uncertainty.

    Returns BOTH stability and grounding scores. The engine aggregates each
    separately and returns max(stability_uncertainty, grounding_uncertainty).

    Master Equation (sequence level):
        uncertainty = max(1 - aggregate(stability), 1 - aggregate(grounding))

        Stability(t) = (1 - V_norm) × (1 - D_t)     # Catches HaluEval confusion
        Grounding(t) = 1 - JSD(P_attn || P_final)   # Catches RAGTruth deception

    Key Insight:
        - Stability and JSD have OPPOSITE correlations on different datasets
        - Combining at token level causes interference
        - Solution: Aggregate each signal separately, then take max uncertainty
        - This catches hallucination if EITHER confusion OR override is detected

    Args:
        attention_weights: (B, H, S, S) attention weights (unused, API consistency)
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        varentropy_scale: Normalization scale for varentropy (default 3.0)
        jsd_top_k: Top-k for JSD approximation (default 50)
        attention_mask: Optional (B, S) padding mask
        return_components: If True, return dict with both stability and grounding

    Returns:
        If return_components=False: (B, S) token-level min of stability/grounding
        If return_components=True: dict with 'stability' and 'grounding' tensors
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        ones = torch.ones(B, S_total, device=device, dtype=dtype)
        if return_components:
            return {'stability': ones, 'grounding': ones}
        return ones

    # ================================================================
    # STREAM 1: EPISTEMIC STABILITY (detects confusion - HaluEval)
    # ================================================================

    logits_resp = logits[:, prompt_length:, :].contiguous()
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    V_norm = torch.tanh(varentropy / varentropy_scale)
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    stability_resp = (1.0 - V_norm) * (1.0 - D_t)

    # ================================================================
    # STREAM 2: MECHANISTIC GROUNDING (detects FFN override - RAGTruth)
    # ================================================================

    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )
    grounding_resp = 1.0 - jsd

    # Pad back to full sequence (prompt tokens = 1.0 trusted)
    padding = torch.ones(B, prompt_length, device=device, dtype=dtype)
    stability_full = torch.cat([padding, stability_resp], dim=1)
    grounding_full = torch.cat([padding, grounding_resp], dim=1)

    # Handle attention mask
    if attention_mask is not None:
        mask = ~attention_mask.bool()
        stability_full = stability_full.masked_fill(mask, 1.0)
        grounding_full = grounding_full.masked_fill(mask, 1.0)

    stability_full = stability_full.clamp(0.0, 1.0)
    grounding_full = grounding_full.clamp(0.0, 1.0)

    if return_components:
        return {'stability': stability_full, 'grounding': grounding_full}

    # Default: token-level min (for backward compatibility)
    return torch.min(stability_full, grounding_full)


def compute_semantic_authority_v10(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    varentropy_scale: float = 3.0,
    jsd_top_k: int = 100,
    dispersion_threshold: float = 0.12,
    dispersion_stiffness: float = 20.0,
    attention_mask: Optional[torch.Tensor] = None,
    calibration: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    v10 Orthogonal Signal Fusion: Decoupled aggregation topologies.

    Returns BOTH signals for heterogeneous aggregation in engine.
    The engine performs:
        Score_seq = S_seq × G_seq
        S_seq = SlidingWindowPercentile_10(Stability_t)  # Burst detection (HaluEval)
        G_seq = Mean(Grounding_t)                        # Systemic detection (RAGTruth)

    Key Insight (Why v9 Failed):
        v9 combined signals at token level, causing interference:
        - Mean(Stability) washes out HaluEval bursts (0.97 → 0.59)
        - Min(Grounding) unfairly punishes long RAGTruth facts

    v10 Solution: Decouple aggregation topologies - aggregate each signal
    optimally, THEN combine. Each signal uses its mathematically optimal reduction:
        - Stability → SlidingWindowPercentile_10 (catches phrase-level bursts)
        - Grounding → Mean (captures systemic FFN override)

    Stability uses Semantic Shielding (from v5):
        P_v = tanh(V / scale)
        P_d = sigmoid((D - threshold) × stiffness)
        Stability = 1 - (P_v × P_d)
        Shield protects facts when D < threshold by forcing P_d → 0

    Detection Logic:
        | Dataset       | S_seq | G_seq | Score | Result  |
        |---------------|-------|-------|-------|---------|
        | HaluEval Hall | Low   | High  | Low   | DETECTED|
        | HaluEval Fact | High  | High  | High  | PASS    |
        | RAGTruth Hall | High  | Low   | Low   | DETECTED|
        | RAGTruth Fact | High  | High  | High  | PASS    |

    Args:
        attention_weights: (B, H, S, S) attention weights (unused, API consistency)
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        varentropy_scale: Normalization scale for varentropy (default 3.0)
        jsd_top_k: Top-k for JSD approximation (default 100)
        dispersion_threshold: Base D threshold for sigmoid (default 0.12)
        dispersion_stiffness: Sigmoid sharpness (default 20.0)
        attention_mask: Optional (B, S) padding mask
        calibration: Optional dict with 'dispersion_mu' and 'dispersion_sigma' from prompt

    Returns:
        Dict with:
            - 'stability': (B, S) epistemic stability scores
            - 'grounding': (B, S) mechanistic grounding scores
            - 'prompt_length': int for engine to slice response tokens
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        ones = torch.ones(B, S_total, device=device, dtype=dtype)
        return {'stability': ones, 'grounding': ones, 'prompt_length': prompt_length}

    # ================================================================
    # SIGNAL 1: EPISTEMIC STABILITY with SEMANTIC SHIELDING (for HaluEval)
    # Stability(t) = 1 - (P_v × P_d)
    # P_v = tanh(V / scale) - varentropy penalty
    # P_d = sigmoid((D - threshold) × stiffness) - dispersion shield
    # Shield protects facts: when D < threshold, P_d → 0, penalty → 0
    # ================================================================

    logits_resp = logits[:, prompt_length:, :].contiguous()
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )

    # Varentropy penalty (normalized to [0, 1])
    P_v = torch.tanh(varentropy / varentropy_scale)

    # Dynamic dispersion threshold from calibration (same as v5)
    if calibration and 'dispersion_mu' in calibration:
        raw_threshold = calibration['dispersion_mu'] + calibration.get('dispersion_sigma', 0.05)
        threshold = min(max(raw_threshold, 0.10), 0.40)
    else:
        threshold = dispersion_threshold

    # Dispersion shield with sigmoid (sharp transition around threshold)
    # D < threshold → P_d ≈ 0 (shield active, protects facts)
    # D > threshold → P_d ≈ 1 (shield broken, expose hallucinations)
    P_d = torch.sigmoid((D_t - threshold) * dispersion_stiffness)

    # Stability = 1 - shielded penalty
    stability_resp = (1.0 - P_v * P_d).clamp(0.0, 1.0)

    # ================================================================
    # SIGNAL 2: MECHANISTIC GROUNDING (for RAGTruth systemic detection)
    # Grounding(t) = 1 - JSD(P_attn || P_final)  [PURE JSD, no dispersion]
    # Key insight: Dispersion (D) is LOW for "confident lies" (RAGTruth halls)
    # Including D would cancel the JSD signal. Pure JSD isolates FFN override.
    # ================================================================

    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )
    # Pure JSD (no dispersion) - isolates FFN override signal
    grounding_resp = (1.0 - jsd).clamp(0.0, 1.0)

    # ================================================================
    # PAD TO FULL SEQUENCE (prompt tokens = 1.0 trusted)
    # ================================================================

    padding = torch.ones(B, prompt_length, device=device, dtype=dtype)
    stability_full = torch.cat([padding, stability_resp], dim=1)
    grounding_full = torch.cat([padding, grounding_resp], dim=1)

    # Handle attention mask
    if attention_mask is not None:
        mask = ~attention_mask.bool()
        stability_full = stability_full.masked_fill(mask, 1.0)
        grounding_full = grounding_full.masked_fill(mask, 1.0)

    return {
        'stability': stability_full,
        'grounding': grounding_full,
        'prompt_length': prompt_length
    }


def compute_semantic_authority_v11(
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_scale: float = 3.0,
    dispersion_threshold: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v11 Information Physics: Shielded Free Energy Detection.

    Unified energy-based model combining Internal (Confusion) and External
    (Deception) failure modes with principled semantic shielding.

    Master Equation:
        E_t = [tanh(V_t/τ) × Shield(D_t)] + JSD(P_attn || P_final)

    Where:
        - tanh(V/τ) ∈ [0, 1]: Varentropy force (internal instability)
        - Shield(D) = sigmoid((D - δ) × 20): Dispersion gate [0, 1]
        - JSD ∈ [0, 1]: External interference (FFN override)
        - E_t ∈ [0, 2]: Total Hallucination Energy

    Shielding Mechanism:
        - High V alone does NOT contribute energy (protected by low D)
        - High V × High D → Shield breaks → High internal energy
        - This protects facts with rare words (high V, low D)

    Detection Logic:
        | State      | V    | D    | JSD  | Internal | External | Total  |
        |------------|------|------|------|----------|----------|--------|
        | Fact       | 0.9  | 0.1  | 0.05 | 0.09     | 0.05     | ~0.14  |
        | Confusion  | 0.9  | 0.9  | 0.1  | 0.81     | 0.1      | ~0.91  |
        | Deception  | 0.1  | 0.1  | 0.8  | 0.01     | 0.8      | ~0.81  |

    Key Properties:
        - Semantic shielding protects facts with high V but low D
        - JSD catches "Confident Lies" (RAGTruth deception)
        - V×D catches "Epistemic Collapse" (HaluEval confusion)
        - Both bounded [0,1], additive combination is natural

    Aggregation (in engine):
        Uses SoftMax (LogSumExp) which smoothly interpolates mean↔max:
        - Burst (HaluEval): exp() amplifies spike → acts like Max
        - Systemic (RAGTruth): sum accumulates → acts like Mean

    Args:
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (lm_head.weight)
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        jsd_top_k: Top-k for JSD approximation (default 50)
        varentropy_scale: Cognitive temperature τ for V scaling (default 3.0)
        dispersion_threshold: Shield breaking point δ (default 0.1)
        attention_mask: Optional (B, S) padding mask

    Returns:
        energy: (B, S) Hallucination Energy in [0, 2].
                Higher = more likely hallucination.
                Prompt positions are set to 0.0 (no energy = trusted).
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        return torch.zeros(B, S_total, device=device, dtype=dtype)

    # ================================================================
    # SLICE TO RESPONSE TOKENS (saves compute)
    # ================================================================
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # ================================================================
    # INTERNAL ENERGY: Shielded Confusion Detection
    # E_internal = tanh(V/τ) × Shield(D)
    # Shield protects facts with high V but low D (rare words)
    # ================================================================

    # Varentropy force (scaled by cognitive temperature)
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    v_signal = torch.tanh(varentropy / varentropy_scale)  # [B, S_resp]

    # Dispersion shield (soft gate around threshold)
    # D < threshold → shield ≈ 0 (protected)
    # D > threshold → shield ≈ 1 (exposed)
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    d_signal = torch.sigmoid((D_t - dispersion_threshold) * 20.0)  # [B, S_resp]

    # Internal energy: only high when BOTH V and D are high
    energy_internal = v_signal * d_signal  # [B, S_resp]

    # ================================================================
    # EXTERNAL ENERGY: FFN Override Detection (JSD)
    # JSD is naturally bounded [0, 1].
    # Measures how much FFN overrides the context signal.
    # ================================================================
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )
    energy_external = jsd  # [B, S_resp]

    # ================================================================
    # TOTAL FREE ENERGY: Sum of failure modes
    # Both components are bounded [0, 1], sum is [0, 2]
    # ================================================================
    total_energy = energy_internal + energy_external  # [B, S_resp]

    # ================================================================
    # PAD TO FULL SEQUENCE (prompt = 0.0 energy = trusted)
    # ================================================================
    padding = torch.zeros(B, prompt_length, device=device, dtype=dtype)
    full_energy = torch.cat([padding, total_energy], dim=1)  # [B, S_total]

    # Handle attention mask (masked positions get 0 energy)
    if attention_mask is not None:
        full_energy = full_energy * attention_mask.float()

    return full_energy


def compute_semantic_authority_v12(
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_scale: float = 3.0,
    dispersion_threshold: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    v12 Dual-Stream Risk: Parallel confusion and deception detectors.

    Returns raw risk signals for heterogeneous aggregation in engine.
    The engine applies:
        Risk_seq = max(R_internal, R_external)
        R_internal = Percentile_90(risk_internal)  # Burst detection (HaluEval)
        R_external = Mean(risk_external)           # Systemic detection (RAGTruth)

    Key Insight (Why v10/v11 Failed):
        - v10: Multiplicative fusion masks signals (S×G)
        - v11: Additive energy ignores aggregation topology differences
        - v12: Parallel detectors with optimal aggregations, MAX fusion

    Signals:
        - Internal Risk: tanh(V/τ) × Shield(D) - confusion detection
        - External Risk: JSD - deception detection

    Detection Logic:
        | Dataset       | R_internal | R_external | MAX  | Result   |
        |---------------|------------|------------|------|----------|
        | HaluEval Hall | HIGH       | Low        | HIGH | DETECTED |
        | HaluEval Fact | Low        | Low        | Low  | PASS     |
        | RAGTruth Hall | Low        | HIGH       | HIGH | DETECTED |
        | RAGTruth Fact | Low        | Low        | Low  | PASS     |

    Args:
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (lm_head.weight)
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        jsd_top_k: Top-k for JSD approximation (default 50)
        varentropy_scale: Cognitive temperature τ for V scaling (default 3.0)
        dispersion_threshold: Shield breaking point δ (default 0.1)
        attention_mask: Optional (B, S) padding mask

    Returns:
        Dict with:
            - 'risk_internal': (B, S) confusion risk scores
            - 'risk_external': (B, S) deception risk scores
            - 'prompt_length': int for engine to slice response tokens
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        zeros = torch.zeros(B, S_total, device=device, dtype=dtype)
        return {'risk_internal': zeros, 'risk_external': zeros, 'prompt_length': prompt_length}

    # ================================================================
    # SLICE TO RESPONSE TOKENS (saves compute)
    # ================================================================
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # ================================================================
    # INTERNAL RISK: Confusion Detector
    # Signal: tanh(V/τ) × Shield(D)
    # High when BOTH varentropy and dispersion are high (epistemic collapse)
    # ================================================================
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    v_signal = torch.tanh(varentropy / varentropy_scale)  # [B, S_resp]

    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    d_signal = torch.sigmoid((D_t - dispersion_threshold) * 20.0)  # [B, S_resp]

    risk_internal = v_signal * d_signal  # [B, S_resp]

    # ================================================================
    # EXTERNAL RISK: Deception Detector
    # Signal: JSD(P_attn || P_final)
    # High when FFN overrides context signal (parametric override)
    # ================================================================
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )
    risk_external = jsd  # [B, S_resp]

    # ================================================================
    # PAD TO FULL SEQUENCE (prompt = 0.0 risk = trusted)
    # ================================================================
    padding = torch.zeros(B, prompt_length, device=device, dtype=dtype)
    risk_internal_full = torch.cat([padding, risk_internal], dim=1)
    risk_external_full = torch.cat([padding, risk_external], dim=1)

    # Handle attention mask (masked positions get 0 risk)
    if attention_mask is not None:
        risk_internal_full = risk_internal_full * attention_mask.float()
        risk_external_full = risk_external_full * attention_mask.float()

    return {
        'risk_internal': risk_internal_full,
        'risk_external': risk_external_full,
        'prompt_length': prompt_length
    }


def compute_semantic_authority_v13(
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_scale: float = 3.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    v13.1 Adaptive Regime Switching: Computes both Stability and Grounding signals.

    The engine performs adaptive fusion based on sequence length:
        Score_seq = (1 - w) × S_seq + w × G_seq
        w = sigmoid((Length - τ_len) × α)

    Scientific Justification:
        - Short sequences (QA): Confusion-based failures → Stability detects
        - Long sequences (RAG): Override-based failures → Grounding detects

    Signals:
        - Stability: (1 - V_norm) × (1 - D_t) - HIGH when confident and focused
        - Grounding: 1 - JSD(P_attn || P_final) - HIGH when attention and FFN agree

    Detection Logic:
        | Dataset       | Length | w    | Primary Signal | Result   |
        |---------------|--------|------|----------------|----------|
        | HaluEval Hall | ~10    | ~0.0 | LOW Stability  | DETECTED |
        | HaluEval Fact | ~10    | ~0.0 | HIGH Stability | PASS     |
        | RAGTruth Hall | ~100   | ~1.0 | LOW Grounding  | DETECTED |
        | RAGTruth Fact | ~100   | ~1.0 | HIGH Grounding | PASS     |

    Args:
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (lm_head.weight)
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        jsd_top_k: Top-k for JSD approximation (default 50)
        varentropy_scale: Cognitive temperature τ for V scaling (default 3.0)
        attention_mask: Optional (B, S) padding mask

    Returns:
        Dict with:
            - 'stability': (B, S) epistemic stability scores (HIGH = stable/good)
            - 'grounding': (B, S) mechanistic grounding scores (HIGH = grounded/good)
            - 'prompt_length': int for engine to slice response tokens
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        ones = torch.ones(B, S_total, device=device, dtype=dtype)
        return {'stability': ones, 'grounding': ones, 'prompt_length': prompt_length}

    # ================================================================
    # SLICE TO RESPONSE TOKENS (saves compute)
    # ================================================================
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # ================================================================
    # SIGNAL A: EPISTEMIC STABILITY (For Short/QA)
    # High when model is confident and focused (low V, low D)
    # Stability_t = (1 - V_norm) × (1 - D_t)
    # ================================================================
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    V_norm = torch.tanh(varentropy / varentropy_scale)  # [B, S_resp]

    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    stability_signal = ((1.0 - V_norm) * (1.0 - D_t)).clamp(0.0, 1.0).to(dtype)  # [B, S_resp]

    # ================================================================
    # SIGNAL B: MECHANISTIC GROUNDING (For Long/RAG)
    # High when attention and FFN agree (low JSD)
    # Grounding_t = 1 - JSD(P_attn || P_final)
    # ================================================================
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )
    grounding_signal = (1.0 - jsd).clamp(0.0, 1.0).to(dtype)  # [B, S_resp]

    # ================================================================
    # PAD TO FULL SEQUENCE (prompt = 1.0 = trusted)
    # ================================================================
    padding = torch.ones(B, prompt_length, device=device, dtype=dtype)
    stability_full = torch.cat([padding, stability_signal], dim=1)
    grounding_full = torch.cat([padding, grounding_signal], dim=1)

    # Handle attention mask (masked positions get 1.0 = trusted)
    if attention_mask is not None:
        mask = attention_mask.float()
        stability_full = stability_full * mask + (1.0 - mask)
        grounding_full = grounding_full * mask + (1.0 - mask)

    return {
        'stability': stability_full,
        'grounding': grounding_full,
        'prompt_length': prompt_length
    }


def compute_semantic_authority_v15(
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_scale: float = 3.0,
    dispersion_threshold: float = 0.15,
    dispersion_stiffness: float = 20.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    v15 Coherence-Interaction Model.

    Master Equation:
        Score = (1 - JSD_seq) × (1 - max_t(V_t × Sigmoid(D_t)))

    Key Innovation - Semantic Shielding:
        - V × Sigmoid(D) only penalizes when BOTH V and D are high
        - High V with low D is protected (valid reasoning)
        - Low V with high D is protected (rare words)

    Args:
        prompt_length: Number of prompt tokens
        logits: Model logits [B, S, V]
        embed_matrix: Embedding matrix [V, D]
        h_attn: Hidden states after attention [B, S, D]
        h_block: Hidden states after FFN [B, S, D]
        jsd_top_k: Number of top tokens for JSD calculation
        varentropy_scale: Normalization scale for varentropy
        dispersion_threshold: Gate threshold (tau) for dispersion shield
        dispersion_stiffness: Sigmoid sharpness for dispersion gate
        attention_mask: Optional attention mask [B, S]

    Returns:
        Dict with 'jsd', 'confusion', 'prompt_length' for engine aggregation
    """
    from .entropy import compute_varentropy
    from .semantics import compute_semantic_dispersion
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    if S_resp <= 0:
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        return {'jsd': zeros, 'confusion': zeros, 'prompt_length': prompt_length}

    # Slice to response tokens
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # === SIGNAL 1: MECHANISTIC INTEGRITY (JSD) ===
    # Detects FFN override (deception pattern)
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )  # [B, S_resp] in [0, 1]

    # === SIGNAL 2: EPISTEMIC COHERENCE (V × Sigmoid(D)) ===
    # Detects confusion with semantic shielding

    # Varentropy (model uncertainty)
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    V_norm = torch.tanh(varentropy / varentropy_scale)  # [B, S_resp] in [0, 1]

    # Semantic Dispersion (prediction scatter)
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )  # [B, S_resp] in [0, 1]

    # Dispersion Shield (sigmoid gate)
    # D < threshold → gate ≈ 0 (protected)
    # D > threshold → gate ≈ 1 (exposed)
    D_gate = torch.sigmoid((D_t - dispersion_threshold) * dispersion_stiffness)

    # Confusion signal: only high when BOTH V and D are high
    confusion = V_norm * D_gate  # [B, S_resp] in [0, 1]

    return {
        'jsd': jsd,                   # [B, S_resp]
        'confusion': confusion,       # [B, S_resp]
        'prompt_length': prompt_length
    }


def compute_semantic_authority_v16(
    attention_weights: torch.Tensor,
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_scale: float = 3.0,
    attention_mask: Optional[torch.Tensor] = None,
    query_states: Optional[torch.Tensor] = None,
    key_states: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    v16 Grounded-Risk Architecture.

    Replaces the failed "Semantic Shield" (Dispersion) with "Provenance Shield" (Authority).

    Master Equation:
        Risk_seq = max(R_deception, R_confusion)
        R_deception = Mean(JSD)           # FFN override detection
        R_confusion = Percentile_90(V × (1 - A))  # Ungrounded uncertainty

    Key Innovation - Authority Shield:
        - High V is only penalized if ALSO ungrounded (low A)
        - Valid Summary: High V × Low(1-A) = LOW risk (grounded reasoning)
        - Confused Hall: High V × High(1-A) = HIGH risk (drifting)

    Why this fixes Summarization:
        - v15 used D (dispersion) as shield - fails because valid summaries have high D
        - v16 uses A (authority) as shield - valid summaries attend to source (high A)

    Args:
        attention_weights: (B, H, S, S) attention from semantic layer
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        h_attn: Hidden states after attention [B, S, D]
        h_block: Hidden states after FFN [B, S, D]
        jsd_top_k: Top-k for JSD computation
        varentropy_scale: V normalization for tanh
        attention_mask: Optional (B, S) padding mask
        query_states: Optional (B, H, S, D) for Triton optimization
        key_states: Optional (B, H, S, D) for Triton optimization

    Returns:
        Dict with 'risk_deception', 'risk_confusion', 'prompt_length'
    """
    from .entropy import compute_varentropy
    from ..ops import (
        compute_logit_divergence_jsd,
        compute_authority_flow_recursive,
        _AUTHORITY_TRITON_AVAILABLE,
        compute_flash_authority_v3,
    )

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    if S_resp <= 0:
        zeros = torch.zeros(B, 1, device=device, dtype=dtype)
        return {'risk_deception': zeros, 'risk_confusion': zeros, 'prompt_length': prompt_length}

    # Slice to response tokens
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # === SIGNAL 1: DECEPTION RISK (JSD) ===
    # Detects FFN override - when model's final output diverges from attention signal
    risk_deception = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )  # [B, S_resp] in [0, 1]

    # === SIGNAL 2: CONFUSION RISK (V × (1 - A)) ===
    # Detects ungrounded uncertainty - high varentropy without authority

    # 2a. Varentropy (model uncertainty) - computed on response tokens
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    V_norm = torch.tanh(varentropy / varentropy_scale)  # [B, S_resp] in [0, 1]

    # 2b. Authority Flow (provenance from context)
    # NOTE: gamma must be full sequence [B, S_total] for authority flow functions
    gamma_full = torch.ones(B, S_total, device=device, dtype=dtype)

    use_triton = (
        _AUTHORITY_TRITON_AVAILABLE
        and query_states is not None
        and key_states is not None
        and query_states.is_cuda
        and (query_states.shape[0] == 1 or attention_mask is None)
    )

    if use_triton:
        A_t = compute_flash_authority_v3(
            query_states, key_states, gamma_full, prompt_length
        )
        if attention_mask is not None:
            A_t = A_t * attention_mask.float()
    else:
        A_t = compute_authority_flow_recursive(
            attention_weights, prompt_length, gamma_full, attention_mask
        )

    # Slice authority to response tokens
    A_resp = A_t[:, prompt_length:]  # [B, S_resp]

    # 2c. Ungrounded Risk = Uncertainty × Lack of Provenance
    # High V is only penalized when Authority is LOW
    # - Valid Summary: High V, High A → Low (1-A) → Low Risk
    # - Confused Hall: High V, Low A → High (1-A) → High Risk
    risk_confusion = V_norm * (1.0 - A_resp).clamp(0, 1)  # [B, S_resp] in [0, 1]

    return {
        'risk_deception': risk_deception,    # [B, S_resp]
        'risk_confusion': risk_confusion,    # [B, S_resp]
        'varentropy': varentropy,            # [B, S_resp] - raw varentropy for v17 gating
        'prompt_length': prompt_length
    }


def compute_semantic_authority_v19(
    prompt_length: int,
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    jsd_top_k: int = 50,
    varentropy_threshold: float = 2.5,
    varentropy_scale: float = 3.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    v19 Hinge-Risk Architecture - Unified Zero-Shot Hallucination Detection.

    Master Equation (Token-Level Risk):
        Risk(t) = JSD(t) + ReLU((V(t) - τ) / scale)

    Key Innovation - Non-Monotonic Varentropy Model:
        Previous versions treated V as linear (high V = bad). This fails because:
        - Low V (< 1.5): Confident (good) OR Deceptive (bad) → Need JSD check
        - Med V (1.5 - 3.5): Reasoning (good) → Should be PROTECTED
        - High V (> 4.0): Confused (bad) → Should be PENALIZED

        The ReLU Hinge at τ=2.5 creates this protection:
        - V < τ: Confusion Risk = 0 (Reasoning protected)
        - V > τ: Confusion Risk = tanh((V - τ) / scale) (Confusion penalized)

    Detection Logic:
        | Scenario         | V     | JSD  | Confusion | Deception | Total Risk |
        |------------------|-------|------|-----------|-----------|------------|
        | RAGTruth Hall    | ~1.2  | ~0.5 | 0 (hinge) | HIGH      | HIGH       |
        | HaluEval Hall    | ~5.0  | ~0.1 | HIGH      | Low       | HIGH       |
        | RAGTruth Fact    | ~2.0  | ~0.1 | 0 (hinge) | Low       | LOW        |
        | HaluEval Fact    | ~0.5  | ~0.1 | 0 (hinge) | Low       | LOW        |

    Aggregation: Max (Safety Veto) - if ANY token has high risk, flag sequence.

    Args:
        prompt_length: Token index where response begins
        logits: (B, S, V) model output logits
        embed_matrix: (V, D) output embedding matrix (lm_head.weight)
        h_attn: (B, S, D) hidden state after attention (before MLP)
        h_block: (B, S, D) hidden state after full block (after MLP)
        jsd_top_k: Top-k for JSD approximation (default 50)
        varentropy_threshold: Reasoning ceiling τ (default 2.5)
            - Below this: V is protected (valid reasoning)
            - Above this: V contributes to confusion risk
        varentropy_scale: Normalization scale for excess V (default 3.0)
        attention_mask: Optional (B, S) padding mask

    Returns:
        risk: (B, S) Token-level risk scores in [0, ~2].
              Higher = more likely hallucination.
              Prompt positions are set to 0.0 (no risk).
    """
    from .entropy import compute_varentropy
    from ..ops import compute_logit_divergence_jsd

    B, S_total, V = logits.shape
    device = logits.device
    dtype = logits.dtype
    S_resp = S_total - prompt_length

    # Edge case: no response tokens
    if S_resp <= 0:
        return torch.zeros(B, S_total, device=device, dtype=dtype)

    # ================================================================
    # SLICE TO RESPONSE TOKENS (saves compute)
    # ================================================================
    h_attn_resp = h_attn[:, prompt_length:, :]
    h_block_resp = h_block[:, prompt_length:, :]
    logits_resp = logits[:, prompt_length:, :].contiguous()

    # ================================================================
    # SIGNAL 1: DECEPTION RISK (JSD)
    # Catches "Confident Lies" - Low V but FFN overriding context
    # Range: [0, 1]
    # ================================================================
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=jsd_top_k
    )

    # ================================================================
    # SIGNAL 2: CONFUSION RISK (Hinge Varentropy)
    # Catches "Epistemic Collapse" - High V indicating model confusion
    # The ReLU hinge protects valid reasoning (V < threshold)
    # Range: [0, 1] after tanh
    # ================================================================
    varentropy = compute_varentropy(logits_resp, attention_mask=None)

    # ReLU Hinge: Only penalize if V > threshold
    # V < 2.5: excess = 0 → risk = 0 (protected)
    # V = 5.0: excess = 2.5 → risk = tanh(2.5/3) ≈ 0.79
    # V = 8.0: excess = 5.5 → risk = tanh(5.5/3) ≈ 0.96
    v_excess = torch.nn.functional.relu(varentropy - varentropy_threshold)
    confusion_risk = torch.tanh(v_excess / varentropy_scale)

    # ================================================================
    # TOTAL RISK: Additive Combination
    # Both signals contribute independently - no interference
    # Range: [0, ~2] (JSD up to 1 + Confusion up to 1)
    # ================================================================
    token_risk = jsd + confusion_risk

    # ================================================================
    # PAD TO FULL SEQUENCE (prompt = 0.0 risk = trusted)
    # ================================================================
    padding = torch.zeros(B, prompt_length, device=device, dtype=dtype)
    full_risk = torch.cat([padding, token_risk], dim=1)

    # Handle attention mask (masked positions get 0 risk)
    if attention_mask is not None:
        full_risk = full_risk * attention_mask.float()

    return full_risk
