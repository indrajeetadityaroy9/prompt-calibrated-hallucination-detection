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
    gate = fused_stability_gate(h_attn, h_block, sensitivity=10.0)

    # 3. Compute trust from semantic dispersion + varentropy
    dispersion = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    varentropy = compute_varentropy(logits)

    # Normalize varentropy by entropy
    probs = F.softmax(logits, dim=-1)
    log_probs = probs.clamp(min=1e-10).log()
    entropy = -(probs * log_probs).sum(dim=-1)
    v_norm = (varentropy / (entropy + 1e-8)).clamp(0.0, 1.0)

    # Trust = 1 - D × (1 + λ × V)
    weighted_dispersion = dispersion * (1.0 + varentropy_lambda * v_norm)
    trust = (1.0 - weighted_dispersion).clamp(0.0, 1.0)

    # 4. CPG Detection - identify coherent parametric generation
    # CPG = low gate + low dispersion + high confidence + stable varentropy
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
