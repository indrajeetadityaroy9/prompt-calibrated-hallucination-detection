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
) -> torch.Tensor:
    """
    Compute Authority Score with optional spectral roughness penalty.

    Unified interface for v3.1 Authority Flow:
        A(t) = [sum_Prompt A_{t,j}] + [sum_Gen A_{t,j} * A(j)] * M(t)
        A_final(t) = A(t) / (1 + lambda * roughness(t))

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        register_mask: (B, S) register mask M(t) in [0, 1]
        roughness: (B, S) spectral roughness or MLP divergence
        lambda_roughness: Penalty weight for roughness
        previous_authority: (B, S) authority from previous step (streaming)
        attention_mask: (B, S) padding mask
        use_vectorized: Use vectorized approximation (faster, less accurate)

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    if use_vectorized:
        authority = compute_authority_flow_vectorized(
            attention_weights, prompt_length, register_mask, attention_mask
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
