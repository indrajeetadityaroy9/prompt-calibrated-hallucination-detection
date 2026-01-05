"""
Layer Stability Measures for Hallucination Detection (AG-SAR v11.0+).

This module contains two types of stability measures:

1. Layer Drift (v11.0) - EXPERIMENTAL, NEGATIVE RESULT ON RAGTruth
   Measures semantic consistency between intermediate and final layers.
   Preserved for research but NOT recommended for production.

2. Adaptive Gate (v12.2) - Model-Agnostic Online Normalization
   Normalizes the MLP-Attention divergence signal using running statistics,
   making the gate portable across different model architectures.

================================================================================
LAYER DRIFT (v11.0) - NEGATIVE RESULT

Original Hypothesis: "The model considered it, then rejected it"
- Hallucinations show semantic suppression: correct answer in middle layers
  but overridden by parametric memory in final layers.

EMPIRICAL FINDING (Feature Inversion):
Layer Drift showed AUROC=0.23 (worse than random) on RAGTruth. High Drift
correlates with FAITHFULNESS, not hallucination.

The "Effort Hypothesis" explains this:
- Faithful RAG answers require HIGH cognitive effort to suppress pre-trained
  priors and attend to novel context → large vector rotation (high drift)
- Hallucinations follow the path of least resistance → smooth flow (low drift)

Conclusion: Layer Drift measures "Thinking Effort", not "Deception".

================================================================================
ADAPTIVE GATE (v12.2) - Model-Agnostic

Problem: Raw MLP-Attention Divergence varies wildly between models:
- Llama-3 might range 0.1–0.5
- Mistral might range 5.0–10.0
Hard-coded thresholds (0.2, 0.8) break cross-model portability.

Solution: Normalize the divergence signal relative to its own recent history.

Formula:
    z_score = (raw_divergence - μ) / σ
    gate = sigmoid(z_score)

Result:
- Z=0 (Mean divergence) → Gate=0.5 (Neutral)
- Z=+2 (High divergence) → Gate=0.88 (Use JEPA/context)
- Z=-2 (Low divergence) → Gate=0.12 (Use Truth Vector)

================================================================================
References:
- "DoLa: Decoding by Contrasting Layers" (Chuang et al., 2023)
- "Contrastive Decoding" (Li et al., 2023)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_layer_drift(
    hidden_states_mid: torch.Tensor,
    hidden_states_final: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float = 1.0,
    use_kl: bool = False,
) -> torch.Tensor:
    """
    Compute Layer Drift (Mind-Change Detection).

    Measures semantic distance between probability distributions at
    mid-layer vs final-layer. High drift indicates the model "changed
    its mind" between intermediate processing and final output.

    Algorithm:
    1. Project both hidden states to vocabulary space using lm_head
    2. Convert to probability distributions via softmax
    3. Measure cosine distance (or KL divergence) between distributions

    Interpretation:
    - Low drift (< 0.3): Stable prediction, model didn't reconsider
    - High drift (> 0.7): Model changed prediction, potential suppression

    Args:
        hidden_states_mid: (B, S, D) hidden states from mid-layer (~layer N/2)
        hidden_states_final: (B, S, D) hidden states from final layer
        lm_head_weight: (V, D) language model head weight matrix
        temperature: Softmax temperature for probability computation
        use_kl: Use KL divergence instead of cosine distance

    Returns:
        drift: (B, S) layer drift scores in [0, 1]
               0.0 = No change (stable prediction)
               1.0 = Complete change (different distributions)

    Example:
        >>> # Model predicts "Paris" in mid-layer but "London" in final layer
        >>> # This indicates potential suppression/override
        >>> drift = compute_layer_drift(h_mid, h_final, lm_head.weight)
        >>> # drift will be high (~0.7+) for positions where prediction changed
    """
    # Project hidden states to logits
    # lm_head_weight: (V, D), hidden: (B, S, D) -> logits: (B, S, V)
    logits_mid = torch.matmul(hidden_states_mid, lm_head_weight.T)
    logits_final = torch.matmul(hidden_states_final, lm_head_weight.T)

    # Convert to probability distributions
    probs_mid = F.softmax(logits_mid / temperature, dim=-1)
    probs_final = F.softmax(logits_final / temperature, dim=-1)

    if use_kl:
        # KL Divergence: D_KL(P_mid || P_final)
        # Measures how much information is lost when using P_final to approximate P_mid
        kl_div = F.kl_div(
            F.log_softmax(logits_final / temperature, dim=-1),
            probs_mid,
            reduction='none'
        ).sum(dim=-1)  # (B, S)

        # Normalize KL divergence to [0, 1] using sigmoid
        # KL can be unbounded, so we use tanh to squash
        drift = torch.tanh(kl_div / 2.0)
    else:
        # Cosine Distance: 1 - cosine_similarity
        # More stable and bounded in [0, 2], typically [0, 1]
        drift = 1.0 - F.cosine_similarity(probs_mid, probs_final, dim=-1)

    # Clamp to [0, 1] for numerical stability
    drift = drift.clamp(0.0, 1.0)

    return drift


def compute_top_k_drift(
    hidden_states_mid: torch.Tensor,
    hidden_states_final: torch.Tensor,
    lm_head_weight: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-K Layer Drift (Focused Mind-Change Detection).

    Instead of comparing full distributions, focuses on drift in the
    top-k tokens. More efficient and often more meaningful than full
    distribution comparison.

    Algorithm:
    1. Project both hidden states to logits
    2. Get top-k tokens from final layer (what model actually said)
    3. Compare probabilities of these tokens in mid vs final layer

    Args:
        hidden_states_mid: (B, S, D) hidden states from mid-layer
        hidden_states_final: (B, S, D) hidden states from final layer
        lm_head_weight: (V, D) language model head weight matrix
        k: Number of top tokens to compare
        temperature: Softmax temperature

    Returns:
        drift: (B, S) top-k focused drift scores in [0, 1]
    """
    # Project to logits
    logits_mid = torch.matmul(hidden_states_mid, lm_head_weight.T)
    logits_final = torch.matmul(hidden_states_final, lm_head_weight.T)

    # Get probabilities
    probs_mid = F.softmax(logits_mid / temperature, dim=-1)
    probs_final = F.softmax(logits_final / temperature, dim=-1)

    # Get top-k indices from final layer
    _, top_indices = torch.topk(probs_final, k=k, dim=-1)  # (B, S, K)

    # Gather top-k probabilities from both layers
    probs_mid_topk = torch.gather(probs_mid, -1, top_indices)  # (B, S, K)
    probs_final_topk = torch.gather(probs_final, -1, top_indices)  # (B, S, K)

    # Cosine distance in the top-k probability subspace
    drift = 1.0 - F.cosine_similarity(probs_mid_topk, probs_final_topk, dim=-1)

    return drift.clamp(0.0, 1.0)


def compute_rank_drift(
    hidden_states_mid: torch.Tensor,
    hidden_states_final: torch.Tensor,
    lm_head_weight: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Rank Drift (Token Ranking Change Detection).

    Measures how much the ranking of top tokens changed between layers.
    Focuses on whether the "winner" changed, not just probability values.

    Algorithm:
    1. Get top-1 token from each layer
    2. If different, compute inverse rank of mid-layer's top-1 in final layer
    3. High rank difference = model strongly changed its mind

    Args:
        hidden_states_mid: (B, S, D) hidden states from mid-layer
        hidden_states_final: (B, S, D) hidden states from final layer
        lm_head_weight: (V, D) language model head weight matrix
        k: Number of top ranks to consider

    Returns:
        drift: (B, S) rank drift scores in [0, 1]
               0.0 = Same top-1 token (no change)
               1.0 = Mid-layer top-1 not in final top-k (complete change)
    """
    # Project to logits
    logits_mid = torch.matmul(hidden_states_mid, lm_head_weight.T)
    logits_final = torch.matmul(hidden_states_final, lm_head_weight.T)

    # Get top-1 from mid layer
    top1_mid = logits_mid.argmax(dim=-1)  # (B, S)

    # Get top-k indices from final layer
    _, topk_final = torch.topk(logits_final, k=k, dim=-1)  # (B, S, K)

    # Check if mid-layer top-1 is in final top-k
    # Create mask where mid-layer top-1 matches any of final top-k
    top1_mid_expanded = top1_mid.unsqueeze(-1)  # (B, S, 1)
    matches = (topk_final == top1_mid_expanded)  # (B, S, K)

    # Find the rank (position) of mid-layer top-1 in final top-k
    # If not found, rank = k (maximum drift)
    match_positions = matches.float() * torch.arange(
        k, device=matches.device
    ).float().unsqueeze(0).unsqueeze(0)

    # Get minimum position (rank) where match occurs, or k if no match
    has_match = matches.any(dim=-1)  # (B, S)
    rank = torch.where(
        has_match,
        match_positions.max(dim=-1).values,  # Position of match
        torch.full_like(top1_mid.float(), k)  # Not found = max rank
    )

    # Normalize rank to [0, 1]
    # rank 0 = same top-1 = drift 0
    # rank k = not in top-k = drift 1
    drift = rank / k

    return drift.clamp(0.0, 1.0)


def apply_drift_penalty(
    trust: torch.Tensor,
    drift: torch.Tensor,
    sensitivity: float = 1.0,
) -> torch.Tensor:
    """
    Apply Layer Drift penalty to Trust scores.

    Master Equation v12:
        final_trust = trust * (1 - drift * sensitivity)

    When drift is high (model changed its mind), trust is penalized.
    This catches cases where:
    - Model confidently outputs X but internally considered Y
    - Parametric memory overrode context-based reasoning

    Args:
        trust: (B, S) base trust/authority scores
        drift: (B, S) layer drift scores
        sensitivity: Scale factor for drift penalty (1.0 = linear)

    Returns:
        penalized_trust: (B, S) trust after drift penalty
    """
    penalty = 1.0 - drift * sensitivity
    penalty = penalty.clamp(0.0, 1.0)

    return trust * penalty


# =============================================================================
# ADAPTIVE GATE (v12.2) - Model-Agnostic Online Normalization
# =============================================================================


class AdaptiveGate:
    """
    Online Z-Score Normalization for Model-Agnostic Stability Gating.

    Maintains running Mean (μ) and Std Dev (σ) of raw divergence signal,
    then normalizes to Z-Score and applies sigmoid for [0,1] gate value.

    This makes the gate signal portable across different model architectures
    without manual threshold tuning.

    Args:
        momentum: EMA momentum for statistics update (default 0.01)
                  Lower = more stable, higher = more responsive
        warmup_samples: Number of samples before using adaptive normalization

    Example:
        >>> gate = AdaptiveGate(momentum=0.01)
        >>> for batch in data:
        ...     raw_div = compute_divergence(h_attn, h_block)
        ...     normalized = gate.get_gate(raw_div)
    """

    def __init__(self, momentum: float = 0.01, warmup_samples: int = 10):
        self.momentum = momentum
        self.warmup_samples = warmup_samples

        # Running statistics (EMA)
        self.mu = 0.0
        self.sq_mu = 1.0  # E[X²] for variance computation
        self.n_samples = 0
        self.initialized = False

    def update(self, val: float) -> None:
        """Update running statistics with new observation."""
        if not self.initialized:
            self.mu = val
            self.sq_mu = val ** 2
            self.initialized = True
        else:
            self.mu = (1 - self.momentum) * self.mu + self.momentum * val
            self.sq_mu = (1 - self.momentum) * self.sq_mu + self.momentum * (val ** 2)

        self.n_samples += 1

    @property
    def sigma(self) -> float:
        """Compute standard deviation from running moments."""
        variance = max(self.sq_mu - self.mu ** 2, 1e-8)
        return variance ** 0.5

    def get_gate(self, raw_divergence: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized gate value from raw divergence.

        Args:
            raw_divergence: Raw MLP-Attention divergence (scalar or tensor)

        Returns:
            Normalized gate in [0, 1] via sigmoid(z-score)
        """
        # Handle tensor input
        if isinstance(raw_divergence, torch.Tensor):
            val = raw_divergence.mean().item()
        else:
            val = float(raw_divergence)

        # Update running statistics
        self.update(val)

        # During warmup, use simple sigmoid on raw value
        if self.n_samples < self.warmup_samples:
            return torch.sigmoid(raw_divergence)

        # Compute Z-Score normalization
        if isinstance(raw_divergence, torch.Tensor):
            z_tensor = (raw_divergence - self.mu) / (self.sigma + 1e-6)
            return torch.sigmoid(z_tensor)
        else:
            z_score = (val - self.mu) / (self.sigma + 1e-6)
            return torch.sigmoid(torch.tensor(z_score))

    def reset(self) -> None:
        """Reset running statistics."""
        self.mu = 0.0
        self.sq_mu = 1.0
        self.n_samples = 0
        self.initialized = False

    def get_stats(self) -> dict:
        """Get current running statistics for debugging."""
        return {
            "mu": self.mu,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
            "initialized": self.initialized,
        }


class AdaptiveGateBatch:
    """
    Batch-aware Adaptive Gate for per-token normalization.

    Extends AdaptiveGate to handle full sequence tensors (B, S)
    while maintaining global statistics across all positions.

    Args:
        momentum: EMA momentum for statistics update
        warmup_samples: Samples before using adaptive normalization
    """

    def __init__(self, momentum: float = 0.01, warmup_samples: int = 100):
        self.momentum = momentum
        self.warmup_samples = warmup_samples

        # Global statistics (across all positions)
        self.global_mu = 0.0
        self.global_sq_mu = 1.0
        self.n_samples = 0
        self.initialized = False

    def update_global(self, vals: torch.Tensor) -> None:
        """Update global statistics with batch of observations."""
        batch_mean = vals.mean().item()
        batch_sq_mean = (vals ** 2).mean().item()

        if not self.initialized:
            self.global_mu = batch_mean
            self.global_sq_mu = batch_sq_mean
            self.initialized = True
        else:
            self.global_mu = (1 - self.momentum) * self.global_mu + self.momentum * batch_mean
            self.global_sq_mu = (1 - self.momentum) * self.global_sq_mu + self.momentum * batch_sq_mean

        self.n_samples += vals.numel()

    @property
    def sigma(self) -> float:
        """Compute standard deviation from running moments."""
        variance = max(self.global_sq_mu - self.global_mu ** 2, 1e-8)
        return variance ** 0.5

    def get_gate(self, raw_divergence: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized gate values from raw divergence tensor.

        Args:
            raw_divergence: (B, S) raw MLP-Attention divergence

        Returns:
            (B, S) normalized gate values in [0, 1]
        """
        # Update running statistics
        self.update_global(raw_divergence)

        # During warmup, use simple sigmoid
        if self.n_samples < self.warmup_samples:
            return torch.sigmoid(raw_divergence)

        # Compute Z-Score normalization
        z_score = (raw_divergence - self.global_mu) / (self.sigma + 1e-6)

        return torch.sigmoid(z_score)

    def reset(self) -> None:
        """Reset running statistics."""
        self.global_mu = 0.0
        self.global_sq_mu = 1.0
        self.n_samples = 0
        self.initialized = False

    def get_stats(self) -> dict:
        """Get current running statistics for debugging."""
        return {
            "mu": self.global_mu,
            "sigma": self.sigma,
            "n_samples": self.n_samples,
            "initialized": self.initialized,
        }


def compute_adaptive_stability_gate(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    adaptive_gate: Optional[AdaptiveGateBatch] = None,
    sensitivity: float = 1.0,
) -> torch.Tensor:
    """
    Compute stability gate with optional adaptive normalization.

    This is a drop-in replacement for compute_stability_gate that supports
    model-agnostic adaptive normalization via online Z-Score.

    Args:
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP
        adaptive_gate: Optional AdaptiveGateBatch for online normalization
        sensitivity: Scale factor for divergence (only used if no adaptive_gate)

    Returns:
        (B, S) gate values in [0, 1]

    Example:
        >>> adaptive = AdaptiveGateBatch(momentum=0.01)
        >>> gate = compute_adaptive_stability_gate(h_attn, h_block, adaptive)
    """
    # Compute raw divergence: 1 - cosine_similarity
    # High divergence = MLP changed the representation significantly
    cos_sim = F.cosine_similarity(h_attn, h_block, dim=-1)
    raw_divergence = 1.0 - cos_sim

    if adaptive_gate is not None:
        # Use adaptive normalization (model-agnostic)
        return adaptive_gate.get_gate(raw_divergence)
    else:
        # Fallback to original exponential gate
        gate = torch.exp(-sensitivity * raw_divergence)
        return gate
