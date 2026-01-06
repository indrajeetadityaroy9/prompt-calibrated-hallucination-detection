"""
Adaptive Gate for Model-Agnostic Stability Gating (AG-SAR v8.0).

This module provides adaptive normalization for MLP-Attention divergence
signals, making the gate portable across different model architectures.

Problem: Raw MLP-Attention Divergence varies wildly between models:
- Llama-3 might range 0.1-0.5
- Mistral might range 5.0-10.0
Hard-coded thresholds (0.2, 0.8) break cross-model portability.

Solution: Normalize the divergence signal relative to its own recent history.

Formula:
    z_score = (raw_divergence - μ) / σ
    gate = sigmoid(z_score)

Result:
- Z=0 (Mean divergence) → Gate=0.5 (Neutral)
- Z=+2 (High divergence) → Gate=0.88 (Trust context)
- Z=-2 (Low divergence) → Gate=0.12 (Trust parametric)
"""

import torch
import torch.nn.functional as F
from typing import Optional


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
