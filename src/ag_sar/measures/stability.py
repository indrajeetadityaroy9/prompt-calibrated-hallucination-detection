"""
Adaptive Gate for Model-Agnostic Stability Gating.

Normalizes MLP-Attention divergence using online Z-score,
making thresholds portable across model architectures.
"""

import torch
import torch.nn.functional as F
from typing import Optional


class AdaptiveGateBatch:
    """
    Batch-aware adaptive gate with online Z-score normalization.

    Maintains running statistics (μ, σ) of divergence signal,
    normalizes to Z-score, applies sigmoid for [0, 1] gate.
    """

    def __init__(self, momentum: float = 0.01, warmup_samples: int = 100):
        self.momentum = momentum
        self.warmup_samples = warmup_samples
        self.global_mu = 0.0
        self.global_sq_mu = 1.0
        self.n_samples = 0
        self.initialized = False

    @property
    def sigma(self) -> float:
        variance = max(self.global_sq_mu - self.global_mu ** 2, 1e-8)
        return variance ** 0.5

    def get_gate(self, raw_divergence: torch.Tensor) -> torch.Tensor:
        """Compute normalized gate from raw divergence (B, S)."""
        batch_mean = raw_divergence.mean().item()
        batch_sq_mean = (raw_divergence ** 2).mean().item()

        if not self.initialized:
            self.global_mu = batch_mean
            self.global_sq_mu = batch_sq_mean
            self.initialized = True
        else:
            self.global_mu = (1 - self.momentum) * self.global_mu + self.momentum * batch_mean
            self.global_sq_mu = (1 - self.momentum) * self.global_sq_mu + self.momentum * batch_sq_mean

        self.n_samples += raw_divergence.numel()

        if self.n_samples < self.warmup_samples:
            return torch.sigmoid(raw_divergence)

        z_score = (raw_divergence - self.global_mu) / (self.sigma + 1e-6)
        return torch.sigmoid(z_score)

    def seed_from_statistics(self, mu: float, sigma: float, n_samples: int = 100) -> None:
        """Seed gate with pre-computed statistics (bypasses warmup)."""
        self.global_mu = mu
        self.global_sq_mu = max(sigma, 1e-6) ** 2 + mu ** 2
        self.n_samples = n_samples
        self.initialized = True

    def reset(self) -> None:
        """Reset running statistics."""
        self.global_mu = 0.0
        self.global_sq_mu = 1.0
        self.n_samples = 0
        self.initialized = False
