"""
Dual-Subspace Projection Score (DPS) for DSG hallucination detection.

Projects output hidden states onto two orthogonal reference subspaces:
- Context subspace V_ctx: SVD of context hidden states (input-dependent)
- Reasoning subspace V_rsn: Bottom singular vectors of lm_head (model-dependent)

DPS(t) = s_rsn / (s_ctx + s_rsn + eps), range [0,1], higher = riskier.

References:
    - "Mind the Gap: Spectral Analysis of Rank Collapse" (ICLR 2025)
    - "Retrieval Head Mechanistically Explains Long-Context Factuality" (2024)
    - HARP: reasoning subspace via SVD of unembedding (arXiv:2509.11536)
"""

import math

import torch
import numpy as np
from typing import Dict
from torch import Tensor

from ..numerics import EPS


def _find_spectral_gap_rank(
    S: Tensor,
    remove_top: int = 1,
    n_samples: int = None,
    n_features: int = None,
) -> int:
    """
    Rank via Marchenko-Pastur boundary (primary) or ratio gap (fallback).

    Primary (when n_samples, n_features provided):
        Estimate noise variance from bottom-half singular values, compute
        MP upper edge λ+ = σ²(1 + √(d/n))², count SVs exceeding it.

    Fallback:
        k = argmax_i (S[i] / S[i+1]) + 1
        Threshold: e ≈ 2.718 (1 nat in log-space — information-theoretic
        justification). Falls back to len(S) if spectrum is flat.

    Args:
        S: Singular values (descending order).
        remove_top: Skip top-N singular values before gap detection.
        n_samples: Number of data points (rows of data matrix).
        n_features: Dimensionality (columns of data matrix).

    Reference: Marchenko & Pastur (1967); Bai & Silverstein (2010).
    """
    if len(S) <= 1:
        return len(S)
    S_trimmed = S[remove_top:]
    if len(S_trimmed) <= 1:
        return len(S)

    # Primary: Marchenko-Pastur boundary
    if n_samples is not None and n_features is not None and n_samples > 1:
        gamma = n_features / n_samples
        # Estimate noise variance from bottom half of spectrum
        noise_svs = S_trimmed[len(S_trimmed) // 2:]
        if len(noise_svs) > 0:
            sigma2 = (noise_svs ** 2).mean().item() / n_features
            if sigma2 > 0:
                mp_upper = sigma2 * (1 + math.sqrt(gamma)) ** 2
                k = int((S_trimmed ** 2 / n_features > mp_upper).sum().item())
                if k >= 1:
                    return k + remove_top

    # Fallback: ratio gap with e threshold (1 nat in log-space)
    ratios = S_trimmed[:-1] / (S_trimmed[1:] + EPS)
    if ratios.max().item() < math.e:
        return len(S)
    return max(1, int(ratios.argmax().item()) + 1 + remove_top)


class DualSubspaceGrounding:
    """
    Dual-Subspace Projection Score (DPS) for DSG.

    Measures where output representations sit relative to two subspaces:
    - Context subspace V_ctx (SVD of context hidden states, spectral gap rank)
    - Reasoning subspace V_rsn (bottom spectral gap of lm_head.weight SVD)

    DPS(t) = s_rsn / (s_ctx + s_rsn + eps)
    Range [0,1], higher = more reasoning-driven = riskier.
    """

    def __init__(
        self,
        lm_head_weight: Tensor,
    ):
        # One-time reasoning subspace SVD (cached per model load)
        self._reasoning_basis = self._compute_reasoning_basis(lm_head_weight)

        # Per-input context subspace (set via set_context_basis)
        # These are left uninitialized; callers must call set_context_basis before use.
        self._context_basis: Tensor
        self._context_center: Tensor
        self._context_rank: int = 0

    def _compute_reasoning_basis(self, lm_head_weight: Tensor) -> Tensor:
        """
        SVD of lm_head.weight, keep bottom singular vectors via spectral gap.

        Scans reversed singular values for the largest relative gap from the bottom.
        Cap at len(S) // 2 to avoid taking too many.
        """
        with torch.no_grad():
            w = lm_head_weight.float()
            _, S, Vh = torch.linalg.svd(w, full_matrices=False)

            # Bottom spectral gap: scan reversed singular values
            # No outlier removal for bottom SVs of lm_head (no mean-direction artifact)
            S_rev = S.flip(0)
            k_r = _find_spectral_gap_rank(S_rev, remove_top=0)
            k_r = min(k_r, len(S) // 2)
            k_r = max(1, k_r)

            # Bottom k_r singular vectors (least contribution to next-token prediction)
            V_rsn = Vh[-k_r:]  # [k_r, hidden_dim]

        return V_rsn.to(lm_head_weight.device)

    def set_context_basis(self, context_hidden: Tensor) -> None:
        """Compute context subspace from prefill hidden states. Call once per input."""
        context_hidden = context_hidden.float()
        context_mean = context_hidden.mean(dim=0, keepdim=True)
        context_centered = context_hidden - context_mean

        U, S, Vh = torch.linalg.svd(context_centered, full_matrices=False)

        # Spectral gap rank with Marchenko-Pastur and rank floor
        n_context = context_hidden.shape[0]
        hidden_dim = context_hidden.shape[1]
        k = _find_spectral_gap_rank(
            S, remove_top=1,
            n_samples=n_context,
            n_features=hidden_dim,
        )
        k = max(k, max(3, int(math.ceil(math.sqrt(n_context)))))
        k = min(k, len(S), n_context)

        self._context_basis = Vh[:k]  # [k, hidden_dim]
        self._context_center = context_mean
        self._context_rank = k

    def compute_dps(
        self,
        layer_hidden_states: Dict[int, Tensor],
        n_layers: int,
        active_layers: list = None,
    ) -> float:
        """
        Compute DPS for a single token.

        Args:
            layer_hidden_states: Hidden states per layer.
            n_layers: Total number of layers.
            active_layers: If provided, use these layers (from JSD-variance
                Otsu selection). Falls back to middle-third if None.
        """
        if active_layers is not None:
            target_layers = [i for i in layer_hidden_states if i in active_layers]
        else:
            # Fallback: middle third
            start = n_layers // 3
            end = 2 * n_layers // 3
            target_layers = [i for i in layer_hidden_states if start <= i < end]

        if not target_layers:
            return 0.5

        V_ctx = self._context_basis.to(dtype=torch.float32)
        V_rsn = self._reasoning_basis.to(dtype=torch.float32)
        mu = self._context_center.squeeze(0).to(dtype=torch.float32)

        dps_values = []
        for layer_idx in target_layers:
            h = layer_hidden_states[layer_idx].float().squeeze()
            h_centered = h - mu

            h_norm = torch.norm(h_centered) + EPS

            # Context projection
            proj_ctx = V_ctx.T @ (V_ctx @ h_centered)
            s_ctx = torch.norm(proj_ctx) / h_norm

            # Reasoning projection
            proj_rsn = V_rsn.T @ (V_rsn @ h_centered)
            s_rsn = torch.norm(proj_rsn) / h_norm

            dps_layer = s_rsn / (s_ctx + s_rsn + EPS)
            dps_values.append(dps_layer.item())

        return float(np.mean(dps_values))

    @property
    def context_basis(self) -> Tensor:
        """V_ctx for use by POS signal."""
        return self._context_basis

    @property
    def context_center(self) -> Tensor:
        """mu_context for use by POS signal."""
        return self._context_center
