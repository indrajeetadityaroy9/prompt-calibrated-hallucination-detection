"""
Dual-Subspace Projection Score (DPS) and Context-Grounding Direction (CGD)
for DSG hallucination detection.

DPS projects output hidden states onto two orthogonal reference subspaces:
- Context subspace V_ctx: SVD of context hidden states (input-dependent)
- Reasoning subspace V_rsn: Bottom singular vectors of lm_head (model-dependent)

DPS(t) = s_rsn / (s_ctx + s_rsn + eps), range [0,1], higher = riskier.
Magnitude-gated: DPS blends toward 0.5 when ||h_centered|| is small.

CGD measures whether generation moves toward or away from context representation:
d_ctx = context_center - prompt_center
CGD(t) = (1 - cos(h_gen - prompt_center, d_ctx)) / 2, range [0,1], higher = riskier.

References:
    - "Mind the Gap: Spectral Analysis of Rank Collapse" (ICLR 2025)
    - HARP: reasoning subspace via SVD of unembedding (arXiv:2509.11536)
    - ContextFocus: Activation Steering Directions (arXiv:2601.04131)
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
        MP upper edge lambda+ = sigma^2(1 + sqrt(d/n))^2, count SVs exceeding it.

    Fallback:
        k = argmax_i (S[i] / S[i+1]) + 1
        Threshold: e ~ 2.718 (1 nat in log-space).

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
        noise_svs = S_trimmed[len(S_trimmed) // 2:]
        if len(noise_svs) > 0:
            sigma2 = (noise_svs ** 2).mean().item() / n_features
            if sigma2 > 0:
                mp_upper = sigma2 * (1 + math.sqrt(gamma)) ** 2
                k = int((S_trimmed ** 2 / n_features > mp_upper).sum().item())
                if k >= 1:
                    return k + remove_top

    # Fallback: ratio gap with e threshold
    ratios = S_trimmed[:-1] / (S_trimmed[1:] + EPS)
    if ratios.max().item() < math.e:
        return len(S)
    return max(1, int(ratios.argmax().item()) + 1 + remove_top)


class DualSubspaceGrounding:
    """
    DPS + CGD signals for DSG.

    DPS: Measures where output representations sit relative to two subspaces.
    CGD: Measures cosine alignment of generation with context direction.
    """

    def __init__(
        self,
        lm_head_weight: Tensor,
    ):
        # One-time reasoning subspace SVD (cached per model load)
        self._reasoning_basis = self._compute_reasoning_basis(lm_head_weight)

        # Per-input state (set via set_context_basis / set_prompt_center)
        self._context_basis: Tensor = None  # type: ignore[assignment]
        self._context_center: Tensor = None  # type: ignore[assignment]
        self._context_rank: int = 0
        self._prompt_center: Tensor = None  # type: ignore[assignment]
        self._tau: float = None  # type: ignore[assignment]

    def _compute_reasoning_basis(self, lm_head_weight: Tensor) -> Tensor:
        """SVD of lm_head.weight, keep bottom singular vectors via spectral gap."""
        with torch.no_grad():
            w = lm_head_weight.float()
            _, S, Vh = torch.linalg.svd(w, full_matrices=False)

            S_rev = S.flip(0)
            k_r = _find_spectral_gap_rank(S_rev, remove_top=0)
            k_r = min(k_r, len(S) // 2)
            k_r = max(1, k_r)

            V_rsn = Vh[-k_r:]  # [k_r, hidden_dim]

        return V_rsn.to(lm_head_weight.device)

    def set_context_basis(self, context_hidden: Tensor) -> None:
        """Compute context subspace from prefill hidden states. Call once per input."""
        context_hidden = context_hidden.float()
        context_mean = context_hidden.mean(dim=0, keepdim=True)
        context_centered = context_hidden - context_mean

        U, S, Vh = torch.linalg.svd(context_centered, full_matrices=False)

        n_context = context_hidden.shape[0]
        hidden_dim = context_hidden.shape[1]
        k = _find_spectral_gap_rank(
            S, remove_top=1,
            n_samples=n_context,
            n_features=hidden_dim,
        )
        k = max(k, int(math.ceil(math.sqrt(n_context))))
        k = min(k, len(S), n_context)

        self._context_basis = Vh[:k]  # [k, hidden_dim]
        self._context_center = context_mean
        self._context_rank = k

    def set_prompt_center(self, prompt_hidden: Tensor) -> None:
        """
        Set prompt center from non-context prompt tokens.

        Used by CGD signal to compute context-grounding direction.
        """
        self._prompt_center = prompt_hidden.float().mean(dim=0)

    def set_magnitude_tau(self, prefill_hidden: Tensor, context_center: Tensor) -> None:
        """
        Derive tau from median ||h_centered|| of prefill hidden states.

        Used by DPS magnitude gate to suppress unreliable DPS ratios
        when the hidden state is close to the context centroid.
        """
        h_centered = prefill_hidden.float() - context_center.squeeze(0).float()
        norms = torch.norm(h_centered, dim=-1)
        self._tau = float(norms.median().item())

    def compute_dps(
        self,
        layer_hidden_states: Dict[int, Tensor],
        n_layers: int,
    ) -> float:
        """
        Compute DPS for a single token, with magnitude gating.

        Args:
            layer_hidden_states: Hidden states per layer.
            n_layers: Total number of layers.
        """
        # Middle third of layers
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

            dps_layer = (s_rsn / (s_ctx + s_rsn + EPS)).item()

            # Magnitude gate: suppress DPS when ||h_centered|| is small
            # (ratio is unreliable near the centroid)
            if self._tau is not None and self._tau > EPS:
                mag = torch.norm(h_centered).item()
                gate = 1.0 - math.exp(-(mag ** 2) / (self._tau ** 2))
                # Blend toward 0.5 (uninformative) when gate is low
                dps_layer = 0.5 + (dps_layer - 0.5) * gate

            dps_values.append(dps_layer)

        return float(np.mean(dps_values))

    def compute_grounding_direction(self, h_gen: Tensor) -> float:
        """
        Context-Grounding Direction (CGD) score.

        Cosine between (h_gen - prompt_center) and (context_center - prompt_center).
        Measures whether generation moves toward context or away from it.

        Returns (1 - cos_sim) / 2 in [0,1]. Higher = away from context = riskier.

        Reference: ContextFocus (arXiv:2601.04131) — adapted for training-free
        detection using prefill-derived directions.
        """
        if self._prompt_center is None or self._context_center is None:
            return 0.5

        with torch.no_grad():
            prompt_center = self._prompt_center.float()
            context_center = self._context_center.squeeze(0).float()

            d_ctx = context_center - prompt_center
            d_ctx_norm = torch.norm(d_ctx)
            if d_ctx_norm < EPS:
                return 0.5

            h = h_gen.float().squeeze()
            h_relative = h - prompt_center
            h_norm = torch.norm(h_relative)
            if h_norm < EPS:
                return 0.5

            cos_sim = (h_relative @ d_ctx) / (h_norm * d_ctx_norm)
            cos_sim = cos_sim.clamp(-1.0, 1.0)

        # Map: cos_sim = 1 (toward context) → 0 risk, cos_sim = -1 (away) → 1 risk
        return float((1.0 - cos_sim.item()) / 2.0)

    @property
    def context_basis(self) -> Tensor:
        """V_ctx for use by POS signal."""
        return self._context_basis

    @property
    def context_center(self) -> Tensor:
        """mu_context for use by POS signal."""
        return self._context_center
