"""
Dual-Subspace Projection Score (DPS) and Context-Grounding Direction (CGD)
for AG-SAR hallucination detection.

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
from typing import Dict, List
from torch import Tensor

from ..numerics import EPS, otsu_threshold


def _find_spectral_gap_rank(
    S: Tensor,
    remove_top: int = 1,
    n_samples: int = None,
    n_features: int = None,
) -> int:
    """Rank via Marchenko-Pastur boundary (primary) or ratio gap (fallback). Marchenko & Pastur (1967)."""
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
            sigma2 = (noise_svs ** 2).mean().item() / n_samples
            if sigma2 > 0:
                mp_upper = sigma2 * (1 + math.sqrt(gamma)) ** 2
                k = int((S_trimmed ** 2 / n_samples > mp_upper).sum().item())
                if k >= 1:
                    return k + remove_top

    # Fallback: Otsu on log-ratios (data-driven, consistent with codebase)
    ratios = S_trimmed[:-1] / (S_trimmed[1:] + EPS)
    ratios = torch.clamp(ratios, min=EPS)  # Prevent log(0) → -inf
    if len(ratios) < 3:
        # Too few values for reliable Otsu; use argmax with conservative threshold
        if ratios.max().item() < 2.0:
            return len(S)
        return max(1, int(ratios.argmax().item()) + 1 + remove_top)
    log_ratios = torch.log(ratios).cpu().numpy()
    threshold = otsu_threshold(log_ratios)
    above = np.where(log_ratios >= threshold)[0]
    if len(above) == 0:
        return len(S)
    return max(1, int(above[0]) + 1 + remove_top)


class DualSubspaceGrounding:
    """
    DPS + CGD signals for AG-SAR.

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
        self._dps_layer_indices: List[int] = []

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
        """SVD of centered context hidden states → V_ctx with Marchenko-Pastur rank."""
        if context_hidden.shape[0] < 2:
            raise ValueError("Context must contain at least 2 tokens for SVD.")
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
        """Mean of non-context prompt tokens. CGD anchor."""
        self._prompt_center = prompt_hidden.float().mean(dim=0)

    def set_magnitude_tau(self, prefill_hidden: Tensor, context_center: Tensor) -> None:
        """tau = median(||h - mu_ctx||). DPS magnitude gate threshold."""
        h_centered = prefill_hidden.float() - context_center.squeeze(0).float()
        norms = torch.norm(h_centered, dim=-1)
        self._tau = max(float(norms.median().item()), EPS)

    def set_dps_layers(self, layer_indices: List[int]) -> None:
        """Set data-driven DPS layer indices (overrides middle-third default)."""
        self._dps_layer_indices = layer_indices

    def dps_from_hidden(self, h: Tensor) -> float:
        """DPS = gate × (s_rsn/(s_ctx+s_rsn)) + (1-gate) × 0.5. Gate = 1 - exp(-||h||²/τ²)."""
        V_ctx = self._context_basis.to(dtype=torch.float32)
        V_rsn = self._reasoning_basis.to(dtype=torch.float32)
        mu = self._context_center.squeeze(0).to(dtype=torch.float32)

        h = h.float().squeeze()
        h_centered = h - mu
        mag = torch.norm(h_centered).item()
        h_norm = mag + EPS

        s_ctx = torch.norm(V_ctx.T @ (V_ctx @ h_centered)).item() / h_norm
        s_rsn = torch.norm(V_rsn.T @ (V_rsn @ h_centered)).item() / h_norm

        dps_raw = s_rsn / (s_ctx + s_rsn + EPS)
        gate = 1.0 - math.exp(-(mag ** 2) / (self._tau ** 2))
        return 0.5 + (dps_raw - 0.5) * gate

    def compute_dps(
        self,
        layer_hidden_states: Dict[int, Tensor],
        n_layers: int,
    ) -> float:
        """Mean DPS over data-selected layers."""
        target_layers = [i for i in self._dps_layer_indices if i in layer_hidden_states]
        if not target_layers:
            return 0.5  # Neutral when no layers available
        dps_values = [self.dps_from_hidden(layer_hidden_states[i]) for i in target_layers]
        return float(np.mean(dps_values))

    def compute_grounding_direction(self, h_gen: Tensor) -> float:
        """CGD = (1 - cos(h_gen - prompt_center, ctx_center - prompt_center)) / 2. Higher = away from context."""
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
