"""
Dual-Subspace Projection Score (DPS) for AG-SAR hallucination detection.

DPS(t) = s_rsn / (s_ctx + s_rsn + eps), range [0,1], higher = riskier.
Magnitude-gated: DPS blends toward 0.5 when ||h_centered|| is small.
"""

import math
from typing import Dict

import torch
import numpy as np
from torch import Tensor

from ..numerics import EPS, effective_rank


class DualSubspaceGrounding:
    """DPS signal for AG-SAR."""

    def __init__(self, lm_head_weight: Tensor):
        self._reasoning_basis = self._compute_reasoning_basis(lm_head_weight)
        self._context_basis: Tensor = None  # type: ignore[assignment]
        self._context_center: Tensor = None  # type: ignore[assignment]
        self._prompt_center: Tensor = None  # type: ignore[assignment]
        self._tau: float = None  # type: ignore[assignment]

    def _compute_reasoning_basis(self, lm_head_weight: Tensor) -> Tensor:
        """SVD of lm_head.weight, keep bottom singular vectors via effective rank."""
        with torch.no_grad():
            w = lm_head_weight.float()
            _, S, Vh = torch.linalg.svd(w, full_matrices=False)
            k_r = effective_rank(S.flip(0))
            V_rsn = Vh[-k_r:]
        return V_rsn.to("cuda")

    def set_context_basis(self, context_hidden: Tensor) -> None:
        """SVD of centered context hidden states -> V_ctx with effective rank."""
        context_hidden = context_hidden.float()
        context_mean = context_hidden.mean(dim=0, keepdim=True)
        context_centered = context_hidden - context_mean
        _, S, Vh = torch.linalg.svd(context_centered, full_matrices=False)
        k = effective_rank(S)
        self._context_basis = Vh[:k]
        self._context_center = context_mean

    def set_prompt_center(self, prompt_hidden: Tensor) -> None:
        """Mean of non-context prompt tokens."""
        self._prompt_center = prompt_hidden.float().mean(dim=0)

    def set_magnitude_tau(self, prefill_hidden: Tensor, context_center: Tensor) -> None:
        """tau = median(||h - mu_ctx||)."""
        h_centered = prefill_hidden.float() - context_center.squeeze(0).float()
        norms = torch.norm(h_centered, dim=-1)
        self._tau = float(norms.median().item())

    def dps_from_hidden(self, h: Tensor) -> float:
        """DPS = gate * (s_rsn/(s_ctx+s_rsn)) + (1-gate) * 0.5."""
        V_ctx = self._context_basis.to(dtype=torch.float32)
        V_rsn = self._reasoning_basis.to(dtype=torch.float32)
        mu = self._context_center.squeeze(0).to(dtype=torch.float32)

        h = h.float().squeeze()
        h_centered = h - mu
        mag = torch.norm(h_centered).item()
        h_norm = mag + EPS

        s_ctx = torch.norm(V_ctx @ h_centered).item() / h_norm
        s_rsn = torch.norm(V_rsn @ h_centered).item() / h_norm

        dps_raw = s_rsn / (s_ctx + s_rsn + EPS)
        gate = 1.0 - math.exp(-(mag ** 2) / (self._tau ** 2))
        return 0.5 + (dps_raw - 0.5) * gate

    def compute_dps(self, layer_hidden_states: Dict[int, Tensor]) -> float:
        """Mean DPS over all layers."""
        dps_values = [self.dps_from_hidden(h) for h in layer_hidden_states.values()]
        return float(np.mean(dps_values))

    @property
    def context_basis(self) -> Tensor:
        """V_ctx for use by POS signal."""
        return self._context_basis
