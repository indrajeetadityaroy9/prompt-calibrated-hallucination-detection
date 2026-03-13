"""
Candidate-set JSD: POS (parametric override) signal.

JSD between pre-MLP and post-MLP distributions restricted to adaptive candidate set.
JSD-weighted directional override across all layers — no threshold selection needed.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates
from ..numerics import jsd, EPS


class CandidateJSDSignal:
    """JSD(softmax(W_U·norm(h_pre)[cand]), softmax(W_U·norm(h_post)[cand])) for POS."""

    def __init__(self, lm_head: nn.Linear, final_norm: nn.Module):
        self.lm_head = lm_head
        self.final_norm = final_norm
        self._context_basis = None

    def _prepare_pair(self, h_attn: Tensor, h_mlp: Tensor):
        """Ensure 2D and cast to lm_head dtype on cuda."""
        if h_attn.dim() == 1:
            h_attn = h_attn.unsqueeze(0)
            h_mlp = h_mlp.unsqueeze(0)
        dtype = self.lm_head.weight.dtype
        device = self.lm_head.weight.device
        return h_attn.to(dtype=dtype, device=device), h_mlp.to(dtype=dtype, device=device)

    def compute_layer_jsd(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
        candidate_set: Tensor,
    ) -> float:
        """Candidate-set JSD for MLP-induced shift."""
        h_resid_attn, h_resid_mlp = self._prepare_pair(h_resid_attn, h_resid_mlp)

        with torch.no_grad():
            h_pre_norm = self.final_norm(h_resid_attn)
            h_post_norm = self.final_norm(h_resid_mlp)

        w_subset = self.lm_head.weight[candidate_set]
        z_pre_cand = F.linear(h_pre_norm, w_subset).float()
        z_post_cand = F.linear(h_post_norm, w_subset).float()

        p_pre_cand = F.softmax(z_pre_cand, dim=-1)
        p_post_cand = F.softmax(z_post_cand, dim=-1)

        return jsd(p_pre_cand, p_post_cand)

    def set_context_basis(self, V_ctx: Tensor) -> None:
        """Store context subspace basis for directional decomposition."""
        self._context_basis = V_ctx

    def compute_directional_override(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
    ) -> float:
        """Override = clamp(1 - context_ratio / expected_ratio, 0, 1)."""
        h_resid_attn, h_resid_mlp = self._prepare_pair(h_resid_attn, h_resid_mlp)

        with torch.no_grad():
            h_pre = self.final_norm(h_resid_attn).float().squeeze(0)
            h_post = self.final_norm(h_resid_mlp).float().squeeze(0)

            delta = h_post - h_pre
            delta_norm = torch.norm(delta)

            V = self._context_basis.to(dtype=torch.float32, device=delta.device)
            proj = V.T @ (V @ delta)
            context_ratio = torch.norm(proj) / (delta_norm + EPS)

            k = V.shape[0]
            d = V.shape[1]
            expected_ratio = (k / d) ** 0.5

            override = 1.0 - context_ratio / (expected_ratio + EPS)

        return float(override.clamp(0, 1).item())

    def compute_pos(
        self,
        layer_states: dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> float:
        """POS: JSD-weighted mean of directional override across all layers."""
        weighted_sum = 0.0
        weight_sum = 0.0
        for layer_idx, states in layer_states.items():
            jsd_val = self.compute_layer_jsd(
                states.h_resid_attn, states.h_resid_mlp, candidate_set,
            )
            override = self.compute_directional_override(
                states.h_resid_attn, states.h_resid_mlp,
            )
            weighted_sum += jsd_val * override
            weight_sum += jsd_val
        return float(weighted_sum / (weight_sum + EPS))
