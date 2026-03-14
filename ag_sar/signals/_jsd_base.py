"""
Candidate-set JSD: MLP transformation magnitude signal.

JSD between pre-MLP and post-MLP distributions restricted to adaptive candidate set.
All-layer mean JSD — no threshold selection needed.
"""

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates
from ..numerics import jsd


class CandidateJSDSignal:
    """JSD(softmax(W_U·norm(h_pre)[cand]), softmax(W_U·norm(h_post)[cand])) for MLP signal."""

    def __init__(self, lm_head: nn.Linear, final_norm: nn.Module):
        self.lm_head = lm_head
        self.final_norm = final_norm

    def compute_layer_jsd(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
        candidate_set: Tensor,
    ) -> float:
        """Candidate-set JSD for MLP-induced shift at a single layer."""
        dtype = self.lm_head.weight.dtype
        h_resid_attn = h_resid_attn.to(dtype=dtype)
        h_resid_mlp = h_resid_mlp.to(dtype=dtype)

        with torch.no_grad():
            h_pre_norm = self.final_norm(h_resid_attn)
            h_post_norm = self.final_norm(h_resid_mlp)

        w_subset = self.lm_head.weight[candidate_set]
        z_pre_cand = F.linear(h_pre_norm, w_subset).float()
        z_post_cand = F.linear(h_post_norm, w_subset).float()

        p_pre_cand = F.softmax(z_pre_cand, dim=-1)
        p_post_cand = F.softmax(z_post_cand, dim=-1)

        return jsd(p_pre_cand, p_post_cand)

    def compute_mlp_jsd(
        self,
        layer_states: dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> float:
        """MLP signal: mean JSD across all layers."""
        return float(np.mean([
            self.compute_layer_jsd(s.h_resid_attn, s.h_resid_mlp, candidate_set)
            for s in layer_states.values()
        ]))
