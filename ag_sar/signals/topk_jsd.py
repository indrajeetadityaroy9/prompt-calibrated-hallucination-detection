"""
Candidate-set JSD signal for MLP-induced distribution shift.

Computes JSD between pre-MLP and post-MLP token distributions,
restricted to a candidate set for efficiency.

IMPORTANT: Uses model's learned final LayerNorm (with gain weights γ) for
proper Logit Lens projection. The lm_head expects normalized features that
match the training distribution - using parameter-free RMS norm produces
unscaled features that degrade signal quality.

Reference: "Logit Lens" technique requires matching the model's normalization.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates
from ..numerics import safe_softmax, safe_jsd, EPS, otsu_threshold


class CandidateJSDSignal:
    """
    MLP-induced distribution shift computed on CANDIDATE SET (approximation).

    jsd_cand = JSD(p_pre_cand, p_post_cand)

    Where:
    - z_pre = W_U · final_norm(h_resid_attn)  # logits from pre-MLP residual
    - z_post = W_U · final_norm(h_resid_mlp)  # logits from post-MLP residual
    - p_*_cand = softmax(z_*[candidate_set])  # restricted to candidates

    Uses model's learned final_norm (with γ weights) for proper Logit Lens.
    The lm_head was trained with final_norm applied, so intermediate projections
    must use the same normalization for valid logit space comparisons.

    This is an approximation to full-vocab JSD, trading accuracy for efficiency.
    Approximation quality is validated by topk_mass sanity metric.
    """

    def __init__(self, lm_head: nn.Linear, final_norm: nn.Module):
        """
        Initialize JSD signal computer.

        Args:
            lm_head: The model's language model head (unembedding matrix)
            final_norm: The model's final LayerNorm (model.model.norm for LLaMA).
        """
        self.lm_head = lm_head
        self.final_norm = final_norm
        self._context_basis = None
        self._prompt_jsd_mu = None
        self._prompt_jsd_sigma = None

    def compute_layer_jsd(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
        candidate_set: Tensor,
    ) -> float:
        """
        Compute candidate-set JSD for MLP-induced shift on residual stream.

        Args:
            h_resid_attn: Pre-MLP residual hidden state [batch, hidden_dim] or [hidden_dim]
            h_resid_mlp: Post-MLP residual hidden state [batch, hidden_dim] or [hidden_dim]
            candidate_set: Indices of candidate tokens [candidate_size]

        Returns:
            JSD in bits, bounded [0, 1]
        """
        # Handle batch dimension - squeeze if present
        if h_resid_attn.dim() == 1:
            h_resid_attn = h_resid_attn.unsqueeze(0)
            h_resid_mlp = h_resid_mlp.unsqueeze(0)

        # Get lm_head dtype and device for consistent computation
        lm_head_dtype = self.lm_head.weight.dtype
        lm_head_device = self.lm_head.weight.device

        # Convert to same dtype as lm_head for matrix multiplication
        h_resid_attn = h_resid_attn.to(dtype=lm_head_dtype, device=lm_head_device)
        h_resid_mlp = h_resid_mlp.to(dtype=lm_head_dtype, device=lm_head_device)

        # 1. Normalize using model's final_norm (with learned γ) for proper Logit Lens
        with torch.no_grad():
            h_pre_norm = self.final_norm(h_resid_attn)
            h_post_norm = self.final_norm(h_resid_mlp)

        # 2. Slice weights (copy overhead)
        w_subset = self.lm_head.weight[candidate_set] # [K, Dim]

        # 3. Compute logits
        z_pre_cand = F.linear(h_pre_norm, w_subset)
        z_post_cand = F.linear(h_post_norm, w_subset)

        # Ensure float32 for softmax stability
        z_pre_cand = z_pre_cand.float()
        z_post_cand = z_post_cand.float()

        # Compute softmax over candidates
        p_pre_cand = safe_softmax(z_pre_cand, dim=-1)
        p_post_cand = safe_softmax(z_post_cand, dim=-1)

        # Compute JSD
        jsd = safe_jsd(p_pre_cand, p_post_cand)

        return jsd

    def compute_all_layers(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> Dict[int, float]:
        """
        Compute JSD for all captured layers.

        Args:
            layer_states: Dict mapping layer_idx to LayerHiddenStates
            candidate_set: Indices of candidate tokens

        Returns:
            Dict mapping layer_idx to JSD value
        """
        results = {}
        for layer_idx, states in layer_states.items():
            jsd = self.compute_layer_jsd(
                states.h_resid_attn,
                states.h_resid_mlp,
                candidate_set,
            )
            results[layer_idx] = jsd
        return results

    def compute_aggregated(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        candidate_set: Tensor,
        aggregation: str = "sum",
    ) -> float:
        """
        Compute aggregated JSD across layers.

        Args:
            layer_states: Dict mapping layer_idx to LayerHiddenStates
            candidate_set: Indices of candidate tokens
            aggregation: How to aggregate ("sum", "mean", "max")

        Returns:
            Aggregated JSD value
        """
        layer_jsds = self.compute_all_layers(layer_states, candidate_set)

        if not layer_jsds:
            return 0.0

        values = list(layer_jsds.values())

        if aggregation == "sum":
            return sum(values)
        elif aggregation == "mean":
            return sum(values) / len(values)
        elif aggregation == "max":
            return max(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    # --- POS (Parametric Override Score) methods for DSG ---

    def set_context_basis(self, V_ctx: Tensor) -> None:
        """Store context subspace basis for directional decomposition."""
        self._context_basis = V_ctx

    def set_prompt_jsd_stats(self, mu: float, sigma: float) -> None:
        """Store prompt JSD statistics for active layer threshold."""
        self._prompt_jsd_mu = mu
        self._prompt_jsd_sigma = sigma

    def compute_directional_override(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
    ) -> float:
        """
        Compute override ratio: how much LESS context-aligned is the MLP shift
        compared to a random baseline.

        delta = final_norm(h_post) - final_norm(h_pre)
        context_ratio = ||proj_ctx(delta)|| / ||delta||
        expected_ratio = sqrt(k/d)  (random baseline for k-dim subspace in d-dim space)
        override = max(0, 1 - context_ratio / expected_ratio)

        Returns 0 when MLP shift is context-aligned (no risk),
        approaches 1 when MLP shift avoids context subspace (high risk).
        """
        if self._context_basis is None:
            return 0.0

        # Handle batch dimension
        if h_resid_attn.dim() == 1:
            h_resid_attn = h_resid_attn.unsqueeze(0)
            h_resid_mlp = h_resid_mlp.unsqueeze(0)

        lm_head_dtype = self.lm_head.weight.dtype
        lm_head_device = self.lm_head.weight.device
        h_resid_attn = h_resid_attn.to(dtype=lm_head_dtype, device=lm_head_device)
        h_resid_mlp = h_resid_mlp.to(dtype=lm_head_dtype, device=lm_head_device)

        with torch.no_grad():
            h_pre = self.final_norm(h_resid_attn).float().squeeze(0)
            h_post = self.final_norm(h_resid_mlp).float().squeeze(0)

            delta = h_post - h_pre
            delta_norm = torch.norm(delta)

            if delta_norm < EPS:
                return 0.0

            V = self._context_basis.to(dtype=torch.float32, device=delta.device)
            proj = V.T @ (V @ delta)
            context_ratio = torch.norm(proj) / (delta_norm + EPS)

            # Random baseline: expected context_ratio for a random unit vector
            k = V.shape[0]  # context subspace rank
            d = V.shape[1]  # hidden dim
            expected_ratio = (k / d) ** 0.5

            # Normalize: 0 when context-aligned, 1 when context-avoidant
            override = 1.0 - context_ratio / (expected_ratio + EPS)

        return float(override.clamp(0, 1).item())

    def compute_pos(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> float:
        """
        Compute Parametric Override Score for a single token.

        1. Compute per-layer JSD
        2. Select active layers via Otsu bimodal threshold (zero parameters)
        3. Compute directional override for active layers
        4. Return mean override

        Replaces mu+sigma threshold (assumes Gaussian on [0,1]-bounded JSD)
        with Otsu's method — same principled approach used for copying heads.

        Fallback: ceil(sqrt(n_layers)) — natural scale for subset selection.
        """
        if self._prompt_jsd_mu is None:
            return 0.0

        # Per-layer JSD
        layer_jsds = self.compute_all_layers(layer_states, candidate_set)
        if not layer_jsds:
            return 0.0

        # Otsu threshold on current-token JSD values (zero parameters)
        jsd_values = np.array(list(layer_jsds.values()))
        threshold = otsu_threshold(jsd_values)

        # Active layers + directional override
        overrides = []
        for layer_idx, jsd_val in layer_jsds.items():
            if jsd_val > threshold:
                states = layer_states[layer_idx]
                override = self.compute_directional_override(
                    states.h_resid_attn, states.h_resid_mlp
                )
                overrides.append(override)

        if not overrides:
            # Fallback: ceil(sqrt(n_layers)) — natural scale
            import math
            k = max(1, int(math.ceil(math.sqrt(len(layer_jsds)))))
            sorted_layers = sorted(layer_jsds.items(), key=lambda x: x[1], reverse=True)
            for layer_idx, _ in sorted_layers[:k]:
                states = layer_states[layer_idx]
                override = self.compute_directional_override(
                    states.h_resid_attn, states.h_resid_mlp
                )
                overrides.append(override)

        return float(np.mean(overrides))
