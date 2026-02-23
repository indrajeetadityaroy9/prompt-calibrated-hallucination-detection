"""
Candidate-set JSD: POS (parametric override) + DoLa (layer contrast).

JSD between pre-MLP and post-MLP distributions restricted to adaptive candidate set.
Uses model's learned final LayerNorm for proper Logit Lens projection.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates
from ..numerics import safe_softmax, safe_log_softmax, safe_jsd, EPS, otsu_threshold


class CandidateJSDSignal:
    """JSD(softmax(W_U·norm(h_pre)[cand]), softmax(W_U·norm(h_post)[cand])) for POS + DoLa."""

    def __init__(self, lm_head: nn.Linear, final_norm: nn.Module):
        self.lm_head = lm_head
        self.final_norm = final_norm
        self._context_basis = None
        self._prompt_jsd_sigma = None
        self._prompt_jsd_values = None

    def compute_layer_jsd(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
        candidate_set: Tensor,
    ) -> float:
        """Candidate-set JSD for MLP-induced shift: JSD(softmax(W_U·norm(h_pre)[cand]), softmax(W_U·norm(h_post)[cand]))."""
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
        """Compute per-layer JSD for all captured layers."""
        results = {}
        for layer_idx, states in layer_states.items():
            jsd = self.compute_layer_jsd(
                states.h_resid_attn,
                states.h_resid_mlp,
                candidate_set,
            )
            results[layer_idx] = jsd
        return results

    # --- POS (Parametric Override Score) ---

    def set_context_basis(self, V_ctx: Tensor) -> None:
        """Store context subspace basis for directional decomposition."""
        self._context_basis = V_ctx

    def compute_directional_override(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
    ) -> float:
        """Override = max(0, 1 - ||proj_ctx(delta)||/||delta|| / sqrt(k/d)). 0=context-aligned, 1=context-avoidant."""
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
        """POS: per-layer JSD → Otsu active layers → directional override → mean."""
        # Per-layer JSD
        layer_jsds = self.compute_all_layers(layer_states, candidate_set)
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

    # --- DoLa Layer-Contrast Signal ---

    def compute_dola_score(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        generated_token_id: int,
        candidate_set: Tensor,
    ) -> float:
        """DoLa: log P_final(token) - log P_premature(token). Premature = argmax JSD over early layers. Chuang et al. (ICLR 2024)."""
        layer_indices = sorted(layer_states.keys())
        final_layer = layer_indices[-1]

        # Data-driven early/late split: Otsu on layer indices finds natural gap.
        # For uniform indices, returns midpoint (identical to len//2).
        # For non-uniform layer subsets, adapts to actual distribution.
        if len(layer_indices) >= 6:
            idx_array = np.array([i for i in layer_indices if i != final_layer], dtype=float)
            if len(idx_array) >= 2:
                split_val = otsu_threshold(idx_array)
                early_layers = [i for i in layer_indices if i < split_val and i != final_layer]
                if not early_layers:
                    early_layers = layer_indices[:len(layer_indices) // 2]
            else:
                early_layers = layer_indices[:len(layer_indices) // 2]
        else:
            early_layers = layer_indices[:len(layer_indices) // 2]

        lm_head_dtype = self.lm_head.weight.dtype
        lm_head_device = self.lm_head.weight.device
        w_subset = self.lm_head.weight[candidate_set]  # [K, Dim]

        with torch.no_grad():
            # Final layer distribution on candidate set
            h_final = layer_states[final_layer].h_resid_mlp
            if h_final.dim() == 1:
                h_final = h_final.unsqueeze(0)
            h_final = h_final.to(dtype=lm_head_dtype, device=lm_head_device)
            h_final_norm = self.final_norm(h_final)
            z_final = F.linear(h_final_norm, w_subset).float()
            p_final = safe_softmax(z_final, dim=-1)
            log_p_final = safe_log_softmax(z_final, dim=-1)

            # Find premature layer with maximum JSD to final
            max_jsd = -1.0
            best_log_p = None
            for j in early_layers:
                h_j = layer_states[j].h_resid_mlp
                if h_j.dim() == 1:
                    h_j = h_j.unsqueeze(0)
                h_j = h_j.to(dtype=lm_head_dtype, device=lm_head_device)
                h_j_norm = self.final_norm(h_j)
                z_j = F.linear(h_j_norm, w_subset).float()
                p_j = safe_softmax(z_j, dim=-1)

                jsd = safe_jsd(p_final, p_j)
                if jsd > max_jsd:
                    max_jsd = jsd
                    best_log_p = safe_log_softmax(z_j, dim=-1)

            # Guard: if no early layers produced a valid contrast, return neutral
            if best_log_p is None:
                return 0.0

            # Find generated token in candidate set
            token_mask = (candidate_set == generated_token_id)
            if not token_mask.any():
                return 0.0  # Token not in candidate set (should not happen)
            idx = token_mask.nonzero(as_tuple=True)[0][0].item()

            # DoLa contrast: log-prob difference for the actual generated token.
            # Negated so higher = riskier (low contrast = hallucination risk).
            dola = best_log_p.squeeze(0)[idx].item() - log_p_final.squeeze(0)[idx].item()

        return float(dola)
