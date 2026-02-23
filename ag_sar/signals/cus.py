"""
Context Utilization Score (CUS) — Lookback Ratio Bimodality signal for AG-SAR.

Measures bimodality of per-head lookback ratios across all attention heads.
Bimodal LR distribution (some heads attend to context, others don't) indicates
grounded generation. Unimodal distribution indicates hallucination.

CUS = 1 - Otsu coefficient of LR vector.
Range [0,1], higher = more unimodal = riskier.

Reference: Chuang et al. (EMNLP 2024) "Lookback Lens for Detecting and
Mitigating Contextual Hallucinations"
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Tuple

from ..numerics import otsu_threshold as _otsu_threshold, EPS


def compute_layer_affinity(
    attn_weights: Tensor,
    context_mask: Tensor,
) -> Tensor:
    """Per-head copying affinity: mean_{t in ctx} max_{s!=t in ctx} attn[h,t,s]."""
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # [num_heads, seq, seq]

    ctx_indices = context_mask.nonzero(as_tuple=True)[0]
    ctx_len = len(ctx_indices)
    if ctx_len < 2:
        return torch.zeros(attn_weights.shape[0], device=attn_weights.device)

    # Fast path for contiguous mask
    if ctx_len == ctx_indices[-1] - ctx_indices[0] + 1:
        start, end = ctx_indices[0].item(), ctx_indices[-1].item() + 1
        ctx_attn = attn_weights[:, start:end, start:end].float()
    else:
        ctx_attn = attn_weights[:, ctx_indices][:, :, ctx_indices].float()

    # Mask diagonal (s != t)
    mask = ~torch.eye(ctx_len, device=ctx_attn.device, dtype=torch.bool)
    ctx_attn_masked = ctx_attn * mask + (~mask).to(ctx_attn.dtype) * torch.finfo(ctx_attn.dtype).min

    # max_{s!=t} attn[h, t, s] for each head and token
    max_vals = ctx_attn_masked.max(dim=-1).values  # [num_heads, ctx_len]

    # mean over context tokens
    affinity = max_vals.mean(dim=-1)  # [num_heads]
    return affinity


def identify_copying_heads(
    affinities: Dict[int, Tensor],
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """Otsu threshold on per-head affinities → copying heads + affinity map."""
    all_vals = []
    all_keys = []
    for layer_idx, aff in affinities.items():
        for h in range(aff.shape[0]):
            all_vals.append(aff[h].item())
            all_keys.append((layer_idx, h))

    if not all_vals:
        return [], {}

    threshold = _otsu_threshold(np.array(all_vals))
    heads = [k for k, v in zip(all_keys, all_vals) if v >= threshold]
    affinity_map = {k: v for k, v in zip(all_keys, all_vals) if v >= threshold}
    return heads, affinity_map


class ContextUtilizationSignal:
    """
    Lookback Ratio Bimodality as context utilization signal.

    For each generation token, computes the lookback ratio (fraction of
    attention on context) for ALL attention heads, then measures bimodality
    of the LR distribution via the Otsu coefficient.

    Bimodal = healthy: copying heads attend to context (high LR), non-copying
    heads don't (low LR). Unimodal = hallucinating: all heads similar.

    CUS(t) = 1 - (Otsu inter-class variance / total variance)
    Range [0,1], higher = more unimodal = riskier.
    """

    def __init__(
        self,
        copying_heads: List[Tuple[int, int]],
        affinity_map: Dict[Tuple[int, int], float],
        prompt_len: int,
        n_layers: int,
    ):
        self.copying_heads = set(copying_heads)
        self.affinity_map = affinity_map
        self.prompt_len = prompt_len
        self.n_layers = n_layers

    def compute_lookback_ratio_signal(
        self, attention_slices: Dict[int, Tensor], prompt_len: int
    ) -> float:
        """CUS = 1 - weighted_otsu_coefficient(LR). Affinity-weighted bimodality of lookback ratios."""
        lr_values = []
        weights = []
        for layer_idx, attn in attention_slices.items():
            # attn shape: [num_heads, kv_len]
            n_heads = attn.shape[0]
            for h in range(n_heads):
                context_mass = attn[h, :prompt_len].float().sum().item()
                lr_values.append(context_mass)
                # Weight by copying affinity: identified heads contribute more
                weights.append(1.0 + self.affinity_map.get((layer_idx, h), 0.0))

        lr = np.array(lr_values)
        w = np.array(weights)
        w = w / w.sum()  # Normalize to sum to 1

        # Weighted mean and variance
        weighted_mean = np.average(lr, weights=w)
        total_var = np.average((lr - weighted_mean) ** 2, weights=w)
        # Otsu threshold to split into two classes
        threshold = _otsu_threshold(lr)
        mask_low = lr <= threshold
        mask_high = lr > threshold

        if not mask_low.any() or not mask_high.any():
            return 1.0  # Unimodal → no discrimination → risky

        # Weighted Otsu coefficient: inter-class variance / total variance
        w1 = w[mask_low].sum()
        w2 = w[mask_high].sum()
        mu1 = np.average(lr[mask_low], weights=w[mask_low])
        mu2 = np.average(lr[mask_high], weights=w[mask_high])
        inter_var = w1 * w2 * (mu1 - mu2) ** 2
        otsu_coeff = inter_var / (total_var + EPS)

        # High bimodality (otsu_coeff→1) = healthy separation = low risk
        # Low bimodality (otsu_coeff→0) = uniform attention = high risk
        return float(np.clip(1.0 - otsu_coeff, 0.0, 1.0))
