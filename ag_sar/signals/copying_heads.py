"""
Context Utilization Score (CUS) — Attention-based hallucination signal for DSG.

Identifies copying heads during prefill and tracks their context attention
during generation. Higher CUS = less context utilization = riskier.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Tuple

from ..numerics import otsu_threshold as _otsu_threshold


def compute_layer_affinity(
    attn_weights: Tensor,
    ctx_start: int,
    ctx_end: int,
) -> Tensor:
    """
    Compute per-head copying affinity from attention weights.

    affinity(h) = mean_{t in context} max_{s in context, s!=t} attn[h, t, s]

    Args:
        attn_weights: [batch, num_heads, seq, seq] or [num_heads, seq, seq]
        ctx_start: Start index of context tokens
        ctx_end: End index of context tokens (exclusive)

    Returns:
        [num_heads] affinity scores
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # [num_heads, seq, seq]

    ctx_len = ctx_end - ctx_start
    if ctx_len < 2:
        return torch.zeros(attn_weights.shape[0], device=attn_weights.device)

    # Extract context-to-context attention [num_heads, ctx_len, ctx_len]
    ctx_attn = attn_weights[:, ctx_start:ctx_end, ctx_start:ctx_end].float()

    # Mask diagonal (s != t)
    mask = ~torch.eye(ctx_len, device=ctx_attn.device, dtype=torch.bool)
    ctx_attn_masked = ctx_attn * mask + (~mask).float() * (-1e9)

    # max_{s!=t} attn[h, t, s] for each head and token
    max_vals = ctx_attn_masked.max(dim=-1).values  # [num_heads, ctx_len]

    # mean over context tokens
    affinity = max_vals.mean(dim=-1)  # [num_heads]
    return affinity


def identify_copying_heads(
    affinities: Dict[int, Tensor],
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Identify copying heads from precomputed per-layer affinities.

    Uses Otsu's method for parameter-free bimodal thresholding.

    Args:
        affinities: layer_idx -> [num_heads] affinity tensors

    Returns:
        Tuple of:
        - List of (layer_idx, head_idx) tuples for copying heads
        - Dict mapping (layer_idx, head_idx) -> affinity score (for weighting)
    """
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
    Compute CUS per generation token using identified copying heads.

    CUS(t) = 1 - weighted_avg_{copying_heads} sum_{s < prompt_len} attn[l,h][t, s]
    Range [0,1], higher = less context utilization = riskier.

    Weighting: Uses copying affinity scores from head identification (data-driven)
    instead of linear layer-depth heuristic. Heads with higher copying affinity
    are better indicators of context utilization.

    Reference: Wu et al. (2024) "Retrieval Head Mechanistically Explains
    Long-Context Factuality"
    """

    def __init__(
        self,
        copying_heads: List[Tuple[int, int]],
        prompt_len: int,
        n_layers: int,
        head_affinities: Dict[Tuple[int, int], float] = None,
    ):
        self.copying_heads = copying_heads
        self.prompt_len = prompt_len
        self.n_layers = n_layers
        self._head_affinities = head_affinities or {}

    def compute_cus(self, attention_slices: Dict[int, Tensor]) -> float:
        """
        Compute CUS for a single generation token.

        Uses affinity-based weighting: heads with higher copying affinity
        (measured during identification) contribute more to the score.
        This is data-driven — no arbitrary layer-position assumptions.

        Args:
            attention_slices: layer_idx -> [num_heads, kv_len] attention for last position

        Returns:
            CUS value in [0, 1]
        """
        if not self.copying_heads:
            return 0.5  # Neutral if no copying heads

        weighted_masses = []
        weights = []
        for layer_idx, head_idx in self.copying_heads:
            if layer_idx not in attention_slices:
                continue
            attn = attention_slices[layer_idx]  # [num_heads, kv_len]
            if head_idx >= attn.shape[0]:
                continue
            # Sum attention to context positions
            context_mass = attn[head_idx, :self.prompt_len].float().sum().item()
            # Data-driven weight: copying affinity from identification phase
            w = self._head_affinities.get((layer_idx, head_idx), 1.0)
            weighted_masses.append(context_mass * w)
            weights.append(w)

        if not weighted_masses:
            return 0.5

        weighted_avg = sum(weighted_masses) / sum(weights)
        return float(np.clip(1.0 - weighted_avg, 0.0, 1.0))
