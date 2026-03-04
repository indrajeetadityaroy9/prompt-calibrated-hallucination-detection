"""
Self-calibrating prompt statistics for AG-SAR.

Computes PIT reference distributions from prompt tail for DPS and POS signals.
CUS and SPT use direct mode with peer-derived variance.
"""

import math
from typing import Dict

import numpy as np
import torch

from .numerics import effective_rank
from .hooks import LayerHiddenStates


def adaptive_window(prompt_len: int) -> int:
    """Adaptive tail window: sqrt(prompt_len), capped at prompt_len."""
    return min(int(math.ceil(math.sqrt(prompt_len))), prompt_len)


def self_calibrate(
    *,
    dps_signal,
    jsd_signal,
    spt_signal,
    lm_head,
    final_norm,
    prompt_len: int,
    tail_per_layer: Dict[int, Dict],
    ctx_layer_idx: int,
) -> Dict[str, Dict]:
    """Self-calibrating prompt statistics for PIT normalization."""
    stats = {}
    first_key = next(iter(tail_per_layer))
    n_tail = tail_per_layer[first_key]["h_resid_mlp"].shape[0]

    # DPS: all-layer mean per tail position
    dps_values = []
    for t in range(n_tail):
        h_dict = {li: tail_per_layer[li]["h_resid_mlp"][t] for li in tail_per_layer}
        dps_values.append(dps_signal.compute_dps(h_dict))
    dps_arr = np.array(dps_values)
    stats["dps"] = {"sorted_vals": np.sort(dps_arr), "variance": float(np.var(dps_arr))}

    # POS: JSD-weighted override per tail position
    pos_values = []
    lm_head_dtype = lm_head.weight.dtype
    for t in range(n_tail):
        layer_states = {
            li: LayerHiddenStates(
                h_resid_attn=tail_per_layer[li]["h_resid_attn"][t],
                h_resid_mlp=tail_per_layer[li]["h_resid_mlp"][t],
            )
            for li in tail_per_layer
        }
        final_layer = max(layer_states.keys())
        h = layer_states[final_layer].h_resid_mlp
        if h.dim() == 1:
            h = h.unsqueeze(0).unsqueeze(0)
        elif h.dim() == 2:
            h = h.unsqueeze(0)
        with torch.no_grad():
            logits_t = lm_head(final_norm(h.to(dtype=lm_head_dtype))).squeeze()
        probs_t = torch.softmax(logits_t.float(), dim=-1)
        k = max(2, effective_rank(probs_t))
        cand = torch.topk(logits_t, min(k, len(logits_t))).indices
        token_id = logits_t.argmax().item()
        cand = torch.unique(torch.cat([cand, torch.tensor([token_id], device="cuda")]))
        pos_values.append(jsd_signal.compute_pos(layer_states, cand))
    pos_arr = np.array(pos_values)
    stats["pos"] = {"sorted_vals": np.sort(pos_arr), "variance": float(np.var(pos_arr))}

    # CUS: direct mode, variance = median of peer variances
    peer_vars = [stats[s]["variance"] for s in stats]
    stats["cus"] = {"mode": "direct", "variance": float(np.median(peer_vars))}

    # SPT: direct mode, peer-derived variance (MP edge is its own null model)
    spt_signal.reset()
    for t in range(n_tail):
        spt_signal.push(tail_per_layer[ctx_layer_idx]["h_resid_mlp"][t])
    all_peer_vars = [stats[s]["variance"] for s in stats if "variance" in stats[s]]
    stats["spt"] = {"mode": "direct", "variance": float(np.median(all_peer_vars))}

    return stats
