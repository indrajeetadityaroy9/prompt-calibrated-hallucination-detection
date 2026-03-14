"""
Self-calibrating prompt statistics for AG-SAR with cross-signal precision.

Computes PIT reference distributions from prompt tail for PSP and MLP signals.
ENT, SPT, and spectral gap use direct mode.
Full cross-signal covariance matrix captures joint signal geometry for
precision-weighted fusion (Hartung, Knapp & Sinha, 2008).
"""

import math

import numpy as np
import torch
from sklearn.covariance import ledoit_wolf

from .numerics import EPS, effective_rank
from .hooks import LayerHiddenStates


def adaptive_window(prompt_len: int) -> int:
    """Adaptive tail window: sqrt(prompt_len), capped at prompt_len."""
    return min(int(math.ceil(math.sqrt(prompt_len))), prompt_len)


def self_calibrate(
    *,
    psp_signal,
    jsd_signal,
    spt_signal,
    lm_head,
    final_norm,
    prompt_len: int,
    tail_per_layer: dict[int, dict],
    mid_layer_idx: int,
) -> dict[str, dict]:
    """Self-calibrating prompt statistics with cross-signal precision matrix.

    Returns per-signal stats (sorted_vals, variance, mode) plus a shared
    '_cross_signal_precision' entry containing the full inverse covariance matrix.
    """
    stats = {}
    first_key = next(iter(tail_per_layer))
    n_tail = tail_per_layer[first_key]["h_resid_mlp"].shape[0]

    # PSP: all-layer mean per tail position
    psp_values = []
    for t in range(n_tail):
        h_dict = {li: tail_per_layer[li]["h_resid_mlp"][t] for li in tail_per_layer}
        psp_values.append(psp_signal.compute_psp(h_dict))
    psp_arr = np.array(psp_values)
    stats["psp"] = {"sorted_vals": np.sort(psp_arr), "variance": float(np.var(psp_arr))}

    # MLP: JSD per tail position
    mlp_values = []
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
        h = layer_states[final_layer].h_resid_mlp.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits_t = lm_head(final_norm(h.to(dtype=lm_head_dtype))).squeeze()
        probs_t = torch.softmax(logits_t.float(), dim=-1)
        k = max(2, effective_rank(probs_t))
        cand = torch.topk(logits_t, min(k, len(logits_t))).indices
        mlp_values.append(jsd_signal.compute_mlp_jsd(layer_states, cand))
    mlp_arr = np.array(mlp_values)
    stats["mlp"] = {"sorted_vals": np.sort(mlp_arr), "variance": float(np.var(mlp_arr))}

    # ENT: direct mode — variance derived from peer signals (PSP, MLP)
    stats["ent"] = {"mode": "direct", "variance": float(np.median([stats[s]["variance"] for s in stats]))}

    # SPT + spectral gap: compute incrementally as window fills
    spt_signal.reset()
    spt_values = []
    gap_values = []
    for t in range(n_tail):
        spt_signal.push(tail_per_layer[mid_layer_idx]["h_resid_mlp"][t])
        if spt_signal.window_len >= 2:
            spt_val, gap_val = spt_signal.compute_spt()
            spt_values.append(spt_val)
            gap_values.append(gap_val)

    stats["spt"] = {"mode": "direct", "variance": float(np.var(spt_values))}
    stats["spectral_gap"] = {"mode": "direct", "variance": float(np.var(gap_values))}

    # --- Cross-signal precision matrix ---
    # Build from the two signals with genuine per-position variation (PSP, MLP).
    # ENT, SPT, and spectral_gap lack per-position tail samples.
    # Strategy: compute the 2×2 PSP-MLP covariance to capture their coupling,
    # then embed into a 5×5 diagonal precision with the 2×2 off-diagonal block
    # for the PSP-MLP subspace.

    diag_vars = np.array([
        stats["ent"]["variance"],
        stats["mlp"]["variance"],
        stats["psp"]["variance"],
        stats["spt"]["variance"],
        stats["spectral_gap"]["variance"],
    ])

    # Start from diagonal precision, then embed PSP-MLP cross-correlation
    precision = np.diag(1.0 / np.maximum(diag_vars, EPS))

    psp_mlp_matrix = np.column_stack([mlp_arr, psp_arr])  # (n_tail, 2)
    # Ledoit-Wolf shrinkage: parameter-free optimal regularization (Ledoit & Wolf, 2004).
    cov_2x2, _ = ledoit_wolf(psp_mlp_matrix)
    prec_2x2 = np.linalg.inv(cov_2x2)
    # Embed into positions [1,1],[1,2],[2,1],[2,2] (MLP=idx1, PSP=idx2)
    precision[1, 1] = prec_2x2[0, 0]
    precision[1, 2] = prec_2x2[0, 1]
    precision[2, 1] = prec_2x2[1, 0]
    precision[2, 2] = prec_2x2[1, 1]

    stats["_cross_signal_precision"] = precision

    return stats
