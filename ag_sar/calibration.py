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
    layer_keys = sorted(tail_per_layer.keys())
    n_layers = len(layer_keys)

    # --- PSP: batched across all tail positions × all layers ---
    # Build (n_tail, n_layers, d) tensor
    H_all = torch.stack(
        [tail_per_layer[li]["h_resid_mlp"] for li in layer_keys], dim=1
    ).float()
    d = H_all.shape[-1]

    V = psp_signal._prompt_basis.float()
    center = psp_signal._prompt_center
    tau = psp_signal._tau

    H_flat = H_all.reshape(-1, d)
    H_centered = H_flat - center
    mags = torch.norm(H_centered, dim=-1)
    proj_norms = torch.norm(H_centered @ V.T, dim=-1)
    s_prompt = proj_norms / (mags + EPS)
    psp_raw = 1.0 - s_prompt
    gates = 1.0 - torch.exp(-mags.square() / (tau ** 2))
    psp_flat = 0.5 + (psp_raw - 0.5) * gates
    psp_arr = psp_flat.reshape(n_tail, n_layers).mean(dim=1).cpu().numpy()
    stats["psp"] = {"sorted_vals": np.sort(psp_arr), "variance": float(np.var(psp_arr))}

    # --- MLP: batch logits computation, per-position JSD ---
    lm_head_dtype = lm_head.weight.dtype
    final_layer = max(layer_keys)
    h_final_tail = tail_per_layer[final_layer]["h_resid_mlp"]  # (n_tail, d)

    with torch.no_grad():
        all_logits = lm_head(
            final_norm(h_final_tail.unsqueeze(0).to(dtype=lm_head_dtype))
        ).squeeze(0)  # (n_tail, vocab)
    all_probs = torch.softmax(all_logits.float(), dim=-1)

    mlp_values = []
    for t in range(n_tail):
        k = max(2, effective_rank(all_probs[t]))
        cand = torch.topk(all_logits[t], k).indices
        layer_states = {
            li: LayerHiddenStates(
                h_resid_attn=tail_per_layer[li]["h_resid_attn"][t],
                h_resid_mlp=tail_per_layer[li]["h_resid_mlp"][t],
            )
            for li in layer_keys
        }
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
    diag_vars = np.array([
        stats["ent"]["variance"],
        stats["mlp"]["variance"],
        stats["psp"]["variance"],
        stats["spt"]["variance"],
        stats["spectral_gap"]["variance"],
    ])

    precision = np.diag(1.0 / np.maximum(diag_vars, EPS))

    psp_mlp_matrix = np.column_stack([mlp_arr, psp_arr])  # (n_tail, 2)
    # Ledoit-Wolf shrinkage: parameter-free optimal regularization (Ledoit & Wolf, 2004).
    cov_2x2, _ = ledoit_wolf(psp_mlp_matrix)
    prec_2x2 = np.linalg.inv(cov_2x2)
    # Embed into positions [1,1],[1,2],[2,1],[2,2] (MLP=idx1, PSP=idx2)
    precision[1:3, 1:3] = prec_2x2

    stats["_cross_signal_precision"] = precision

    return stats
