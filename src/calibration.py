import numpy as np
import torch

from src.config import LayerHiddenStates
from src.fusion import CalibrationStats, calibrate_cusum
from src.numerics import effective_rank, information_flow_regularity
from src.signals import compute_ent, compute_mlp_jsd


def self_calibrate(*, spectral_analyzer, lm_head, final_norm, tail_per_layer: dict[int, dict], prefill_attentions: tuple, cal_tail_start: int) -> CalibrationStats:
    layer_keys = sorted(tail_per_layer.keys())
    n_tail = tail_per_layer[layer_keys[0]]["h_resid_mlp"].shape[0]

    H_all = torch.stack([tail_per_layer[li]["h_resid_mlp"] for li in layer_keys], dim=1).float()

    rho_vals, spf_vals = zip(*(spectral_analyzer.compute(H_all[t]) for t in range(n_tail)))

    diffs = H_all[:, 1:] - H_all[:, :-1]
    fi_all = diffs.norm(dim=-1) ** 2 / (H_all[:, :-1].norm(dim=-1) ** 2 + torch.finfo(H_all.dtype).eps)
    phi_vals = [information_flow_regularity(fi_all[t]) for t in range(n_tail)]

    h_final = tail_per_layer[max(layer_keys)]["h_resid_mlp"]
    with torch.no_grad():
        all_logits = lm_head(final_norm(h_final.unsqueeze(0).to(dtype=lm_head.weight.dtype))).squeeze(0)
    all_probs = torch.softmax(all_logits.float(), dim=-1)

    mlp_vals = []
    for t in range(n_tail):
        cand = torch.topk(all_logits[t], effective_rank(all_probs[t])).indices
        states = {li: LayerHiddenStates(h_resid_attn=tail_per_layer[li]["h_resid_attn"][t], h_resid_mlp=tail_per_layer[li]["h_resid_mlp"][t]) for li in layer_keys}
        mlp_vals.append(compute_mlp_jsd(states, cand, lm_head, final_norm))

    ent_vals = []
    for t in range(n_tail):
        t_abs = cal_tail_start + t
        ent_vals.append(compute_ent(torch.stack([a[0, :, t_abs, :t_abs + 1] for a in prefill_attentions]), t_abs + 1))

    return calibrate_cusum(np.column_stack([rho_vals, phi_vals, spf_vals, mlp_vals, ent_vals]))
