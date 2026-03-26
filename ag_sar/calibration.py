import numpy as np
import torch

from .numerics import EPS, effective_rank, information_flow_regularity
from .hooks import LayerHiddenStates
from .signals.ent import compute_ent
from .aggregation.fusion import CalibrationStats, calibrate_cusum


def self_calibrate(
    *,
    spectral_analyzer,
    jsd_signal,
    lm_head,
    final_norm,
    tail_per_layer: dict[int, dict],
    prefill_attentions: tuple,
    cal_tail_start: int,
) -> CalibrationStats:
    layer_keys = sorted(tail_per_layer.keys())
    n_tail = tail_per_layer[layer_keys[0]]["h_resid_mlp"].shape[0]

    H_all = torch.stack(
        [tail_per_layer[li]["h_resid_mlp"] for li in layer_keys], dim=1,
    ).float()

    rho_vals, spf_vals, phi_vals = [], [], []
    for t in range(n_tail):
        rho, spf = spectral_analyzer.compute(H_all[t])
        rho_vals.append(rho)
        spf_vals.append(spf)

        fi = []
        for i in range(1, len(layer_keys)):
            prev, curr = H_all[t, i - 1], H_all[t, i]
            fi.append(float(((curr - prev).norm() ** 2 / (prev.norm() ** 2 + EPS)).item()))
        phi_vals.append(information_flow_regularity(torch.tensor(fi)))

    h_final = tail_per_layer[max(layer_keys)]["h_resid_mlp"]
    with torch.no_grad():
        all_logits = lm_head(
            final_norm(h_final.unsqueeze(0).to(dtype=lm_head.weight.dtype))
        ).squeeze(0)
    all_probs = torch.softmax(all_logits.float(), dim=-1)

    mlp_vals = []
    for t in range(n_tail):
        k = max(2, effective_rank(all_probs[t]))
        cand = torch.topk(all_logits[t], k).indices
        states = {
            li: LayerHiddenStates(
                h_resid_attn=tail_per_layer[li]["h_resid_attn"][t],
                h_resid_mlp=tail_per_layer[li]["h_resid_mlp"][t],
            )
            for li in layer_keys
        }
        mlp_vals.append(jsd_signal.compute_mlp_jsd(states, cand))

    ent_vals = []
    for t in range(n_tail):
        t_abs = cal_tail_start + t
        attn_t = torch.stack([a[0, :, t_abs, :t_abs + 1] for a in prefill_attentions])
        ent_vals.append(compute_ent(attn_t, t_abs + 1))

    signal_matrix = np.column_stack([rho_vals, phi_vals, spf_vals, mlp_vals, ent_vals])
    return calibrate_cusum(signal_matrix)
