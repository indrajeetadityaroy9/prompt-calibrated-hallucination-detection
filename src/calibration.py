import numpy as np
import torch

from src.config import LayerHiddenStates
from src.fusion import CalibrationStats, calibrate_cusum
from src.numerics import effective_rank, information_flow_regularity
from src.signals import compute_ent, compute_mlp_jsd


def self_calibrate(*, spectral_analyzer, lm_head, final_norm, prompt_per_layer: dict[int, dict], prefill_attentions: tuple) -> tuple[CalibrationStats, np.ndarray]:
    layer_keys = sorted(prompt_per_layer.keys())
    n_prompt = prompt_per_layer[layer_keys[0]]["h_resid_mlp"].shape[0]
    positions = range(1, n_prompt)

    H_all = torch.stack([prompt_per_layer[li]["h_resid_mlp"] for li in layer_keys], dim=1).float()

    rho_vals, spf_vals = zip(*(spectral_analyzer.compute(H_all[t]) for t in positions))

    diffs = H_all[:, 1:] - H_all[:, :-1]
    fi_all = diffs.norm(dim=-1) ** 2 / H_all[:, :-1].norm(dim=-1) ** 2
    phi_vals = [information_flow_regularity(fi_all[t]) for t in positions]

    h_final = prompt_per_layer[max(layer_keys)]["h_resid_mlp"]
    with torch.no_grad():
        all_logits = lm_head(final_norm(h_final.unsqueeze(0).to(dtype=lm_head.weight.dtype))).squeeze(0)
    all_probs = torch.softmax(all_logits.float(), dim=-1)

    mlp_vals = []
    for t in positions:
        cand = torch.topk(all_logits[t], effective_rank(all_probs[t])).indices
        states = {li: LayerHiddenStates(h_resid_attn=prompt_per_layer[li]["h_resid_attn"][t], h_resid_mlp=prompt_per_layer[li]["h_resid_mlp"][t]) for li in layer_keys}
        mlp_vals.append(compute_mlp_jsd(states, cand, lm_head, final_norm))

    ent_vals = [compute_ent(torch.stack([a[0, :, t, :t + 1] for a in prefill_attentions]), t + 1) for t in positions]

    signal_matrix = np.column_stack([rho_vals, phi_vals, spf_vals, mlp_vals, ent_vals])
    return calibrate_cusum(signal_matrix), signal_matrix
