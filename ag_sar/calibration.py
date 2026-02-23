"""
Shared calibration utilities for AG-SAR hallucination detection.

Provides self-calibrating prompt statistics pipeline used by Detector.
"""

import math
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

from .numerics import otsu_threshold, entropy_adaptive_nucleus


def adaptive_window(prompt_len: int) -> int:
    """Adaptive tail window: sqrt(prompt_len), clamped to [4, prompt_len//2].

    Floor of 4: minimum for stable order statistics (median of 4 values).
    For prompt_len >= 16: sqrt(16) = 4, so the floor is only active for
    very short prompts. For prompt_len >= 100: window >= 10.
    """
    window = int(math.ceil(math.sqrt(prompt_len)))
    return max(4, min(window, prompt_len // 2, prompt_len))


def self_calibrate(
    *,
    jsd_signal,
    dps_signal,
    std_signal=None,
    lm_head,
    final_norm,
    num_layers: int,
    prompt_len: int,
    tail_per_layer: Dict[int, Tensor],
    prefill_hidden: Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Self-calibrating prompt statistics. No hardcoded priors.

    Strategy per signal:
    - DPS: Compute from per-layer prefill tail WITH magnitude gate.
    - CUS: Direct mode — CUS in [0,1] with semantic meaning.
    - POS: From PrefillStatisticsHook JSD stats. MAD-based robust sigma.
    - DoLa: Compute from per-layer prefill tail via layer-contrast.
    - CGD: Compute from prefill tail hidden states.
    - STD: Compute from per-layer prefill tail via trajectory dynamics.

    On the canonical path (layer_subset="all"), all signals produce valid statistics.
    """
    window = adaptive_window(prompt_len)
    tail_start = max(0, prompt_len - window)

    stats = {}

    # Number of tail tokens from per-layer capture
    first_key = next(iter(tail_per_layer))
    n_tail = tail_per_layer[first_key].shape[0]

    # DPS: use per-layer hidden states with magnitude gate (matches generation path)
    dps_values = []
    for t in range(n_tail):
        h_dict = {layer_idx: tail_per_layer[layer_idx][t]
                  for layer_idx in tail_per_layer}
        dps_val = dps_signal.compute_dps(h_dict, num_layers)
        dps_values.append(dps_val)

    # DPS stats (PIT format: sorted reference values + variance)
    dps_arr = np.array(dps_values)
    stats["dps"] = {
        "sorted_vals": np.sort(dps_arr),
        "variance": float(np.var(dps_arr)),
    }

    # CUS: direct mode — value IS the probability
    stats["cus"] = {"mode": "direct"}

    # POS: PIT format from PrefillStatisticsHook JSD values
    pos_sigma = jsd_signal._prompt_jsd_sigma
    pos_sorted = np.sort(np.array(jsd_signal._prompt_jsd_values))
    stats["pos"] = {
        "sorted_vals": pos_sorted,
        "variance": float(pos_sigma ** 2),
    }

    # DoLa: compute from per-layer prefill tail hidden states.
    dola_values = []
    from .hooks import LayerHiddenStates
    for t in range(n_tail):
        layer_states = {}
        for layer_idx in sorted(tail_per_layer.keys()):
            h_t = tail_per_layer[layer_idx][t]
            layer_states[layer_idx] = LayerHiddenStates(
                h_resid_attn=h_t, h_mlp_in=h_t, h_resid_mlp=h_t,
            )
        final_layer = max(layer_states.keys())
        with torch.no_grad():
            h_final = layer_states[final_layer].h_resid_mlp
            if h_final.dim() == 1:
                h_final = h_final.unsqueeze(0).unsqueeze(0)
            elif h_final.dim() == 2:
                h_final = h_final.unsqueeze(0)
            h_final = h_final.to(dtype=lm_head.weight.dtype)
            h_norm = final_norm(h_final)
            logits_t = lm_head(h_norm).squeeze(0).squeeze(0)
        token_id = logits_t.argmax().item()
        probs_t = torch.softmax(logits_t.float(), dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs_t, descending=True)
        mass = entropy_adaptive_nucleus(probs_t)
        k = int((sorted_probs.cumsum(-1) < mass).sum().item()) + 1
        k = max(2, k)
        cand = sorted_indices[:k]
        cand_set = torch.unique(torch.cat([cand, torch.tensor([token_id], device=cand.device)]))
        dola_val = jsd_signal.compute_dola_score(layer_states, token_id, cand_set)
        dola_values.append(dola_val)

    # DoLa stats (PIT format)
    dola_arr = np.array(dola_values)
    stats["dola"] = {
        "sorted_vals": np.sort(dola_arr),
        "variance": float(np.var(dola_arr)),
    }

    # CGD: compute from prefill tail hidden states (single-layer is sufficient)
    cgd_values = []
    for t in range(tail_start, prompt_len):
        if t < prefill_hidden.shape[1]:
            h = prefill_hidden[:, t, :].squeeze(0)
            cgd_val = dps_signal.compute_grounding_direction(h)
            cgd_values.append(cgd_val)

    # CGD stats (PIT format)
    cgd_arr = np.array(cgd_values)
    stats["cgd"] = {
        "sorted_vals": np.sort(cgd_arr),
        "variance": float(np.var(cgd_arr)),
    }

    # STD: compute from per-layer prefill tail via trajectory dynamics.
    # Apply final_norm to h_resid_mlp to approximate h_mlp_in (scale-normalized).
    # This removes residual norm growth artifacts that would corrupt trajectory metrics.
    # The generation path uses actual h_mlp_in from 3-point hooks; calibration
    # approximates via final_norm (same RMSNorm family, removes scale).
    if std_signal is not None:
        std_values = []
        for t in range(n_tail):
            h_dict = {}
            for layer_idx in tail_per_layer:
                h_raw = tail_per_layer[layer_idx][t]
                if h_raw.dim() == 1:
                    h_raw = h_raw.unsqueeze(0)
                with torch.no_grad():
                    h_normed = final_norm(h_raw.to(dtype=lm_head.weight.dtype)).squeeze(0)
                h_dict[layer_idx] = h_normed
            std_val = std_signal.compute_std(h_dict)
            std_values.append(std_val)

        std_arr = np.array(std_values)
        stats["std"] = {
            "sorted_vals": np.sort(std_arr),
            "variance": float(np.var(std_arr)),
        }

    return stats


def select_informative_dps_layers(
    dps_signal,
    tail_per_layer: Dict[int, Tensor],
    num_layers: int,
    n_tail: int,
) -> List[int]:
    """
    Data-driven DPS layer selection via variance-based Otsu thresholding.

    For each hookable layer, computes single-layer DPS values across tail tokens,
    then selects layers with high DPS variance (most discriminative).
    """
    layer_variances = {}
    for layer_idx in sorted(tail_per_layer.keys()):
        dps_vals = []
        for t in range(n_tail):
            h = tail_per_layer[layer_idx][t]
            dps_val = dps_signal.dps_from_hidden(h)
            dps_vals.append(dps_val)

        if len(dps_vals) >= 2:
            layer_variances[layer_idx] = float(np.var(dps_vals))

    # Otsu threshold on variances to select high-variance (discriminative) layers
    var_array = np.array(list(layer_variances.values()))
    layer_indices = list(layer_variances.keys())
    threshold = otsu_threshold(var_array)

    selected = [idx for idx, var in zip(layer_indices, var_array) if var >= threshold]

    # Floor: at least ceil(sqrt(n_hookable_layers)) layers
    min_layers = max(1, int(math.ceil(math.sqrt(len(layer_variances)))))
    if len(selected) < min_layers:
        sorted_by_var = sorted(layer_variances.items(), key=lambda x: x[1], reverse=True)
        selected = [idx for idx, _ in sorted_by_var[:min_layers]]

    return sorted(selected)
