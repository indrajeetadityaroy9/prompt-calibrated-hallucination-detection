"""
Shared calibration utilities for DSG hallucination detection.

Provides self-calibrating prompt statistics pipeline used by DSGDetector.
"""

import math
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .config import DSGConfig, SIGNAL_REGISTRY
from .numerics import mad_sigma


def get_layer_indices(config: DSGConfig, num_layers: int) -> List[int]:
    """Get hookable layer indices from config."""
    if config.layer_subset == "all":
        return list(range(num_layers))
    elif config.layer_subset == "last_third":
        start = num_layers - (num_layers // 3)
        return list(range(start, num_layers))
    elif config.layer_subset == "last_quarter":
        start = num_layers - (num_layers // 4)
        return list(range(start, num_layers))
    elif isinstance(config.layer_subset, list):
        return config.layer_subset
    return list(range(num_layers))


def build_input(
    tokenizer: Any,
    context: str,
    question: str,
    device: torch.device = None,
) -> Tuple[Tensor, int, int, int]:
    """
    Build input_ids from context and question.

    Returns (input_ids, context_start, context_end, prompt_len).
    """
    prefix = tokenizer.encode("Context: ", add_special_tokens=False)
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False) if context else []
    sep = tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    suffix = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

    bos = tokenizer.bos_token_id
    bos_len = 1 if bos is not None else 0

    context_start = bos_len + len(prefix)
    context_end = context_start + len(ctx_tokens)

    tokens = (([bos] if bos is not None else [])
              + prefix + ctx_tokens + sep + q_tokens + suffix)

    input_ids = torch.tensor([tokens], dtype=torch.long)
    if device is not None:
        input_ids = input_ids.to(device)
    return input_ids, context_start, context_end, len(tokens)


def adaptive_window(prompt_len: int) -> int:
    """Adaptive tail window: sqrt(prompt_len), clamped to [16, prompt_len//2]."""
    window = int(math.ceil(math.sqrt(prompt_len)))
    return max(16, min(window, prompt_len // 2, prompt_len))


def self_calibrate(
    *,
    jsd_signal,
    dps_signal,
    lm_head,
    final_norm,
    num_layers: int,
    prompt_len: int,
    tail_per_layer: Dict[int, Tensor] = None,
    prefill_hidden: Tensor = None,
) -> Dict[str, Dict[str, float]]:
    """
    Self-calibrating prompt statistics. No hardcoded priors.

    Strategy per signal:
    - DPS: Compute from per-layer prefill tail WITH magnitude gate.
    - CUS: Direct mode — CUS in [0,1] with semantic meaning.
    - POS: From PrefillStatisticsHook JSD stats. MAD-based robust sigma.
    - DoLa: Compute from per-layer prefill tail via layer-contrast.
    - CGD: Compute from prefill tail hidden states.

    All fallbacks use SIGNAL_REGISTRY — no model-specific constants.
    """
    window = adaptive_window(prompt_len)
    tail_start = max(0, prompt_len - window)

    stats = {}

    # Number of tail tokens from per-layer capture
    n_tail = 0
    if tail_per_layer:
        first_key = next(iter(tail_per_layer))
        n_tail = tail_per_layer[first_key].shape[0]

    # DPS: use per-layer hidden states with magnitude gate (matches generation path)
    dps_values = []
    if tail_per_layer and n_tail > 0 and dps_signal._context_basis is not None:
        for t in range(n_tail):
            h_dict = {layer_idx: tail_per_layer[layer_idx][t]
                      for layer_idx in tail_per_layer}
            dps_val = dps_signal.compute_dps(h_dict, num_layers)
            dps_values.append(dps_val)

    # DPS stats
    if dps_values and len(dps_values) >= 2:
        dps_arr = np.array(dps_values)
        sigma_robust = mad_sigma(dps_arr)
        sigma_std = float(np.std(dps_arr))
        sigma = max(sigma_robust, sigma_std)
        sigma_floor = SIGNAL_REGISTRY["dps"].sigma_floor(sigma_robust)
        stats["dps"] = {
            "mu": float(np.mean(dps_arr)),
            "sigma": max(sigma, sigma_floor),
        }
    else:
        stats["dps"] = SIGNAL_REGISTRY["dps"].fallback_stats()

    # CUS: direct mode — value IS the probability
    stats["cus"] = {"mode": "direct"}

    # POS: use JSD stats from PrefillStatisticsHook
    if hasattr(jsd_signal, '_prompt_jsd_mu') and jsd_signal._prompt_jsd_mu is not None:
        pos_mu = jsd_signal._prompt_jsd_mu
        pos_sigma = jsd_signal._prompt_jsd_sigma if jsd_signal._prompt_jsd_sigma else 0.1
        sigma_floor = SIGNAL_REGISTRY["pos"].sigma_floor(pos_sigma)
        stats["pos"] = {"mu": pos_mu, "sigma": max(pos_sigma, sigma_floor)}
    else:
        stats["pos"] = SIGNAL_REGISTRY["pos"].fallback_stats()

    # DoLa: compute from per-layer prefill tail hidden states.
    dola_values = []
    if tail_per_layer and n_tail > 0 and len(tail_per_layer) >= 2:
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
            topk = min(64, logits_t.shape[0])
            cand = torch.topk(logits_t, topk).indices
            cand_set = torch.unique(torch.cat([cand, torch.tensor([token_id], device=cand.device)]))
            dola_val = jsd_signal.compute_dola_score(layer_states, token_id, cand_set)
            dola_values.append(dola_val)

    if dola_values and len(dola_values) >= 2:
        dola_arr = np.array(dola_values)
        dola_sigma = max(mad_sigma(dola_arr), float(np.std(dola_arr)))
        sigma_floor = SIGNAL_REGISTRY["dola"].sigma_floor(dola_sigma)
        stats["dola"] = {
            "mu": float(np.mean(dola_arr)),
            "sigma": max(dola_sigma, sigma_floor),
        }
    else:
        stats["dola"] = SIGNAL_REGISTRY["dola"].fallback_stats()

    # CGD: compute from prefill tail hidden states (single-layer is sufficient)
    cgd_values = []
    if prefill_hidden is not None and dps_signal._prompt_center is not None:
        for t in range(tail_start, prompt_len):
            if t < prefill_hidden.shape[1]:
                h = prefill_hidden[:, t, :].squeeze(0)
                cgd_val = dps_signal.compute_grounding_direction(h)
                cgd_values.append(cgd_val)

    if cgd_values and len(cgd_values) >= 2:
        cgd_arr = np.array(cgd_values)
        cgd_sigma = max(mad_sigma(cgd_arr), float(np.std(cgd_arr)))
        sigma_floor = SIGNAL_REGISTRY["cgd"].sigma_floor(cgd_sigma)
        stats["cgd"] = {"mu": float(np.mean(cgd_arr)), "sigma": max(cgd_sigma, sigma_floor)}
    else:
        stats["cgd"] = SIGNAL_REGISTRY["cgd"].fallback_stats()

    return stats
