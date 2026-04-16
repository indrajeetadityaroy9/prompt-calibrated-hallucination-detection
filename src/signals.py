import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import LayerHiddenStates
from src.numerics import marchenko_pastur_edge, otsu_coefficient

_LOG2 = math.log(2)


class SpectralAnalyzer:

    def __init__(self):
        self._prompt_eigvecs: Tensor | None = None

    def calibrate(self, prompt_layer_hidden: Tensor) -> None:
        H_mean = prompt_layer_hidden.mean(dim=0).float()
        H_c = H_mean - H_mean.mean(dim=0, keepdim=True)
        L, d = H_c.shape
        C = (H_c @ H_c.T) / d
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
        lam_plus = marchenko_pastur_edge(float(eigvals.median().item()), L / d)
        self._prompt_eigvecs = eigvecs[:, :max(1, int((eigvals > lam_plus).sum().item()))].T

    def compute(self, H_token: Tensor) -> tuple[float, float]:
        H_c = H_token.float()
        H_c = H_c - H_c.mean(dim=0, keepdim=True)
        L, d = H_c.shape
        C = (H_c @ H_c.T) / d
        eigvals = torch.linalg.eigvalsh(C).flip(0)
        eps = torch.finfo(eigvals.dtype).eps
        lam_plus = marchenko_pastur_edge(float(eigvals.median().item()), L / d)
        rho = max(0.0, (float(eigvals[0].item()) - lam_plus) / (lam_plus + eps))
        total_var = float(C.trace().item())
        prompt_var = float((self._prompt_eigvecs @ C @ self._prompt_eigvecs.T).trace().item())
        return rho, prompt_var / (total_var + eps)


def compute_mlp_jsd(layer_states: dict[int, LayerHiddenStates], candidate_set: Tensor, lm_head: nn.Linear, final_norm: nn.Module) -> float:
    dtype = lm_head.weight.dtype
    attn_stack = torch.stack([s.h_resid_attn for s in layer_states.values()]).to(dtype=dtype)
    mlp_stack = torch.stack([s.h_resid_mlp for s in layer_states.values()]).to(dtype=dtype)

    with torch.no_grad():
        pre_norm, post_norm = final_norm(attn_stack), final_norm(mlp_stack)

    w_subset = lm_head.weight[candidate_set]
    p_pre = F.softmax(F.linear(pre_norm, w_subset).float(), dim=-1)
    p_post = F.softmax(F.linear(post_norm, w_subset).float(), dim=-1)

    m = 0.5 * (p_pre + p_post)
    jsd = 0.5 * (torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m) + torch.xlogy(p_post, p_post) - torch.xlogy(p_post, m)).sum(dim=-1) / _LOG2
    return float(jsd.mean().item())


def compute_ent(attn_tensor: Tensor, seq_len: int) -> float:
    a = attn_tensor.float().clamp(min=torch.finfo(torch.float32).tiny)
    H = -(a * a.log2()).sum(dim=-1) / math.log2(seq_len)
    return float(1.0 - otsu_coefficient(H.reshape(-1).cpu().numpy()))
