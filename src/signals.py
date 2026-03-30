import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import LayerHiddenStates
from src.numerics import EPS, marchenko_pastur_edge, otsu_coefficient


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
        sigma2 = float(eigvals.median().item())
        lam_plus = marchenko_pastur_edge(sigma2, L / d)
        k = max(1, int((eigvals > lam_plus).sum().item()))
        self._prompt_eigvecs = eigvecs[:, :k].T

    def compute(self, H_token: Tensor) -> tuple[float, float]:
        H = H_token.float()
        H_c = H - H.mean(dim=0, keepdim=True)
        L, d = H_c.shape
        C = (H_c @ H_c.T) / d
        eigvals = torch.linalg.eigvalsh(C).flip(0)
        sigma2 = float(eigvals.median().item())
        lam_plus = marchenko_pastur_edge(sigma2, L / d)
        lam1 = float(eigvals[0].item())
        rho = max(0.0, (lam1 - lam_plus) / (lam_plus + EPS))
        total_var = float(C.trace().item())
        prompt_var = float((self._prompt_eigvecs @ C @ self._prompt_eigvecs.T).trace().item())
        spf = prompt_var / (total_var + EPS)
        return rho, spf


def compute_mlp_jsd(layer_states: dict[int, LayerHiddenStates], candidate_set: Tensor, lm_head: nn.Linear, final_norm: nn.Module) -> float:
    dtype = lm_head.weight.dtype

    attn_stack = torch.stack([s.h_resid_attn for s in layer_states.values()]).to(dtype=dtype)
    mlp_stack = torch.stack([s.h_resid_mlp for s in layer_states.values()]).to(dtype=dtype)

    with torch.no_grad():
        pre_norm = final_norm(attn_stack)
        post_norm = final_norm(mlp_stack)

    w_subset = lm_head.weight[candidate_set]
    z_pre = F.linear(pre_norm, w_subset).float()
    z_post = F.linear(post_norm, w_subset).float()

    p_pre = F.softmax(z_pre, dim=-1)
    p_post = F.softmax(z_post, dim=-1)

    m = 0.5 * (p_pre + p_post)
    kl_pm = (torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m)).sum(dim=-1)
    kl_qm = (torch.xlogy(p_post, p_post) - torch.xlogy(p_post, m)).sum(dim=-1)
    jsd_bits = (0.5 * kl_pm + 0.5 * kl_qm) / math.log(2)

    return float(jsd_bits.mean().item())


def compute_ent(attn_tensor: Tensor, seq_len: int) -> float:
    log2_n = math.log2(seq_len)
    a = attn_tensor.float().clamp(min=EPS)
    H = -(a * a.log2()).sum(dim=-1) / log2_n
    return float(1.0 - otsu_coefficient(H.reshape(-1).cpu().numpy()))
