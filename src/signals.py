import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import LayerHiddenStates
from src.numerics import marchenko_pastur_edge, otsu


class SpectralAnalyzer:

    def __init__(self):
        self._prompt_eigvecs: Tensor | None = None

    def calibrate(self, prompt_layer_hidden: Tensor) -> None:
        H = prompt_layer_hidden.float()
        H_c = H - H.mean(dim=-2, keepdim=True)
        P, L, d = H.shape
        C = torch.einsum("tli,tki->lk", H_c, H_c) / (P * d)
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
        lam_plus = marchenko_pastur_edge(float(eigvals.median().item()), L / d)
        k = int((eigvals > lam_plus).sum().item())
        self._prompt_eigvecs = eigvecs[:, :k].T

    def compute(self, H_token: Tensor) -> tuple[float, float]:
        H_c = H_token.float() - H_token.float().mean(dim=0, keepdim=True)
        L, d = H_c.shape
        C = (H_c @ H_c.T) / d
        eigvals = torch.linalg.eigvalsh(C).flip(0)
        lam_plus = marchenko_pastur_edge(float(eigvals.median().item()), L / d)
        rho = max(0.0, (float(eigvals[0].item()) - lam_plus) / lam_plus)
        prompt_var = float((self._prompt_eigvecs @ C @ self._prompt_eigvecs.T).trace().item())
        return rho, prompt_var / float(C.trace().item())


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
    jsd = 0.5 * (torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m) + torch.xlogy(p_post, p_post) - torch.xlogy(p_post, m)).sum(dim=-1) / math.log(2)
    return float(jsd.mean().item())


def compute_ent(attn_tensor: Tensor, seq_len: int) -> float:
    a = attn_tensor.float()
    H = -torch.xlogy(a, a).sum(dim=-1) / math.log(seq_len)
    return float(1.0 - otsu(H.reshape(-1).cpu().numpy())[1])
