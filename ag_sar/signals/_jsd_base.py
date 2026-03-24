import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates


class CandidateJSDSignal:

    def __init__(self, lm_head: nn.Linear, final_norm: nn.Module):
        self.lm_head = lm_head
        self.final_norm = final_norm

    def compute_mlp_jsd(
        self,
        layer_states: dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> float:
        dtype = self.lm_head.weight.dtype

        attn_stack = torch.stack([s.h_resid_attn for s in layer_states.values()]).to(dtype=dtype)
        mlp_stack = torch.stack([s.h_resid_mlp for s in layer_states.values()]).to(dtype=dtype)

        with torch.no_grad():
            pre_norm = self.final_norm(attn_stack)
            post_norm = self.final_norm(mlp_stack)

        w_subset = self.lm_head.weight[candidate_set]
        z_pre = F.linear(pre_norm, w_subset).float()
        z_post = F.linear(post_norm, w_subset).float()

        p_pre = F.softmax(z_pre, dim=-1)
        p_post = F.softmax(z_post, dim=-1)

        m = 0.5 * (p_pre + p_post)
        kl_pm = (torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m)).sum(dim=-1)
        kl_qm = (torch.xlogy(p_post, p_post) - torch.xlogy(p_post, m)).sum(dim=-1)
        jsd_bits = (0.5 * kl_pm + 0.5 * kl_qm) / math.log(2)

        return float(jsd_bits.mean().item())
