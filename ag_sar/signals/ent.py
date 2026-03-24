import math

import numpy as np
from torch import Tensor

from ..numerics import EPS, otsu_coefficient


def compute_ent(attn_tensor: Tensor, seq_len: int) -> float:
    log2_n = math.log2(seq_len)
    a = attn_tensor.float().clamp(min=EPS)
    H = -(a * a.log2()).sum(dim=-1) / log2_n
    return float(np.clip(1.0 - otsu_coefficient(H.reshape(-1).cpu().numpy()), 0.0, 1.0))
