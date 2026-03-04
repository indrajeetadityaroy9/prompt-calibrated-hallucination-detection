"""SPT — Spectral Phase-Transition score via Marchenko-Pastur BBP threshold.

Sliding-window covariance spectrum of midpoint-layer hidden states.
SPT(t) = 1 - clamp((lambda_1 - lambda_+) / lambda_+, 0, 1)
where lambda_+ = sigma^2 * (1 + sqrt(d/W))^2 is the MP upper edge.
"""

from collections import deque

import torch
from torch import Tensor

from ..numerics import EPS


class SpectralPhaseTransition:

    def __init__(self, hidden_dim: int, window_size: int):
        self._window = deque(maxlen=window_size)
        self._d = hidden_dim

    def push(self, h: Tensor) -> None:
        self._window.append(h.detach().float().squeeze())

    def compute_spt(self) -> float:
        H = torch.stack(list(self._window))
        H = H - H.mean(dim=0, keepdim=True)
        W = H.shape[0]
        S = torch.linalg.svdvals(H)
        eigs = (S ** 2) / W
        sigma2 = float(eigs.median().item())
        gamma = self._d / W
        lambda_plus = sigma2 * (1.0 + gamma ** 0.5) ** 2
        excess = (float(eigs[0].item()) - lambda_plus) / (lambda_plus + EPS)
        return 1.0 - min(max(excess, 0.0), 1.0)

    def reset(self):
        self._window.clear()
