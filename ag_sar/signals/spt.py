import torch
from torch import Tensor

from ..numerics import EPS, tracy_widom_cdf


class SpectralPhaseTransition:

    def __init__(self, hidden_dim: int, window_size: int):
        self._d = hidden_dim
        self._size = window_size
        self._buffer: Tensor | None = None
        self._count = 0
        self._pos = 0

    def push(self, h: Tensor) -> None:
        h = h.detach().float().squeeze()
        if self._buffer is None:
            self._buffer = torch.empty(self._size, self._d, device=h.device)
        self._buffer[self._pos] = h
        self._pos = (self._pos + 1) % self._size
        self._count = min(self._count + 1, self._size)

    def compute_spt(self) -> tuple[float, float]:
        H = self._buffer[:self._count] if self._count < self._size else self._buffer
        H = H - H.mean(dim=0, keepdim=True)
        W = H.shape[0]
        S = torch.linalg.svdvals(H)
        eigs = (S ** 2) / W

        sigma2 = float(eigs.median().item())
        gamma = self._d / W
        sqrt_gamma = gamma ** 0.5

        mu_tw = sigma2 * (1.0 + sqrt_gamma) ** 2

        sigma_tw = sigma2 * (1.0 + sqrt_gamma) * (
            1.0 / (W ** 0.5) + 1.0 / (self._d ** 0.5)
        ) ** (1.0 / 3.0)

        lambda_1 = float(eigs[0].item())
        z_tw = (lambda_1 - mu_tw) / (sigma_tw + EPS)

        spt = 1.0 - tracy_widom_cdf(z_tw)

        lambda_2 = float(eigs[1].item())
        spectral_gap = lambda_2 / (lambda_1 + lambda_2 + EPS)

        return spt, spectral_gap

    @property
    def window_len(self) -> int:
        return self._count

    def seed(self, H: Tensor) -> None:
        H = H.detach().float()
        n = H.shape[0]
        if self._buffer is None:
            self._buffer = torch.empty(self._size, self._d, device=H.device)
        if n <= self._size:
            self._buffer[:n] = H
            self._count = n
            self._pos = n % self._size
        else:
            self._buffer[:] = H[-self._size:]
            self._count = self._size
            self._pos = 0

    def reset(self) -> None:
        self._buffer = None
        self._count = 0
        self._pos = 0
