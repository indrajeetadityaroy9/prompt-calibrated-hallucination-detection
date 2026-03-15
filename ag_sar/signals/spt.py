"""SPT — Spectral Phase-Transition score via Tracy-Widom calibrated BBP detection.

Sliding-window covariance spectrum of midpoint-layer hidden states.
SPT(t) = 1 - F_{TW,1}((lambda_1 - mu_TW) / sigma_TW)
where mu_TW = sigma^2 * (1 + sqrt(gamma))^2  is the MP upper edge,
      sigma_TW = sigma^2 * (1 + sqrt(gamma)) * (1/sqrt(W) + 1/sqrt(d))^{1/3}
      is the Tracy-Widom finite-sample scaling rate (Johnstone, 2001).

Also returns the spectral gap ratio lambda_2 / (lambda_1 + lambda_2),
bounded in [0, 0.5], capturing directional coherence of the signal structure.
"""

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
        """Compute TW-calibrated SPT and spectral gap ratio.

        Returns:
            spt: 1 - F_{TW,1}(z_TW).  Range (0, 1), higher = riskier.
            spectral_gap: lambda_2 / (lambda_1 + lambda_2).  Near 0 = coherent, near 0.5 = degenerate.
        """
        H = self._buffer[:self._count] if self._count < self._size else self._buffer
        H = H - H.mean(dim=0, keepdim=True)
        W = H.shape[0]
        S = torch.linalg.svdvals(H)
        eigs = (S ** 2) / W

        sigma2 = float(eigs.median().item())
        gamma = self._d / W
        sqrt_gamma = gamma ** 0.5

        # Marchenko-Pastur upper edge
        mu_tw = sigma2 * (1.0 + sqrt_gamma) ** 2

        # Tracy-Widom finite-sample scaling (Johnstone, 2001)
        sigma_tw = sigma2 * (1.0 + sqrt_gamma) * (
            1.0 / (W ** 0.5) + 1.0 / (self._d ** 0.5)
        ) ** (1.0 / 3.0)

        # Standardized TW statistic
        lambda_1 = float(eigs[0].item())
        z_tw = (lambda_1 - mu_tw) / (sigma_tw + EPS)

        # SPT = 1 - F_{TW,1}(z): high z → strong signal → low risk
        spt = 1.0 - tracy_widom_cdf(z_tw)

        # Spectral gap ratio: lambda_2 / (lambda_1 + lambda_2)
        lambda_2 = float(eigs[1].item())
        spectral_gap = lambda_2 / (lambda_1 + lambda_2 + EPS)

        return spt, spectral_gap

    @property
    def window_len(self) -> int:
        """Current number of states in the sliding window."""
        return self._count

    def seed(self, H: Tensor) -> None:
        """Bulk-load the window from a contiguous (n, d) tensor. Call after reset."""
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
