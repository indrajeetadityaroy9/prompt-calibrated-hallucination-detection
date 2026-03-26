import torch
from torch import Tensor

from ..numerics import EPS, marchenko_pastur_edge


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
        k = int((eigvals > lam_plus).sum().item())
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
