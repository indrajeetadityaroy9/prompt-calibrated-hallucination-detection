"""
Prompt Subspace Projection (PSP) — prompt-grounding signal.

PSP(t) = 1 - ||V_prompt @ h_centered|| / ||h_centered||, range [0,1], higher = riskier.
Magnitude-gated: PSP blends toward 0.5 when ||h_centered|| is small.
"""

import torch
from torch import Tensor

from ..numerics import EPS, effective_rank


class PromptSubspaceProjection:
    """PSP signal for AG-SAR."""

    def __init__(self):
        self._prompt_basis: Tensor = None  # type: ignore[assignment]
        self._prompt_center: Tensor = None  # type: ignore[assignment]
        self._tau: float = None  # type: ignore[assignment]

    def calibrate(self, prompt_hidden: Tensor) -> Tensor:
        """Compute prompt basis, center, magnitude tau. Returns singular values for SPT window."""
        prompt_hidden = prompt_hidden.float()
        self._prompt_center = prompt_hidden.mean(dim=0)
        centered = prompt_hidden - self._prompt_center
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        self._prompt_basis = Vh[:effective_rank(S)]
        norms = torch.norm(centered, dim=-1)
        self._tau = float(norms.median().item())
        return S

    def compute_psp(self, layer_hidden_states: dict[int, Tensor]) -> float:
        """Batched PSP over all layers."""
        V = self._prompt_basis.float()
        H = torch.stack([h.float().squeeze() for h in layer_hidden_states.values()])
        H_centered = H - self._prompt_center
        mags = torch.norm(H_centered, dim=-1)
        proj_norms = torch.norm(H_centered @ V.T, dim=-1)
        s_prompt = proj_norms / (mags + EPS)
        psp_raw = 1.0 - s_prompt
        gates = 1.0 - torch.exp(-mags.square() / (self._tau ** 2))
        psp = 0.5 + (psp_raw - 0.5) * gates
        return float(psp.mean().item())
