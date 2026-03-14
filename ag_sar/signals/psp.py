"""
Prompt Subspace Projection (PSP) — prompt-grounding signal.

PSP(t) = 1 - ||V_prompt @ h_centered|| / ||h_centered||, range [0,1], higher = riskier.
Magnitude-gated: PSP blends toward 0.5 when ||h_centered|| is small.
"""

import math

import torch
import numpy as np
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

    def psp_from_hidden(self, h: Tensor) -> float:
        """PSP = gate * (1 - s_prompt) + (1-gate) * 0.5."""
        V = self._prompt_basis.float()
        h = h.float().squeeze()
        h_centered = h - self._prompt_center
        mag = torch.norm(h_centered).item()

        s_prompt = torch.norm(V @ h_centered).item() / (mag + EPS)
        psp_raw = 1.0 - s_prompt
        gate = 1.0 - math.exp(-(mag ** 2) / (self._tau ** 2))
        return 0.5 + (psp_raw - 0.5) * gate

    def compute_psp(self, layer_hidden_states: dict[int, Tensor]) -> float:
        """Mean PSP over all layers."""
        return float(np.mean([self.psp_from_hidden(h) for h in layer_hidden_states.values()]))
