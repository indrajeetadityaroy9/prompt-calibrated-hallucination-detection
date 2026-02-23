"""
Semantic Trajectory Dynamics (STD) signal for AG-SAR.

Measures directional consistency of hidden state evolution across transformer layers.
Factual tokens show smooth, convergent trajectories; hallucinated tokens show
oscillatory, divergent trajectories.

Inspired by LSD (arXiv 2510.04933) but fully training-free:
- Uses h_mlp_in (post-RMSNorm) to avoid residual norm growth artifact.
- Projects onto context subspace for semantic anchoring.
- Two sub-signals: directional inconsistency + divergence asymmetry.
"""

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..numerics import EPS


class SemanticTrajectoryDynamics:
    """Context-projected layer trajectory dynamics signal."""

    def __init__(self, num_layers: int, context_basis: Optional[Tensor] = None):
        self.num_layers = num_layers
        self._context_basis = context_basis

    def set_context_basis(self, V_ctx: Tensor) -> None:
        """Store context subspace basis for trajectory projection."""
        self._context_basis = V_ctx

    def _project(self, h: Tensor) -> Tensor:
        """Project hidden state onto context subspace: V_ctx^T @ h."""
        V = self._context_basis.to(dtype=torch.float32, device=h.device)
        return V @ h  # [k, ] — k-dim projected representation

    def compute_std(self, layer_hidden_states: Dict[int, Tensor]) -> float:
        """
        STD from h_mlp_in states across layers.

        Two sub-signals (arithmetic mean):
        1. Context-projected directional inconsistency:
           (1 - mean cos(d_proj^l, d_proj^(l+1))) / 2
        2. Context-projected divergence asymmetry:
           sigmoid(late_velocity / early_velocity - 1)

        Args:
            layer_hidden_states: {layer_idx: h_mlp_in} (post-norm hidden states)

        Returns:
            STD score in [0, 1], higher = riskier (oscillatory/divergent)
        """
        if self._context_basis is None:
            return 0.5

        sorted_layers = sorted(layer_hidden_states.keys())
        if len(sorted_layers) < 3:
            return 0.5

        # Project all hidden states onto context subspace
        projected = []
        for idx in sorted_layers:
            h = layer_hidden_states[idx].float().squeeze()
            projected.append(self._project(h))

        # Compute displacement vectors between consecutive layers
        displacements = []
        for i in range(len(projected) - 1):
            d = projected[i + 1] - projected[i]
            displacements.append(d)

        if len(displacements) < 2:
            return 0.5

        # Sub-signal 1: Directional inconsistency
        # Mean cosine between consecutive displacement vectors
        cos_sims = []
        for i in range(len(displacements) - 1):
            d1 = displacements[i]
            d2 = displacements[i + 1]
            n1 = torch.norm(d1)
            n2 = torch.norm(d2)
            if n1 < EPS or n2 < EPS:
                cos_sims.append(1.0)  # Near-zero displacement = consistent
            else:
                cos = F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
                cos_sims.append(cos)

        mean_cos = sum(cos_sims) / len(cos_sims)
        inconsistency = (1.0 - mean_cos) / 2.0  # Map [-1,1] → [0,1]

        # Sub-signal 2: Divergence asymmetry (late vs early velocity)
        mid = len(displacements) // 2
        early_vels = [torch.norm(d).item() for d in displacements[:mid]]
        late_vels = [torch.norm(d).item() for d in displacements[mid:]]

        early_mean = sum(early_vels) / len(early_vels) if early_vels else EPS
        late_mean = sum(late_vels) / len(late_vels) if late_vels else EPS

        # sigmoid(ratio - 1): ratio > 1 (accelerating) → risk > 0.5
        import math
        ratio = late_mean / (early_mean + EPS)
        asymmetry = 1.0 / (1.0 + math.exp(-min(max(ratio - 1.0, -10.0), 10.0)))

        # Arithmetic mean of sub-signals
        return float(max(0.0, min(1.0, (inconsistency + asymmetry) / 2.0)))
