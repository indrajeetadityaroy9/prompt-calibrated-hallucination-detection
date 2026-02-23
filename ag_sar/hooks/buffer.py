"""Ephemeral hidden state buffer for per-token signal computation."""

from dataclasses import dataclass
from typing import Dict
from torch import Tensor


@dataclass
class LayerHiddenStates:
    """All 3 capture points for one layer."""

    h_resid_attn: Tensor  # Residual after attention add (before post_attn_norm)
    h_mlp_in: Tensor  # MLP input (post_attn_norm output, normalized)
    h_resid_mlp: Tensor  # Residual after MLP add (layer output)


class EphemeralHiddenBuffer:
    """
    Per-step buffer capturing 3 points per layer.

    Ephemeral: cleared after each token's signals are computed.
    This prevents memory accumulation during generation.
    """

    def __init__(self):
        self.layer_states: Dict[int, LayerHiddenStates] = {}

    def store(
        self,
        layer_idx: int,
        h_resid_attn: Tensor,
        h_mlp_in: Tensor,
        h_resid_mlp: Tensor,
    ):
        """
        Store bfloat16 hidden states (last position only, detached not cloned).
        """
        self.layer_states[layer_idx] = LayerHiddenStates(
            h_resid_attn=h_resid_attn[:, -1, :].detach().bfloat16(),
            h_mlp_in=h_mlp_in[:, -1, :].detach().bfloat16(),
            h_resid_mlp=h_resid_mlp[:, -1, :].detach().bfloat16(),
        )

    def get_states(self) -> Dict[int, LayerHiddenStates]:
        """Get all captured layer states."""
        return self.layer_states

    def clear(self):
        """Clear all stored states to free memory."""
        self.layer_states.clear()
