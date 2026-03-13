"""Ephemeral hidden state buffer for per-token signal computation."""

from dataclasses import dataclass
from torch import Tensor


@dataclass
class LayerHiddenStates:
    """Two capture points for one layer."""

    h_resid_attn: Tensor  # Residual after attention add (before post_attn_norm)
    h_resid_mlp: Tensor   # Residual after MLP add (layer output)


class EphemeralHiddenBuffer:
    """
    Per-step buffer capturing 2 points per layer.

    Ephemeral: cleared after each token's signals are computed.
    """

    def __init__(self):
        self.layer_states: dict[int, LayerHiddenStates] = {}

    def store(self, layer_idx: int, h_resid_attn: Tensor, h_resid_mlp: Tensor):
        """Store bfloat16 hidden states (last position only, detached)."""
        self.layer_states[layer_idx] = LayerHiddenStates(
            h_resid_attn=h_resid_attn[:, -1, :].detach().bfloat16(),
            h_resid_mlp=h_resid_mlp[:, -1, :].detach().bfloat16(),
        )

    def get_states(self) -> dict[int, LayerHiddenStates]:
        return self.layer_states

    def clear(self) -> None:
        self.layer_states.clear()
