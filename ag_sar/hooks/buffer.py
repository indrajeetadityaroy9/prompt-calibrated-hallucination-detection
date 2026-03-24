from dataclasses import dataclass
from torch import Tensor


@dataclass
class LayerHiddenStates:
    h_resid_attn: Tensor
    h_resid_mlp: Tensor


class EphemeralHiddenBuffer:

    def __init__(self):
        self.layer_states: dict[int, LayerHiddenStates] = {}

    def store(self, layer_idx: int, h_resid_attn: Tensor, h_resid_mlp: Tensor):
        self.layer_states[layer_idx] = LayerHiddenStates(
            h_resid_attn=h_resid_attn[:, -1, :].detach().bfloat16(),
            h_resid_mlp=h_resid_mlp[:, -1, :].detach().bfloat16(),
        )

    def get_states(self) -> dict[int, LayerHiddenStates]:
        return self.layer_states

    def clear(self) -> None:
        self.layer_states.clear()
