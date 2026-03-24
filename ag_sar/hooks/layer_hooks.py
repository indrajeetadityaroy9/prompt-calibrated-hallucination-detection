from torch import Tensor

from .adapter import ModelAdapter
from .buffer import EphemeralHiddenBuffer


class LayerHooks:

    def __init__(self, layer_idx: int, buffer: EphemeralHiddenBuffer, adapter: ModelAdapter):
        self.layer_idx = layer_idx
        self.buffer = buffer
        self.adapter = adapter
        self._h_resid_attn: Tensor | None = None
        self._handles: list = []

    def install(self, layer):
        post_attn_norm = self.adapter.get_post_attn_norm(layer)

        h1 = post_attn_norm.register_forward_pre_hook(self._capture_resid_attn)
        self._handles.append(h1)

        h2 = layer.register_forward_hook(self._capture_resid_mlp_and_store)
        self._handles.append(h2)

    def _capture_resid_attn(self, module, args):
        self._h_resid_attn = args[0]

    def _capture_resid_mlp_and_store(self, module, args, output):
        self.buffer.store(self.layer_idx, self._h_resid_attn, output[0])
        self._h_resid_attn = None  # type: ignore[assignment]

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
