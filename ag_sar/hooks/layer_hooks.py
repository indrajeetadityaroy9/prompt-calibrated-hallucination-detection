"""3-point per-layer hook installation for hidden state capture."""

from typing import List
from torch import Tensor

from .adapter import ModelAdapter
from .buffer import EphemeralHiddenBuffer


class LayerHooks:
    """
    Install hooks to capture all 3 points for a single layer.

    Hook architecture for LLaMA:
    - pre_hook on post_attention_layernorm: capture input (h_resid_attn)
    - forward_hook on post_attention_layernorm: capture output (h_mlp_in)
    - forward_hook on decoder layer: capture output (h_resid_mlp)
    """

    def __init__(self, layer_idx: int, buffer: EphemeralHiddenBuffer, adapter: ModelAdapter):
        self.layer_idx = layer_idx
        self.buffer = buffer
        self.adapter = adapter
        self._h_resid_attn: Tensor = None  # type: ignore[assignment]
        self._h_mlp_in: Tensor = None  # type: ignore[assignment]
        self._handles: List = []

    def install(self, layer):
        """Install hooks on the layer."""
        post_attn_norm = self.adapter.get_post_attn_norm(layer)

        h1 = post_attn_norm.register_forward_pre_hook(
            self._capture_resid_attn
        )
        self._handles.append(h1)

        h2 = post_attn_norm.register_forward_hook(
            self._capture_mlp_in
        )
        self._handles.append(h2)

        h3 = layer.register_forward_hook(
            self._capture_resid_mlp_and_store
        )
        self._handles.append(h3)

    def _capture_resid_attn(self, module, args):
        """Pre-hook: input to post_attention_layernorm = h_resid_attn."""
        self._h_resid_attn = args[0]

    def _capture_mlp_in(self, module, args, output):
        """Forward hook: output of post_attention_layernorm = h_mlp_in."""
        self._h_mlp_in = output

    def _capture_resid_mlp_and_store(self, module, args, output):
        """Forward hook: layer output = h_resid_mlp. Store all 3 to buffer."""
        self.buffer.store(
            self.layer_idx,
            self._h_resid_attn,
            self._h_mlp_in,
            output[0],
        )

        self._h_resid_attn = None  # type: ignore[assignment]
        self._h_mlp_in = None  # type: ignore[assignment]

    def remove(self):
        """Remove all installed hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
