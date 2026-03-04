"""Prefill-phase hook for context capture."""

from typing import List
from torch import Tensor


class PrefillContextHook:
    """Temporary hook installed during prefill to capture context embeddings."""

    def __init__(self, context_mask: Tensor, buffer_ref: List[Tensor]):
        self.context_mask = context_mask
        self.buffer_ref = buffer_ref
        self._handle = None

    def install(self, layer):
        self._handle = layer.register_forward_hook(self._capture_context)

    def _capture_context(self, module, args, output):
        context_hidden = output[0][0, self.context_mask, :]
        self.buffer_ref.append(context_hidden.detach().bfloat16())

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
