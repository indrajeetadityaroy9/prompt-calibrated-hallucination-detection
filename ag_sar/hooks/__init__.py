"""
Hook system for capturing hidden states during generation.

2-point capture per layer:
1. h_resid_attn: Residual after attention add (before post_attn_norm)
2. h_resid_mlp: Residual after MLP add (layer output)
"""

from .adapter import ModelAdapter
from .buffer import LayerHiddenStates, EphemeralHiddenBuffer
from .layer_hooks import LayerHooks
from .prefill_hooks import PrefillContextHook

__all__ = [
    "ModelAdapter",
    "LayerHiddenStates",
    "EphemeralHiddenBuffer",
    "LayerHooks",
    "PrefillContextHook",
]
