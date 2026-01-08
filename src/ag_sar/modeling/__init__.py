"""
AG-SAR Model Interaction Layer.

Architecture-agnostic attention extraction via forward hooks.
Supports GPT-2, Llama, Mistral, Qwen.
"""

from .hooks import ModelAdapter, AttentionCapture

__all__ = [
    "ModelAdapter",
    "AttentionCapture",
]
