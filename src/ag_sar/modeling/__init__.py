"""
AG-SAR Model Interaction Layer.

Provides unified hook-based attention extraction across architectures:
- GPT-2: Fused c_attn hook
- Llama/Mistral/Qwen: Monkey-patched forward for post-RoPE capture

H100 Optimizations:
- load_model_h100(): Load with Flash Attention 2, bfloat16, balanced device_map
- Multi-GPU support with native device tensor handling
"""

from .hooks import ModelAdapter, AttentionCapture, load_model_h100
from .predictor import JepaPredictor
from .online_predictor import OnlineJepaPredictor

__all__ = [
    "ModelAdapter",
    "AttentionCapture",
    "load_model_h100",
    "JepaPredictor",
    "OnlineJepaPredictor",
]
