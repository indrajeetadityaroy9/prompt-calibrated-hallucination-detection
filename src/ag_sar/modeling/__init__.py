"""
AG-SAR Model Interaction Layer.

This module provides architecture-agnostic attention extraction via forward hooks.
It enables AG-SAR to capture Q, K, V states and attention weights from various
transformer architectures without modifying model code.

Supported Architectures:
    - GPT-2: Fused c_attn hook for QKV extraction (no RoPE)
    - Llama-3/3.1: Monkey-patched forward for post-RoPE Q/K capture + GQA
    - Mistral/Mixtral: Similar to Llama with MoE-compatible hooks
    - Qwen/Qwen2: Qwen-specific attention pattern handling

Critical Implementation Notes:
    1. Llama-style models apply RoPE AFTER projection. We capture Q/K after
       RoPE application to ensure position information is embedded.

    2. Multi-GPU models (via device_map): Tensors are stored on their native
       device to avoid expensive NVLink transfers. Aggregation happens only
       when computing final scores.

    3. transformers version constraint: AG-SAR requires transformers >= 4.40.0,
       < 4.45.0. Version 4.45+ introduced breaking changes to attention hooks.

Pipeline Position:
    ModelAdapter is instantiated by the AGSAR engine at initialization.
    It registers hooks that persist across forward passes, capturing attention
    data into AttentionCapture containers.

Public API:
    - ModelAdapter: Main class for hook registration and attention extraction
    - AttentionCapture: Dataclass container for captured attention tensors
    - load_model_h100: Convenience function for H100-optimized model loading
"""

from .hooks import ModelAdapter, AttentionCapture, load_model_h100

__all__ = [
    "ModelAdapter",
    "AttentionCapture",
    "load_model_h100",
]
