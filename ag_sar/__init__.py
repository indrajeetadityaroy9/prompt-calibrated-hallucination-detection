"""
AG-SAR: Zero-Shot Hallucination Detection for LLaMA 3.1

A zero-shot hallucination detector that computes internal signals during generation
and aggregates them via Noisy-OR for polarity-stable risk scoring.

Deployment mode: Single-pass during generation (one forward per decoded token with KV-cache)
Evaluation mode: Forced decoding for labeled datasets (stepwise with ground-truth tokens)
"""

__version__ = "0.2.0"

from .config import DetectorConfig, SpanRisk, DetectionResult
from .hooks import EphemeralHiddenBuffer, LayerHooks, PrefillContextHook, HookManager
from .numerics import (
    safe_softmax,
    safe_jsd,
    max_cosine_similarity,
)

__all__ = [
    # Config
    "DetectorConfig",
    "SpanRisk",
    "DetectionResult",
    # Hooks
    "EphemeralHiddenBuffer",
    "LayerHooks",
    "PrefillContextHook",
    "HookManager",
    # Numerics
    "safe_softmax",
    "safe_jsd",
    "max_cosine_similarity",
]
