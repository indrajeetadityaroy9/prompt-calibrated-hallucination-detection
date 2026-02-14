"""
AG-SAR: Zero-Shot Hallucination Detection via Decoupled Spectral Grounding.

DSG decomposes hallucination risk along the causal chain:
  Context Utilization (attention) → Parametric Override (FFN) → Dual-Subspace Projection (representation)
fused via prompt-anchored Noisy-OR for polarity-stable risk scoring.
"""

__version__ = "1.0.0"

from .config import DSGConfig, DSGTokenSignals, SpanRisk, DetectionResult
from .hooks import EphemeralHiddenBuffer, LayerHooks, PrefillContextHook
from .numerics import safe_softmax, safe_jsd, max_cosine_similarity, EPS

__all__ = [
    # Config
    "DSGConfig",
    "DSGTokenSignals",
    "SpanRisk",
    "DetectionResult",
    # Hooks
    "EphemeralHiddenBuffer",
    "LayerHooks",
    "PrefillContextHook",
    # Numerics
    "safe_softmax",
    "safe_jsd",
    "max_cosine_similarity",
    "EPS",
]
