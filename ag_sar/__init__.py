"""
AG-SAR: Zero-Shot Hallucination Detection via Decoupled Spectral Grounding.

DSG decomposes hallucination risk via 5 signals:
  CUS (attention) -> POS (FFN) -> DPS (geometry) -> DoLa (factuality) -> CGD (activation steering)
fused via entropy-gated aggregation with prompt-anchored calibration.
"""

__version__ = "1.0.0"

from .config import DSGConfig, DSGTokenSignals, DetectionResult, NormMode, SignalMetadata, SIGNAL_REGISTRY
from .hooks import EphemeralHiddenBuffer, LayerHooks, PrefillContextHook
from .numerics import safe_softmax, safe_jsd, EPS
from .aggregation.span_merger import RiskySpan

__all__ = [
    # Config
    "DSGConfig",
    "DSGTokenSignals",
    "DetectionResult",
    "NormMode",
    "SignalMetadata",
    "SIGNAL_REGISTRY",
    # Hooks
    "EphemeralHiddenBuffer",
    "LayerHooks",
    "PrefillContextHook",
    # Numerics
    "safe_softmax",
    "safe_jsd",
    "EPS",
    # Aggregation
    "RiskySpan",
]
