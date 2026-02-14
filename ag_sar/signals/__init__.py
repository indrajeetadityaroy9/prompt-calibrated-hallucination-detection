"""
Signal computation modules for DSG hallucination detection.

Three mechanistically distinct signal families:
- CUS (Context Utilization Score): Attention-based copying head tracking
- POS (Parametric Override Score): FFN directional decomposition
- DPS (Dual-Subspace Projection Score): Context vs reasoning geometry
"""

from .topk_jsd import CandidateJSDSignal
from .context_grounding import DualSubspaceGrounding
from .copying_heads import ContextUtilizationSignal, identify_copying_heads, compute_layer_affinity

__all__ = [
    "CandidateJSDSignal",
    "DualSubspaceGrounding",
    "ContextUtilizationSignal",
    "identify_copying_heads",
    "compute_layer_affinity",
]
