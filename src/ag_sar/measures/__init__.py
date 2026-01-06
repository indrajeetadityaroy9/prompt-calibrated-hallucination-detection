"""
AG-SAR Core Measures (Algorithms) - v8.0 SOTA.

Components:
- authority: Authority Flow, MLP Divergence, Gated Authority, Semantic Authority
- semantics: Semantic Dispersion (consistency over confidence)
- stability: Adaptive Gate (model-agnostic online normalization)
- entropy: Token entropy (baseline metric)
"""

# Core components
from .authority import (
    compute_authority_score,
    compute_mlp_divergence,
    compute_gated_authority,
    compute_semantic_authority,
)

from .entropy import (
    compute_token_entropy,
)

from .semantics import (
    compute_semantic_dispersion,
    compute_semantic_trust,
)

from .stability import (
    AdaptiveGate,
    AdaptiveGateBatch,
    compute_adaptive_stability_gate,
)

__all__ = [
    # Authority
    "compute_authority_score",
    "compute_mlp_divergence",
    "compute_gated_authority",
    "compute_semantic_authority",
    # Entropy
    "compute_token_entropy",
    # Semantic Dispersion
    "compute_semantic_dispersion",
    "compute_semantic_trust",
    # Adaptive Gate
    "AdaptiveGate",
    "AdaptiveGateBatch",
    "compute_adaptive_stability_gate",
]
