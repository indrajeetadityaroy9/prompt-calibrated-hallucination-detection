"""
AG-SAR Core Measures (Algorithms) - v12.2 Model-Agnostic.

SOTA v12.2 components:
- authority: Authority Flow, MLP Divergence, Gated Authority, Semantic Authority
- semantics: Semantic Dispersion (consistency over confidence)
- stability: Layer Drift, Adaptive Gate (model-agnostic online normalization)
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
    # Layer Drift (v11.0 - experimental)
    compute_layer_drift,
    compute_top_k_drift,
    compute_rank_drift,
    apply_drift_penalty,
    # Adaptive Gate (v12.2 - model-agnostic)
    AdaptiveGate,
    AdaptiveGateBatch,
    compute_adaptive_stability_gate,
)

from .symbolic import (
    # Symbolic Entity Overlap (v13.0 - Hybrid Controller)
    compute_context_overlap,
    compute_numeric_consistency,
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
    # Layer Stability / Drift (v11.0)
    "compute_layer_drift",
    "compute_top_k_drift",
    "compute_rank_drift",
    "apply_drift_penalty",
    # Adaptive Gate (v12.2)
    "AdaptiveGate",
    "AdaptiveGateBatch",
    "compute_adaptive_stability_gate",
    # Symbolic Overlap (v13.0 - Hybrid Controller)
    "compute_context_overlap",
    "compute_numeric_consistency",
]
