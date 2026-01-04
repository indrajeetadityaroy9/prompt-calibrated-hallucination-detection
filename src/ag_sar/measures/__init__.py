"""
AG-SAR Core Measures (Algorithms).

SOTA v8.0 components:
- authority: Authority Flow, MLP Divergence, Gated Authority, Semantic Authority
- semantics: Semantic Dispersion (consistency over confidence)
- entropy: Token entropy (baseline metric)

Ablation code (v4/v5/v6) is archived in legacy_research/ for paper reproducibility.
"""

# Core components
from .authority import (
    compute_authority_score,
    compute_register_mask,
    compute_mlp_divergence,
    # Canonical function names
    compute_gated_authority,
    compute_semantic_authority,
    # Deprecated aliases (backward compatibility)
    compute_v7_gated_authority,
    compute_v8_semantic_authority,
)

from .entropy import (
    compute_token_entropy,
)

from .semantics import (
    # Semantic Dispersion
    compute_semantic_dispersion,
    compute_semantic_trust,
    compute_entropy_weighted_dispersion,
)

__all__ = [
    # Authority (core)
    "compute_authority_score",
    "compute_register_mask",
    "compute_mlp_divergence",
    "compute_gated_authority",
    "compute_semantic_authority",
    # Authority (deprecated aliases)
    "compute_v7_gated_authority",
    "compute_v8_semantic_authority",
    # Entropy
    "compute_token_entropy",
    # Semantic Dispersion
    "compute_semantic_dispersion",
    "compute_semantic_trust",
    "compute_entropy_weighted_dispersion",
]
