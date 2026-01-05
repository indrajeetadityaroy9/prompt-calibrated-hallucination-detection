"""
AG-SAR Core Measures (Algorithms) - v8.0 Gold Master.

SOTA v8.0 components:
- authority: Authority Flow, MLP Divergence, Gated Authority, Semantic Authority
- semantics: Semantic Dispersion (consistency over confidence)
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
]
