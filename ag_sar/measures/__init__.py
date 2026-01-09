"""
AG-SAR Core Measures.

Three-pillar uncertainty quantification:
1. Authority Flow - information provenance from attention
2. Semantic Dispersion - top-k prediction consistency
3. Stability Gating - MLP-attention agreement
"""

from .authority import (
    compute_semantic_authority,
    compute_emergence_gated_trust,
    compute_semantic_authority_v3,
    compute_semantic_authority_v4,
    compute_semantic_authority_v5,
    compute_semantic_authority_v6,
    compute_semantic_authority_v7,
    compute_semantic_authority_v8,
    compute_semantic_authority_v9,
    compute_semantic_authority_v10,
    compute_semantic_authority_v11,
    compute_semantic_authority_v12,
    compute_semantic_authority_v13,
    compute_semantic_authority_v15,
    compute_semantic_authority_v16,
    compute_semantic_authority_v19,
)
from .geometry import (
    compute_local_intrinsic_dimension,
    compute_manifold_score,
    compute_participation_ratio,
)
from .entropy import (
    compute_token_entropy,
    compute_varentropy,
    compute_epiplexity,
    compute_gaussian_complexity,
)
from .semantics import compute_semantic_dispersion
from .stability import AdaptiveGateBatch

__all__ = [
    "compute_semantic_authority",
    "compute_emergence_gated_trust",
    "compute_semantic_authority_v3",
    "compute_semantic_authority_v4",
    "compute_semantic_authority_v5",
    "compute_semantic_authority_v6",
    "compute_semantic_authority_v7",
    "compute_semantic_authority_v8",
    "compute_semantic_authority_v9",
    "compute_semantic_authority_v10",
    "compute_semantic_authority_v11",
    "compute_semantic_authority_v12",
    "compute_semantic_authority_v13",
    "compute_semantic_authority_v15",
    "compute_semantic_authority_v16",
    "compute_semantic_authority_v19",
    "compute_local_intrinsic_dimension",
    "compute_manifold_score",
    "compute_participation_ratio",
    "compute_token_entropy",
    "compute_varentropy",
    "compute_epiplexity",
    "compute_gaussian_complexity",
    "compute_semantic_dispersion",
    "AdaptiveGateBatch",
]
