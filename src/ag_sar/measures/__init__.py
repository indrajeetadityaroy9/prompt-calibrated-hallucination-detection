"""
AG-SAR Core Measures.

Three-pillar uncertainty quantification:
1. Authority Flow - information provenance from attention
2. Semantic Dispersion - top-k prediction consistency
3. Stability Gating - MLP-attention agreement
"""

from .authority import compute_semantic_authority
from .entropy import compute_token_entropy, compute_varentropy
from .semantics import compute_semantic_dispersion
from .stability import AdaptiveGateBatch

__all__ = [
    "compute_semantic_authority",
    "compute_token_entropy",
    "compute_varentropy",
    "compute_semantic_dispersion",
    "AdaptiveGateBatch",
]
