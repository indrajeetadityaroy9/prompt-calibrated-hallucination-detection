"""
AG-SAR Core Measures (Algorithms).

Consolidated uncertainty quantification metrics:
- authority: v3.1 Authority Flow & v3.2 MLP Divergence
- graph: Matrix-free eigenvector centrality via power iteration
- entropy: Token entropy (baseline metric)
"""

from .authority import (
    compute_authority_score,
    compute_register_mask,
    compute_mlp_divergence,
)

from .graph import (
    compute_centrality,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)

from .entropy import (
    compute_token_entropy,
)

__all__ = [
    # Authority
    "compute_authority_score",
    "compute_register_mask",
    "compute_mlp_divergence",
    # Graph
    "compute_centrality",
    "compute_sink_aware_centrality",
    "aggregate_value_norms",
    # Entropy
    "compute_token_entropy",
]
