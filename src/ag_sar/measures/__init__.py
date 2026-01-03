"""
AG-SAR Core Measures (Algorithms).

Consolidated uncertainty quantification metrics:
- authority: v3.1 Authority Flow & v3.2 MLP Divergence
- graph: Matrix-free eigenvector centrality via power iteration
- entropy: Token entropy and Graph-Shifted Entropy (GSE)
- spectral: Manifold-Consistent Spectral Surprisal (MC-SS)
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
    compute_hebbian_weights,
)

from .entropy import (
    compute_token_entropy,
    compute_graph_shifted_entropy,
    normalize_relevance,
    detect_hallucination,
)

from .spectral import (
    compute_bounded_surprisal,
    compute_manifold_consistent_spectral_surprisal,
    compute_token_surprisal,
    compute_graph_shifted_surprisal,
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
    "compute_hebbian_weights",
    # Entropy
    "compute_token_entropy",
    "compute_graph_shifted_entropy",
    "normalize_relevance",
    "detect_hallucination",
    # Spectral
    "compute_bounded_surprisal",
    "compute_manifold_consistent_spectral_surprisal",
    "compute_token_surprisal",
    "compute_graph_shifted_surprisal",
]
