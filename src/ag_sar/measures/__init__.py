"""
AG-SAR Core Measures (Algorithms).

Consolidated uncertainty quantification metrics:
- authority: v3.1 Authority Flow & v3.2 MLP Divergence
- graph: Matrix-free eigenvector centrality via power iteration
- entropy: Token entropy (baseline metric)
- manifold: v5.0 Local Intrinsic Dimension (LID) for confabulation detection
- spectral_structural: v6.0 Laplacian Entropy & Layer-Contrastive Divergence
"""

from .authority import (
    compute_authority_score,
    compute_register_mask,
    compute_mlp_divergence,
    compute_v7_gated_authority,
    compute_v8_semantic_authority,
)

from .graph import (
    compute_centrality,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)

from .entropy import (
    compute_token_entropy,
)

from .manifold import (
    compute_lid,
    compute_lid_fast,
    compute_lid_penalty,
    compute_curvature_proxy,
    # v5.1 Adaptive Calibration
    ManifoldSignature,
    compute_manifold_signature,
    compute_adaptive_lid_penalty,
    compute_lid_with_calibration,
)

from .spectral_structural import (
    # v6.0 Spectral-Structural Methods
    compute_laplacian_entropy,
    compute_laplacian_entropy_per_token,
    compute_layer_divergence,
    compute_layer_divergence_from_hidden,
    compute_spectral_score,
    compute_attention_structure_score,
)

from .semantics import (
    # v8.0 Semantic Dispersion
    compute_semantic_dispersion,
    compute_semantic_trust,
    compute_entropy_weighted_dispersion,
)

__all__ = [
    # Authority
    "compute_authority_score",
    "compute_register_mask",
    "compute_mlp_divergence",
    "compute_v7_gated_authority",
    "compute_v8_semantic_authority",
    # Graph
    "compute_centrality",
    "compute_sink_aware_centrality",
    "aggregate_value_norms",
    # Entropy
    "compute_token_entropy",
    # Manifold (v5.0/v5.1)
    "compute_lid",
    "compute_lid_fast",
    "compute_lid_penalty",
    "compute_curvature_proxy",
    "ManifoldSignature",
    "compute_manifold_signature",
    "compute_adaptive_lid_penalty",
    "compute_lid_with_calibration",
    # Spectral-Structural (v6.0)
    "compute_laplacian_entropy",
    "compute_laplacian_entropy_per_token",
    "compute_layer_divergence",
    "compute_layer_divergence_from_hidden",
    "compute_spectral_score",
    "compute_attention_structure_score",
    # Semantic Dispersion (v8.0)
    "compute_semantic_dispersion",
    "compute_semantic_trust",
    "compute_entropy_weighted_dispersion",
]
