"""
Ablation Components for AG-SAR.

These modules implement experimental and ablation variants that are NOT
part of the default AG-SAR method. They are preserved for:
1. Reproducing paper ablation experiments
2. Research into alternative mechanisms
3. Backward compatibility with older configs

Default AG-SAR uses:
- Authority Flow (core)
- Unified Gating (core)
- Semantic Dispersion (core)

Ablation components:
- manifold: Local Intrinsic Dimension (LID) for confabulation detection
- spectral: Laplacian Entropy & Layer-Contrastive Divergence
- legacy_graph: Matrix-free eigenvector centrality (superseded by Authority Flow)
"""

from .manifold import (
    compute_lid,
    compute_lid_fast,
    compute_lid_penalty,
    compute_curvature_proxy,
    # Adaptive Calibration
    ManifoldSignature,
    compute_manifold_signature,
    compute_adaptive_lid_penalty,
    compute_lid_with_calibration,
)

from .spectral import (
    # Spectral-Structural Methods
    compute_laplacian_entropy,
    compute_laplacian_entropy_per_token,
    compute_layer_divergence,
    compute_layer_divergence_from_hidden,
    compute_spectral_score,
    compute_attention_structure_score,
)

from .legacy_graph import (
    compute_centrality,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)

__all__ = [
    # Manifold (LID)
    "compute_lid",
    "compute_lid_fast",
    "compute_lid_penalty",
    "compute_curvature_proxy",
    "ManifoldSignature",
    "compute_manifold_signature",
    "compute_adaptive_lid_penalty",
    "compute_lid_with_calibration",
    # Spectral-Structural
    "compute_laplacian_entropy",
    "compute_laplacian_entropy_per_token",
    "compute_layer_divergence",
    "compute_layer_divergence_from_hidden",
    "compute_spectral_score",
    "compute_attention_structure_score",
    # Legacy Graph
    "compute_centrality",
    "compute_sink_aware_centrality",
    "aggregate_value_norms",
]
