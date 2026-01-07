"""
AG-SAR Core Measures Module.

This module implements the three-pillar uncertainty quantification mechanism:

1. Authority Flow (authority.py):
   Computes information provenance by recursively tracing attention weights from
   response tokens back to prompt tokens. The authority of a generated token
   reflects how much of its "information source" derives from the provided context
   versus the model's parametric memory.

   Key functions:
   - compute_authority_score: Base authority flow computation
   - compute_gated_authority: Authority with unified stability gating
   - compute_semantic_authority: Full mechanism with semantic dispersion

2. Semantic Dispersion (semantics.py):
   Measures the consistency of top-k predictions in embedding space. Low dispersion
   (synonyms like US/USA/America) indicates grounded generation; high dispersion
   (unrelated alternatives like Paris/London/Tokyo) indicates hallucination risk.

   Key functions:
   - compute_semantic_dispersion: Dispatcher for dispersion algorithms
   - compute_semantic_trust: Converts dispersion to trust score

3. Stability Gating (stability.py):
   Computes a gate value G ∈ [0,1] that indicates whether to trust context-derived
   authority (G≈1) or parametric confidence (G≈0). Based on MLP-attention divergence.

   Key classes:
   - AdaptiveGate: Online z-score normalization for model-agnostic gating
   - AdaptiveGateBatch: Batch-aware version for sequence tensors

4. Token Entropy (entropy.py):
   Baseline predictive entropy computation for comparison with AG-SAR measures.

Pipeline Position:
    These measures are composed by the AGSAR engine (engine.py) to compute final
    uncertainty scores. For most use cases, prefer the AGSAR class over calling
    these functions directly.
"""

# Authority Flow components
from .authority import (
    compute_authority_score,
    compute_mlp_divergence,
    compute_gated_authority,
    compute_semantic_authority,
)

# Token entropy (baseline measure)
from .entropy import (
    compute_token_entropy,
)

# Semantic Dispersion components
from .semantics import (
    compute_semantic_dispersion,
    compute_semantic_trust,
)

# Adaptive Stability Gate components
from .stability import (
    AdaptiveGate,
    AdaptiveGateBatch,
    compute_adaptive_stability_gate,
)

__all__ = [
    # Authority Flow
    "compute_authority_score",
    "compute_mlp_divergence",
    "compute_gated_authority",
    "compute_semantic_authority",
    # Token Entropy
    "compute_token_entropy",
    # Semantic Dispersion
    "compute_semantic_dispersion",
    "compute_semantic_trust",
    # Adaptive Stability Gate
    "AdaptiveGate",
    "AdaptiveGateBatch",
    "compute_adaptive_stability_gate",
]
