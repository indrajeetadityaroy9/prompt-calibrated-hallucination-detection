"""
AG-SAR: Attention-Graph Shifting Attention to Relevance.

Single-pass hallucination detection by analyzing attention patterns.

Core equation:
    Uncertainty(t) = 1 - Authority(t)
    Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)

Where:
- Flow: Information provenance from prompt tokens
- Gate: MLP-attention agreement (context reliance indicator)
- Trust: 1 - Dispersion × (1 + λ × Varentropy)
"""

__version__ = "0.5.0"

# Core API
from .engine import AGSAR
from .config import AGSARConfig
from .modeling import ModelAdapter, AttentionCapture

# Measures
from .measures import (
    compute_semantic_authority,
    compute_token_entropy,
    compute_varentropy,
    compute_semantic_dispersion,
)

# Operations
from .ops import (
    compute_authority_flow_vectorized,
    compute_mlp_divergence,
    fused_stability_gate,
)

# Utilities
from .utils import (
    enable_tf32,
    enable_h100_optimizations,
    is_tf32_enabled,
    get_optimal_dtype,
    get_model_dtype,
    get_model_device,
)

__all__ = [
    "__version__",
    "AGSAR",
    "AGSARConfig",
    "ModelAdapter",
    "AttentionCapture",
    "compute_semantic_authority",
    "compute_token_entropy",
    "compute_varentropy",
    "compute_semantic_dispersion",
    "compute_authority_flow_vectorized",
    "compute_mlp_divergence",
    "fused_stability_gate",
    "enable_tf32",
    "enable_h100_optimizations",
    "is_tf32_enabled",
    "get_optimal_dtype",
    "get_model_dtype",
    "get_model_device",
]
