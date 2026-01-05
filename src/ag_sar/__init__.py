"""
AG-SAR: Attention-Graph Shifting Attention to Relevance.

Zero-latency uncertainty quantification framework for LLMs.
Detects hallucinations by analyzing internal attention graph structure
without external semantic models.

Key Features:
    - Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
    - Zero external latency: pure internal model analysis
    - Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
    - Core: Authority Flow + Unified Gating + Semantic Dispersion

Example:
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> config = AGSARConfig()  # Uses default settings (semantic dispersion enabled)
    >>> agsar = AGSAR(model, tokenizer, config)
    >>> score = agsar.compute_uncertainty(prompt, response)
    >>> is_hall, conf, details = agsar.detect_hallucination(prompt, response)
"""

__version__ = "0.4.0"

# Enable H100/Hopper optimizations at import
# - TF32 for 3x faster FP32 matmul
# - Flash SDP for memory-efficient attention
# - cuDNN benchmark for optimized algorithms
from .utils import enable_h100_optimizations
enable_h100_optimizations()

# Core API
from .engine import AGSAR
from .config import AGSARConfig
from .modeling import ModelAdapter, AttentionCapture

# Measures (for advanced users)
from .measures import (
    # Authority (core)
    compute_authority_score,
    compute_mlp_divergence,
    compute_gated_authority,
    compute_semantic_authority,
    # Entropy
    compute_token_entropy,
    # Semantic Dispersion
    compute_semantic_dispersion,
    compute_semantic_trust,
)

# Operations (for custom pipelines)
from .ops import (
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_mlp_divergence as compute_mlp_divergence_op,
    compute_stability_gate,
    align_gqa_heads,
    _TRITON_AVAILABLE,
)

# Utilities
from .utils import (
    enable_tf32,
    enable_h100_optimizations,
    is_tf32_enabled,
    is_h100,
    get_optimal_dtype,
    optimize_for_inference,
    safe_normalize,
    apply_attention_mask,
    get_model_dtype,
    get_model_device,
)

# Presets (v9.0 Task-Adaptive)
from .presets import (
    load_preset,
    get_available_presets,
    clear_preset_cache,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "AGSAR",
    "AGSARConfig",
    "ModelAdapter",
    "AttentionCapture",
    # Measures - Authority
    "compute_authority_score",
    "compute_mlp_divergence",
    "compute_gated_authority",
    "compute_semantic_authority",
    # Measures - Entropy
    "compute_token_entropy",
    # Measures - Semantic Dispersion
    "compute_semantic_dispersion",
    "compute_semantic_trust",
    # Operations
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    "compute_stability_gate",
    "align_gqa_heads",
    "_TRITON_AVAILABLE",
    # Utilities - H100 Optimization
    "enable_tf32",
    "enable_h100_optimizations",
    "is_tf32_enabled",
    "is_h100",
    "get_optimal_dtype",
    "optimize_for_inference",
    "safe_normalize",
    "apply_attention_mask",
    "get_model_dtype",
    "get_model_device",
    # Presets
    "load_preset",
    "get_available_presets",
    "clear_preset_cache",
]
