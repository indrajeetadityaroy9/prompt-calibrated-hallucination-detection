"""
AG-SAR: Attention-Graph Shifting Attention to Relevance.

This module implements a zero-latency hallucination detection framework that
identifies extrinsic hallucinations (unfaithfulness to source context) by
analyzing internal attention graph structure. The approach requires no external
semantic models or multiple forward passes.

Mechanism Overview:
    AG-SAR combines three complementary signals to quantify uncertainty:

    1. Authority Flow: Computes information provenance by recursively tracing
       attention weights from response tokens back to prompt tokens. Tokens
       that derive authority primarily from the prompt are considered grounded;
       those deriving authority from parametric memory are flagged as uncertain.

    2. Unified Gating: Dynamically balances context-derived authority against
       parametric confidence based on the model's attention distribution. When
       attention concentrates on context, trust authority flow; when attention
       is diffuse, incorporate parametric signals.

    3. Semantic Dispersion: Measures consistency of top-k predictions in
       embedding space. Low dispersion (synonyms like US/USA/America) indicates
       grounded generation; high dispersion (unrelated alternatives) indicates
       hallucination risk.

Pipeline Position:
    AG-SAR operates as a post-hoc uncertainty quantifier. Given a frozen LLM
    and a (prompt, response) pair, it instruments the forward pass to extract
    attention patterns, then computes uncertainty without modifying generation.

Assumptions:
    - Model uses standard multi-head attention (GPT-2, Llama, Mistral, Qwen)
    - Prompt contains the source context to verify against
    - Response is already generated (not streaming token-by-token)

Scope Limitation:
    AG-SAR detects EXTRINSIC hallucinations (contradicting provided context),
    NOT INTRINSIC hallucinations (contradicting world knowledge). For RAG
    faithfulness monitoring where context defines ground truth.

Public API:
    - AGSAR: Main engine class for uncertainty computation
    - AGSARConfig: Configuration dataclass with validated parameters
    - compute_*: Low-level measure functions for custom pipelines
    - load_preset: Task-specific configuration presets (qa, rag, summarization)
"""

__version__ = "0.4.0"

# Enable hardware optimizations at import time.
# TF32 provides 3x speedup for FP32 matmul on Ampere/Hopper GPUs.
# Flash SDP enables memory-efficient attention when available.
from .utils import enable_h100_optimizations
enable_h100_optimizations()

# Core API - Primary user-facing interface
from .engine import AGSAR
from .config import AGSARConfig
from .modeling import ModelAdapter, AttentionCapture

# Measures - Individual uncertainty signals (for custom pipelines)
from .measures import (
    # Authority Flow: Information provenance tracking
    compute_authority_score,
    compute_mlp_divergence,
    compute_gated_authority,
    compute_semantic_authority,
    # Token Entropy: Predictive uncertainty
    compute_token_entropy,
    # Semantic Dispersion: Top-k consistency in embedding space
    compute_semantic_dispersion,
    compute_semantic_trust,
)

# Operations - Low-level tensor computations (for advanced users)
from .ops import (
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_mlp_divergence as compute_mlp_divergence_op,
    compute_stability_gate,
    align_gqa_heads,
    _TRITON_AVAILABLE,
)

# Utilities - Hardware optimization and tensor helpers
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

# Presets - Task-specific calibration parameters
from .presets import (
    load_preset,
    get_available_presets,
    clear_preset_cache,
)

__all__ = [
    # Version
    "__version__",
    # Core API
    "AGSAR",
    "AGSARConfig",
    "ModelAdapter",
    "AttentionCapture",
    # Measures - Authority Flow
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
    # Utilities
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
