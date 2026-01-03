"""
AG-SAR v3.1: Recursive Authority Flow for Zero-Latency Hallucination Detection

Implements the unified mathematical model from literature synthesis:
- Mechanism 1: Register Filter (Papers 1 & 2) - EMA Z-score + Sigmoid gate
- Mechanism 2: Authority Flow (Paper 6 corrected) - Prompt Recharge + Gen Flow
- Mechanism 3: Spectral Roughness (Paper 9) - Pre-MLP deviation metric
- Mechanism 4: SnapKV Eviction (Paper 5) - Authority-weighted voting

Example:
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
    >>> tokenizer = AutoTokenizer.from_pretrained('gpt2')
    >>> config = AGSARConfig(enable_register_filter=True, lambda_roughness=10.0)
    >>> ag_sar = AGSAR(model, tokenizer, config)
    >>> result = ag_sar.compute_uncertainty("The capital of France is", "Paris")
"""

from .config import AGSARConfig
from .utils import enable_tf32, is_tf32_enabled, get_model_dtype, get_model_device, apply_attention_mask, safe_normalize

# Enforce TF32 at import time for ~3x matmul speedup on H100/A100
# This ensures TF32 is enabled before any tensor operations
enable_tf32()
from .attention_extractor import AttentionExtractor
from .centrality import (
    matrix_free_power_iteration,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)
from .kernels import centrality_flash_fwd
from .uncertainty import (
    compute_token_entropy,
    normalize_relevance,
    compute_graph_shifted_entropy,
    detect_hallucination,
    compute_per_token_uncertainty,
)
from .ag_sar import AGSAR
from .multi_gpu import (
    MultiGPUAGSAR,
    GPUPool,
    get_optimal_gpu_count,
    distribute_samples,
)
# v3.1: Pure PyTorch operations for O(N) streaming inference
from .ops import (
    fisher_kurtosis,
    welford_update,
    compute_register_mask,
    compute_spectral_roughness,
    compute_spectral_roughness_gqa,
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_snapkv_eviction,
    compress_kv_cache,
    align_gqa_heads,
    get_gqa_config,
    EMAState,
)

__version__ = "0.3.1"

__all__ = [
    # Main class
    "AGSAR",
    "AGSARConfig",
    # Multi-GPU support
    "MultiGPUAGSAR",
    "GPUPool",
    "get_optimal_gpu_count",
    "distribute_samples",
    # Utils
    "enable_tf32",
    "is_tf32_enabled",
    "get_model_dtype",
    "get_model_device",
    "apply_attention_mask",
    "safe_normalize",
    # Attention extraction
    "AttentionExtractor",
    # Centrality (matrix-free via Triton)
    "matrix_free_power_iteration",
    "compute_sink_aware_centrality",
    "aggregate_value_norms",
    "centrality_flash_fwd",
    # Uncertainty (GSE)
    "compute_token_entropy",
    "normalize_relevance",
    "compute_graph_shifted_entropy",
    "detect_hallucination",
    "compute_per_token_uncertainty",
    # v3.1: Pure PyTorch O(N) operations
    "fisher_kurtosis",
    "welford_update",
    "compute_register_mask",
    "compute_spectral_roughness",
    "compute_spectral_roughness_gqa",
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    "compute_snapkv_eviction",
    "compress_kv_cache",
    "align_gqa_heads",
    "get_gqa_config",
    "EMAState",
]
