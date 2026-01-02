"""
AG-SAR: Attention-Graph Shifting Attention to Relevance

Zero-latency uncertainty quantification for LLMs by analyzing
internal attention graph structure.

Example:
    >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
    >>> from ag_sar import AGSAR
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> ag_sar = AGSAR(model, tokenizer)
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

__version__ = "0.1.0"

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
]
