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
from .utils import enable_tf32, get_model_dtype, get_model_device, apply_attention_mask, safe_normalize
from .attention_extractor import AttentionExtractor
from .centrality import (
    power_iteration,
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

__version__ = "0.1.0"

__all__ = [
    # Main class
    "AGSAR",
    "AGSARConfig",
    # Utils
    "enable_tf32",
    "get_model_dtype",
    "get_model_device",
    "apply_attention_mask",
    "safe_normalize",
    # Attention extraction
    "AttentionExtractor",
    # Centrality (matrix-free via Triton)
    "power_iteration",
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
