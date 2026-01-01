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
from .head_filter import (
    compute_attention_entropy,
    compute_head_entropy,
    filter_heads_by_entropy,
)
from .attention_graph import (
    add_residual_connection,
    compute_attention_rollout,
    build_global_attention_graph,
)
from .centrality import (
    power_iteration,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)
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
    # Head filtering
    "compute_attention_entropy",
    "compute_head_entropy",
    "filter_heads_by_entropy",
    # Graph construction
    "add_residual_connection",
    "compute_attention_rollout",
    "build_global_attention_graph",
    # Centrality
    "power_iteration",
    "compute_sink_aware_centrality",
    "aggregate_value_norms",
    # Uncertainty (GSE)
    "compute_token_entropy",
    "normalize_relevance",
    "compute_graph_shifted_entropy",
    "detect_hallucination",
    "compute_per_token_uncertainty",
]
