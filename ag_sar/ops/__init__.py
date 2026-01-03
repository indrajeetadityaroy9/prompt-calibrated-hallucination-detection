"""
AG-SAR v3.2 Operations Module.

Pure PyTorch implementations for zero-latency O(N) operations.
No external dependencies (scipy, etc.) required for core functionality.

Components:
- fisher_kurtosis: Register detection (distinguishes spiky vs flat distributions)
- welford_update: Online EMA statistics for streaming inference
- compute_register_mask: Full register filter with EMA Z-score
- compute_spectral_roughness: Dirichlet energy roughness (v3.1)
- compute_spectral_roughness_gqa: GQA-compatible roughness for Llama-3.1
- compute_mlp_divergence: MLP layer divergence metric (v3.2 for Llama-3)
- compute_authority_flow: Recursive authority with prompt recharge
- compute_authority_flow_vectorized: Batch-mode authority approximation
- compute_snapkv_eviction: Authority-weighted KV cache eviction
- compress_kv_cache: Apply eviction indices to KV states
- align_gqa_heads: Expand KV heads to match Q heads for GQA
- get_gqa_config: Extract GQA configuration from model config
"""

from .functional import (
    fisher_kurtosis,
    welford_update,
    compute_register_mask,
    compute_spectral_roughness,
    compute_spectral_roughness_gqa,
    compute_mlp_divergence,
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_snapkv_eviction,
    compress_kv_cache,
    align_gqa_heads,
    get_gqa_config,
    EMAState,
)

__all__ = [
    "fisher_kurtosis",
    "welford_update",
    "compute_register_mask",
    "compute_spectral_roughness",
    "compute_spectral_roughness_gqa",
    "compute_mlp_divergence",
    "compute_authority_flow",
    "compute_authority_flow_vectorized",
    "compute_snapkv_eviction",
    "compress_kv_cache",
    "align_gqa_heads",
    "get_gqa_config",
    "EMAState",
]
