# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without external semantic models.

**Key Features:**
- Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
- Zero external latency: pure internal model analysis using attention patterns
- Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
- Core metrics: Graph-Shifted Entropy (GSE), Manifold-Consistent Spectral Surprisal (MC-SS), Authority Flow (v3.1)

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install all including eval dependencies
pip install -e ".[all]"

# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v

# Run specific test file
pytest tests/unit/test_ops.py -v

# Run tests with coverage
pytest --cov=ag_sar tests/

# Run benchmark
python benchmarks/benchmark_latency.py --model gpt2 --seq-len 128
```

## Architecture

### Core Package (`src/ag_sar/`)

```
AGSAR Engine (engine.py)
├── Main API: compute_uncertainty(), detect_hallucination()
├── Routes to metric implementations: GSE, MC-SS, v3.1
└── Orchestrates: extract → centrality → entropy → score

ModelAdapter (modeling/hooks.py)
├── Hook-based Q/K/V extraction without O(N²) matrices
├── Architecture detection: GPT-2 (c_attn), Llama/Mistral/Qwen (monkey-patch)
├── GQA expansion: 8 KV-heads → 32 Q-heads for Llama-3.1
└── AttentionCapture dataclass for multi-layer storage

Measures (measures/)
├── authority.py: v3.1 Authority Flow + Register Filter + MLP Divergence
├── graph.py: Matrix-free O(N) eigenvector centrality via power iteration
├── entropy.py: Graph-Shifted Entropy, token entropy
└── spectral.py: MC-SS with Hebbian weighting

Ops (ops/)
├── torch_functional.py: PyTorch implementations with torch.compile
├── triton_kernels.py: Linux-only Triton kernels for H100
└── Exports: EMAState, fisher_kurtosis, compute_authority_flow, etc.

Utils (utils/)
├── tensor.py: H100 optimizations, TF32, Flash Attention
└── numerical.py: safe_normalize, safe_log, attention masking
```

### Data Flow

```
prompt + response → tokenize → ModelAdapter.extract_semantic_qk()
    → Q, K, value_norms, logits (from final N semantic layers)
    → compute_sink_aware_centrality() via matrix-free power iteration
    → compute_token_entropy(logits)
    → Metric routing:
        GSE:  entropy × normalized_relevance
        MC-SS: bounded_surprisal + λ(1 - centrality)
        v3.1: authority_flow / (1 + λ × mlp_divergence)
    → detect_hallucination(uncertainty > threshold)
```

## Critical Implementation Details

### Precision Requirements
- **BFloat16 required for GPT-2** - float16 causes NaN overflow
- TF32 enforced at import via `enable_h100_optimizations()` (~3x speedup on H100)
- `torch.compile` used on hot paths (Ampere+ GPUs only)

### Architecture-Specific Hooks
- **GPT-2**: Uses c_attn hook (fused QKV projection)
- **Llama/Mistral/Qwen**: Monkey-patches attention.forward for post-RoPE Q/K capture
- **GQA**: Auto-expands KV heads via `align_gqa_heads()` (8 KV → 32 Q for Llama-3.1-8B)

### Critical Parameters (AGSARConfig)
- `residual_weight=0.5`: Prevents early-token collapse in power iteration
- `power_iteration_steps=3`: Converges in 2-3 iterations
- `semantic_layers=4`: Final layers contain semantic consolidation
- `sink_token_count=4`: First N tokens masked as structural sinks
- `lambda_roughness=10.0`: MLP divergence penalty (v3.1/v3.2)

### Transformers Version
**Critical**: transformers >= 4.45 has breaking changes for attention hooks. Pinned to `>=4.40.0,<4.45.0`.

## Key Abstractions

1. **Matrix-Free Centrality**: Power iteration computes eigenvector centrality in O(N) memory: `v_{t+1} = (1-α)·Attn(Q,K,v_t) + α·v_t`

2. **Register Filter**: Kurtosis-based detection with EMA adaptation: `M(t) = (t > 4) × Sigmoid(-Z(t) + τ)` filters attention sinks

3. **Authority Flow**: Recursive prompt recharge prevents vanishing authority: `A(t) = Σ_prompt A_{t,j} + Σ_gen A_{t,j} × A(j) × M(t)`

4. **MLP Divergence (v3.2)**: Detects when MLP overrides attention: `δ(t) = 1 - CosineSim(h_attn, h_block)`

## Test Organization

- `tests/unit/`: Individual component tests (ops, config, measures)
- `tests/integration/`: Full pipeline tests with real models
- `tests/conftest.py`: Shared fixtures (small models, mock tokenizers)

## Benchmark Configs

- `benchmarks/configs/scaling_law.yaml`: 8B vs 70B comparison
- `benchmarks/benchmark_latency.py`: Zero-latency verification (<10% overhead)
