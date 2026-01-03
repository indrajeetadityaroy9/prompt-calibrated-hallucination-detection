# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without external semantic models.

**Key Features:**
- Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
- Zero external latency: pure internal model analysis using attention patterns
- Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
- Core metric: v3.1 Authority Flow + v3.2 MLP Divergence

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
pytest tests/unit/test_v31_ops.py -v

# Run tests with coverage
pytest --cov=ag_sar tests/

# Linting and formatting
black .                    # Format code (line-length=100)
ruff check .               # Lint code
ruff check . --fix         # Lint and auto-fix
mypy src/ag_sar/           # Type checking

# Run benchmark
python benchmarks/benchmark_latency.py --model gpt2 --seq-len 128
```

## Architecture

### Core Package (`src/ag_sar/`)

```
AGSAR Engine (engine.py)
├── Main API: compute_uncertainty(), detect_hallucination()
├── v3.1 Authority Flow + v3.2 MLP Divergence
└── Orchestrates: extract → authority_score → mlp_divergence → score

ModelAdapter (modeling/hooks.py)
├── Hook-based Q/K/V extraction without O(N²) matrices
├── Architecture detection: GPT-2 (c_attn), Llama/Mistral/Qwen (monkey-patch)
├── GQA expansion: 8 KV-heads → 32 Q-heads for Llama-3.1
└── AttentionCapture dataclass for multi-layer storage

Measures (measures/)
├── authority.py: v3.1 Authority Flow + Register Filter + MLP Divergence
├── graph.py: Matrix-free O(N) eigenvector centrality via power iteration
└── entropy.py: Token entropy (baseline utility)

Ops (ops/)
├── torch_functional.py: PyTorch implementations with torch.compile
├── triton_kernels.py: Linux-only Triton kernels for H100
└── Exports: EMAState, fisher_kurtosis, compute_authority_flow, etc.

Utils (utils/)
└── tensor.py: H100 optimizations, TF32, Flash Attention, safe_normalize, attention masking
```

### Data Flow (v3.1/v3.2)

```
prompt + response → tokenize → ModelAdapter.extract()
    → Q, K, value_states, attn_outputs, block_outputs
    → compute_register_mask() [Mechanism 1: Kurtosis-based filter]
    → compute_authority_score() [Mechanism 2: Recursive prompt recharge]
    → compute_mlp_divergence() [Mechanism 3: Attention-MLP divergence]
    → uncertainty = 1 - authority / (1 + λ × divergence)
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
- `lambda_roughness=10.0`: MLP divergence penalty (v3.2)
- `kurtosis_threshold=2.0`: Register filter threshold
- `ema_decay=0.995`: Online kurtosis normalization decay

### Transformers Version
**Critical**: transformers >= 4.45 has breaking changes for attention hooks. Pinned to `>=4.40.0,<4.45.0`.

## Key Abstractions

1. **Matrix-Free Centrality**: Power iteration computes eigenvector centrality in O(N) memory: `v_{t+1} = (1-α)·Attn(Q,K,v_t) + α·v_t`

2. **Register Filter (Mechanism 1)**: Kurtosis-based detection with EMA adaptation: `M(t) = (t > 4) × Sigmoid(-Z(t) + τ)` filters attention sinks

3. **Authority Flow (Mechanism 2)**: Recursive prompt recharge prevents vanishing authority: `A(t) = Σ_prompt A_{t,j} + Σ_gen A_{t,j} × A(j) × M(t)`

4. **MLP Divergence (Mechanism 3)**: Detects when MLP overrides attention: `δ(t) = 1 - CosineSim(h_attn, h_block)`

## Test Organization

- `tests/unit/`: Individual component tests (ops, config, centrality, hooks, v31_ops)
- `tests/integration/`: Full pipeline tests with real models
- `tests/conftest.py`: Shared fixtures (small models, mock tokenizers)

## Benchmarks

- `benchmarks/benchmark_latency.py`: Zero-latency verification (<10% overhead target)
