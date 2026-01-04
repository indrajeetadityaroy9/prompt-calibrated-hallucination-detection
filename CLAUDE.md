# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without external semantic models.

**Key Features:**
- Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
- Zero external latency: pure internal model analysis using attention patterns
- Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
- SOTA v8.0: Authority Flow + Unified Gating + Semantic Dispersion

## H100 Installation

Follow this specific order to avoid CUDA version mismatches on H100 systems:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with explicit CUDA index (do not let pip guess)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt

# 4. Post-install setup for baselines (NLTK sentence tokenization)
python -m nltk.downloader punkt

# 5. Verify environment
python -c "import torch; import transformers; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}'); print(f'Transformers: {transformers.__version__}'); print(f'H100 Detected: {torch.cuda.get_device_capability()[0] >= 9}')"
```

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
pytest tests/unit/test_core_ops.py -v

# Run single test function
pytest tests/unit/test_ops.py::test_fisher_kurtosis -v

# Run tests with coverage
pytest --cov=ag_sar tests/

# Linting and formatting
black .                    # Format code (line-length=100)
ruff check .               # Lint code
ruff check . --fix         # Lint and auto-fix
mypy src/ag_sar/           # Type checking

# Run latency benchmark
python -m experiments.analysis.benchmark_latency --model gpt2 --seq-len 128

# Run experiment (e.g., HaluEval QA benchmark)
python -m experiments.main --config experiments/configs/01_main_sota.yaml

# Dry run (print config only)
python -m experiments.main --config experiments/configs/01_main_sota.yaml --dry-run

# Reproduce all paper experiments
./reproduce_paper.sh
```

## Architecture

### Core Package (`src/ag_sar/`)

```
AGSAR Engine (engine.py)
├── Main API: compute_uncertainty(), detect_hallucination()
├── Authority Flow + Unified Gating + Semantic Dispersion
└── Orchestrates: extract → authority_score → gating → dispersion → score

ModelAdapter (modeling/hooks.py)
├── Hook-based Q/K/V extraction without O(N²) matrices
├── Architecture detection: GPT-2 (c_attn), Llama/Mistral/Qwen (monkey-patch)
├── GQA expansion: 8 KV-heads → 32 Q-heads for Llama-3.1
└── AttentionCapture dataclass for multi-layer storage

Measures (measures/)
├── authority.py: Authority Flow + Register Filter + Gated/Semantic Authority
├── semantics.py: Semantic Dispersion (consistency over confidence)
└── entropy.py: Token entropy (baseline utility)

Note: Ablation code (LID, Spectral, Legacy Graph) archived in legacy_research/

Ops (ops/)
├── torch_functional.py: PyTorch implementations with torch.compile
├── triton_kernels.py: Linux-only Triton kernels for H100
└── Exports: EMAState, fisher_kurtosis, compute_authority_flow, etc.

Utils (utils/)
└── tensor.py: H100 optimizations, TF32, Flash Attention, safe_normalize, attention masking
```

### Data Flow

```
prompt + response → tokenize → ModelAdapter.extract()
    → Q, K, value_states, attn_outputs, block_outputs
    → compute_register_mask() [Mechanism 1: Kurtosis-based filter]
    → compute_authority_score() [Mechanism 2: Recursive prompt recharge]
    → compute_gated_authority() [Mechanism 3: Unified RAG/free-gen gating]
    → compute_semantic_authority() [Mechanism 4: Semantic dispersion]
    → detect_hallucination(uncertainty > threshold)
```

## Critical Implementation Details

### Precision Requirements
- **BFloat16 required for GPT-2** - float16 causes NaN overflow
- TF32 enforced at import via `enable_h100_optimizations()` (~3x speedup on H100)
- `torch.compile` used on hot paths (Ampere+ GPUs only, compute capability >= 8)

### Architecture-Specific Hooks
- **GPT-2**: Uses c_attn hook (fused QKV projection), no RoPE
- **Llama/Mistral/Qwen**: Monkey-patches attention.forward for post-RoPE Q/K capture
- **GQA**: Auto-expands KV heads via `align_gqa_heads()` (8 KV → 32 Q for Llama-3.1-8B)

### Critical Parameters (AGSARConfig)

- `residual_weight=0.5`: Prevents early-token collapse in power iteration
- `power_iteration_steps=3`: Converges in 2-3 iterations
- `semantic_layers=4`: Final layers contain semantic consolidation
- `sink_token_count=4`: First N tokens masked as structural sinks
- `lambda_roughness=10.0`: MLP divergence penalty
- `kurtosis_threshold=2.0`: Register filter threshold
- `ema_decay=0.995`: Online kurtosis normalization decay
- `hallucination_threshold=0.7`: Default detection threshold
- `enable_unified_gating=True`: Context-dependent RAG/free-gen gating
- `enable_semantic_dispersion=True`: Semantic consistency over raw confidence

For ablation studies (v3.1 baseline comparison), set `enable_unified_gating=False`.

### Transformers Version
**Critical**: transformers >= 4.45 has breaking changes for attention hooks that break the monkey-patch mechanism. The dependency is pinned to `>=4.40.0,<4.45.0` in pyproject.toml. If you must use a newer version, expect hook registration failures on Llama/Mistral/Qwen architectures.

### Platform Notes
- **Triton kernels**: Linux-only (`triton_kernels.py`). Falls back to `torch_functional.py` on macOS/Windows.
- **Force PyTorch backend**: Set `AG_SAR_USE_TORCH=1` environment variable to bypass Triton even on Linux.
- **Flash Attention**: Install with `pip install -e ".[h100]"` for full H100 SDPA acceleration. Requires CUDA 12+ and Linux.
- **Multi-GPU**: Supports `device_map="balanced"` for large models. Tensors stay on native device until final aggregation.

### Streaming & State Management
- **EMA state**: Maintains running Welford statistics for kurtosis normalization across tokens.
- **Reset between pairs**: Call `agsar.reset()` between different prompt-response pairs to clear streaming state.
- **Cleanup hooks**: Call `agsar.cleanup()` when done to remove model hooks and free resources.

## Key Abstractions

1. **Matrix-Free Centrality**: Power iteration computes eigenvector centrality in O(N) memory: `v_{t+1} = (1-α)·Attn(Q,K,v_t) + α·v_t`

2. **Register Filter (Mechanism 1)**: Kurtosis-based detection with EMA adaptation: `M(t) = (t > 4) × Sigmoid(-Z(t) + τ)` filters attention sinks

3. **Authority Flow (Mechanism 2)**: Recursive prompt recharge prevents vanishing authority: `A(t) = Σ_prompt A_{t,j} + Σ_gen A_{t,j} × A(j) × M(t)`

4. **MLP Divergence (Mechanism 3)**: Detects when MLP overrides attention: `δ(t) = 1 - CosineSim(h_attn, h_block)`

## Archived Ablation Code

For paper reproducibility (Table 3), ablation code is archived in `legacy_research/`:

| Component | Archived Location | Description |
|-----------|-------------------|-------------|
| LID (Manifold) | `legacy_research/ablations/manifold.py` | v5.0 Local Intrinsic Dimension |
| Spectral | `legacy_research/ablations/spectral.py` | v6.0 Laplacian entropy + DoLa |
| Legacy Graph | `legacy_research/ablations/legacy_graph.py` | v1/v2 centrality-based approach |

To reproduce ablations, copy files back to `src/ag_sar/measures/ablations/` (see `legacy_research/README.md`).

For v3.1 baseline comparison (pure Authority Flow without gating):
```python
config = AGSARConfig(enable_unified_gating=False, enable_semantic_dispersion=False)
```

## Test Organization

- `tests/unit/`: Individual component tests (ops, config, centrality, hooks, core_ops)
- `tests/integration/`: Full pipeline tests with real models
- `tests/conftest.py`: Shared fixtures (device, dtype, tensor dimensions)

## Experiments (Laboratory)

The `experiments/` directory contains scientific evaluation code, separated from the core library:

```
experiments/
├── main.py                    # Single CLI entry point
├── core/
│   ├── engine.py              # BenchmarkEngine orchestrator
│   ├── metrics.py             # AUROC, AUPRC, bootstrap CI
│   └── logging.py             # JSONL streaming logger
├── data/
│   ├── base.py                # EvaluationDataset ABC
│   ├── halueval.py            # HaluEval loader
│   └── ragtruth.py            # RAGTruth loader
├── methods/
│   ├── base.py                # UncertaintyMethod ABC
│   ├── agsar_wrapper.py       # AG-SAR method
│   ├── logprob.py             # Log-probability baseline
│   ├── entropy.py             # Entropy baseline
│   ├── selfcheck.py           # SelfCheck baseline
│   └── eigenscore.py          # EigenScore baseline
├── configs/
│   ├── schema.py              # Pydantic config validation
│   ├── 00_ci_smoke_test.yaml  # CI validation
│   ├── 01_main_sota.yaml      # Table 1: SOTA comparison
│   ├── 02_scaling_law.yaml    # Figure 2: Scaling to 70B
│   ├── 03_generalization.yaml # Table 2: RAGTruth generalization
│   └── 04_moe_robustness.yaml # Discussion: MoE architecture
└── analysis/
    └── benchmark_latency.py   # Zero-latency verification
```

**Paper Experiments:**
- `00_ci_smoke_test.yaml`: CI validation (fast sanity check)
- `01_main_sota.yaml`: Table 1 - SOTA comparison on HaluEval QA
- `02_scaling_law.yaml`: Figure 2 - Scaling to Llama-3.1-70B
- `03_generalization.yaml`: Table 2 - RAGTruth natural hallucinations
- `04_moe_robustness.yaml`: Discussion - MoE architecture (Mixtral)

Note: Ablation experiment config (`05_mechanism_ablation.yaml`) archived in `legacy_research/`.
