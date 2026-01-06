# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without external semantic models.

**Key Features:**
- Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
- Zero external latency: pure internal model analysis using attention patterns
- Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
- SOTA v8.0: Authority Flow + Unified Gating + Semantic Dispersion

## Quick Usage

```python
from ag_sar import AGSAR, AGSARConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

agsar = AGSAR(model, tokenizer)  # Default v8.0 config
score = agsar.compute_uncertainty("What is the capital of France?", "Paris")

# Detect context violations (extrinsic hallucinations)
# NOTE: AG-SAR detects unfaithfulness to provided context, not factual errors
context = "The Eiffel Tower is located in Paris, France."
question = "Where is the Eiffel Tower?"
response = "The Eiffel Tower is in London."
is_violation, conf, details = agsar.detect_context_violation(
    f"{context}\n\n{question}", response
)

# Always cleanup when done
agsar.cleanup()
```

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
├── Main API: compute_uncertainty(), compute_uncertainty_raw(), detect_hallucination()
├── Authority Flow + Unified Gating + Semantic Dispersion
└── Orchestrates: extract → authority_score → gating → dispersion → score

ModelAdapter (modeling/hooks.py)
├── Hook-based Q/K/V extraction without O(N²) matrices
├── Architecture detection: GPT-2 (c_attn), Llama/Mistral/Qwen (monkey-patch)
├── GQA expansion: 8 KV-heads → 32 Q-heads for Llama-3.1
└── AttentionCapture dataclass for multi-layer storage

Measures (measures/)
├── authority.py: Authority Flow + Gated/Semantic Authority
├── semantics.py: Semantic Dispersion (consistency over confidence)
├── stability.py: Adaptive Gate for model-agnostic normalization
└── entropy.py: Token entropy (baseline utility)

Ops (ops/)
├── torch_functional.py: PyTorch implementations with torch.compile
├── triton_kernels.py: Linux-only Triton kernels for H100
└── Exports: compute_authority_flow, compute_mlp_divergence, compute_stability_gate, etc.

Utils (utils/)
└── tensor.py: H100 optimizations, TF32, Flash Attention, safe_normalize, attention masking
```

### Data Flow

```
prompt + response → tokenize → ModelAdapter.extract()
    → Q, K, value_states, attn_outputs, block_outputs
    → compute_authority_score() [Mechanism 1: Recursive prompt recharge]
    → compute_gated_authority() [Mechanism 2: Unified RAG/free-gen gating]
    → compute_semantic_authority() [Mechanism 3: Semantic dispersion]
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

- `semantic_layers=4`: Final layers contain semantic consolidation
- `hallucination_threshold=0.7`: Default detection threshold
- `enable_unified_gating=True`: Context-dependent RAG/free-gen gating
- `stability_sensitivity=1.0`: Gate sharpness for MLP stability
- `parametric_weight=0.5`: Weight for confidence when ignoring context
- `enable_semantic_dispersion=True`: Semantic consistency over raw confidence
- `dispersion_k=5`: Top-k tokens for dispersion calculation
- `dispersion_sensitivity=1.0`: Scale factor for dispersion penalty

For ablation studies (v3.1 baseline comparison), set `enable_unified_gating=False`.

### Transformers Version
**Critical**: transformers >= 4.45 has breaking changes for attention hooks that break the monkey-patch mechanism. The dependency is pinned to `>=4.40.0,<4.45.0` in pyproject.toml. If you must use a newer version, expect hook registration failures on Llama/Mistral/Qwen architectures.

**Symptoms of version mismatch:** `RuntimeError: Attention weights not captured` on Llama/Mistral/Qwen models. GPT-2 will still work (uses different hook mechanism).

### Platform Notes
- **Triton kernels**: Linux-only (`triton_kernels.py`). Falls back to `torch_functional.py` on macOS/Windows.
- **Force PyTorch backend**: Set `AG_SAR_USE_TORCH=1` environment variable to bypass Triton even on Linux.
- **Flash Attention**: Install with `pip install -e ".[h100]"` for full H100 SDPA acceleration. Requires CUDA 12+ and Linux.
- **Multi-GPU**: Supports `device_map="balanced"` for large models. Tensors stay on native device until final aggregation.

### Resource Management
- **Cleanup hooks**: Call `agsar.cleanup()` when done to remove model hooks and free resources.

## Key Abstractions

1. **Authority Flow**: Recursive prompt recharge tracks signal provenance: `A(t) = Σ_prompt A_{t,j} + Σ_gen A_{t,j} × A(j)`

2. **Stability Gate**: Detects when MLP overrides attention: `Gate(t) = exp(-sensitivity × (1 - CosineSim(h_attn, h_block)))`

3. **Semantic Dispersion**: Measures consistency of top-k predictions: low dispersion (synonyms) = grounded, high dispersion (unrelated) = hallucination

## Version History (Algorithm Evolution)

| Version | Config Flags | Description |
|---------|-------------|-------------|
| **v3.1** | `enable_unified_gating=False, enable_semantic_dispersion=False` | Pure Authority Flow (paper baseline) |
| **v7.0** | `enable_unified_gating=True, enable_semantic_dispersion=False` | Adds context-dependent gating |
| **v8.0** | `enable_unified_gating=True, enable_semantic_dispersion=True` | **Default.** Adds semantic dispersion |
| **v9.0** | `AGSARConfig.from_preset("qa")` | Task-specific parameter presets |

### v9.0 Task-Adaptive Mode
Use `AGSARConfig.from_preset()` for task-optimized parameters:
- **QA**: Conservative aggregation (percentile_10), T=1.2
- **RAG**: Trust context more (parametric_weight=0.3), T=1.5
- **Summarization**: Higher dispersion_k=10 for long-form, T=2.5
- **Attribution**: Moderate conservative (percentile_25), T=2.0

### Archived Features (_legacy/)
Experimental features from v10.0-v13.0 (Truth Vector, JEPA Predictor, Hybrid Controller)
are archived in `_legacy/`. See `_legacy/README.md` for details.

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
│   ├── ragtruth.py            # RAGTruth loader
│   └── fava.py                # FAVA attribution loader
├── methods/
│   ├── base.py                # UncertaintyMethod ABC
│   ├── agsar_wrapper.py       # AG-SAR method (v8.0/v9.0)
│   ├── logprob.py             # Log-probability baseline
│   ├── entropy.py             # Entropy baseline
│   ├── selfcheck.py           # SelfCheck baseline
│   ├── eigenscore.py          # EigenScore baseline
│   └── llm_check.py           # LLM-Check baselines (NeurIPS 2024)
├── configs/
│   ├── schema.py              # Pydantic config validation
│   ├── agsar_only.yaml        # AG-SAR v8.0 only
│   ├── agsar_task_adaptive.yaml    # AG-SAR v9.0
│   └── unified_eval.yaml      # Full benchmark suite
└── analysis/
    └── benchmark_latency.py   # Zero-latency verification
```

**Paper Experiments** (run via `./reproduce_paper.sh` or individually with `python -m experiments.main --config <config>`):

| Config | Paper Section | Description |
|--------|---------------|-------------|
| `00_ci_smoke_test.yaml` | - | CI validation (fast sanity check) |
| `01_main_sota.yaml` | Table 1 | SOTA comparison on HaluEval QA |
| `02_scaling_law.yaml` | Figure 2 | Scaling to Llama-3.1-70B |
| `03_generalization.yaml` | Table 2 | RAGTruth generalization |
| `unified_eval.yaml` | Full Suite | All datasets with all methods |

## Known Issues

1. **Transformers version sensitivity**: Pin to `<4.45.0` for attention hook compatibility.

2. **Single-token responses**: `var()` warning for single-token responses in variance computation.
