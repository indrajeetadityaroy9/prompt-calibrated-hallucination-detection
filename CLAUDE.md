# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a research framework for single-pass inference-time hallucination detection in LLMs via internal attention graph analysis. It achieves O(N) memory complexity through matrix-free eigenvector centrality computation.

**Core equation:**
```
Uncertainty(t) = 1 - Authority(t)
Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)
Trust(t) = 1 - Dispersion(t) × (1 + λ × Varentropy(t))
```

## Build & Development Commands

```bash
# Installation
pip install -e ".[dev]"          # Development (testing, linting)
pip install -e ".[eval]"         # Evaluation framework (benchmarks)
pip install -e ".[h100]"         # H100 acceleration (Linux only)
pip install -e ".[all]"          # Everything

# Testing
pytest tests/                    # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests
pytest tests/unit/test_config.py -v  # Single test file

# Linting & Formatting
black ag_sar/ experiments/       # Format code (100 char line length)
ruff check ag_sar/ experiments/  # Style check
mypy ag_sar/                     # Type checking

# Run experiments
python -m experiments.main --config experiments/configs/validation/smoke_test.yaml
python -m experiments.main --config path/to/config.yaml --dry-run  # Preview config
python -m experiments.main --config path/to/config.yaml --seed 123 --deterministic
```

## Architecture

### Core Package (`ag_sar/`)

**Entry Point:** `engine.py` - The `AGSAR` class orchestrates the full pipeline:
```python
agsar = AGSAR(model, tokenizer, config=AGSARConfig())
agsar.calibrate_on_prompt(prompt)  # Optional adaptive threshold tuning
uncertainty = agsar.compute_uncertainty(prompt, response)
```

**Three-Pillar Measures** (`measures/`):
- `authority.py` - Authority flow computation (recursive attention from prompt tokens)
- `semantics.py` - Semantic dispersion (top-k prediction consistency)
- `stability.py` - Stability gating (MLP-attention agreement)
- `entropy.py` - Token entropy and varentropy

**Model Interaction** (`modeling/`):
- `hooks.py` - `ModelAdapter` extracts attention via forward hooks (monkey-patches attention modules)
- `adapters.py` - Architecture-specific adapters for RoPE and GQA handling
- Supports: GPT-2, Llama, Qwen, Mistral, Mixtral

**Low-Level Ops** (`ops/`):
- `torch_functional.py` - PyTorch fallback implementations
- `triton_*.py` - Triton GPU kernels (Linux only, auto-fallback on Mac/Windows)
- Environment: `AG_SAR_USE_TORCH=1` forces PyTorch backend

### Experiments Framework (`experiments/`)

- `main.py` - Unified CLI entry point
- `configs/` - YAML experiment configurations (organized by purpose: benchmarks/, ablations/, scaling/, etc.)
- `configs/schema.py` - Pydantic config validation
- `evaluation/` - Benchmark engine and determinism utilities
- `methods/` - Baseline implementations (LogProb, SelfCheck, EigenScore, Semantic Entropy, etc.)
- `data/` - Dataset loaders (HaluEval, RAGTruth, TruthfulQA, WikiText, FAVA)

## Critical Dependencies

**Pinned versions that must not be changed without careful testing:**
- `transformers>=4.40.0,<4.45.0` - **>=4.45 breaks attention hooks**
- `numpy>=1.20.0,<2.0` - 2.x breaks pandas/scipy compatibility
- `scipy>=1.10,<1.14` - 1.14+ breaks linesearch imports

**Platform-specific:**
- Triton kernels are Linux-only; Mac/Windows automatically use PyTorch fallback
- `flash-attn` (H100 optimization) requires Linux + CUDA 12+

## Key Concepts

1. **Attention hooks are fragile** - Changes to `modeling/hooks.py` or `modeling/adapters.py` require testing across multiple architectures (GPT-2, Llama, Qwen, Mistral)

2. **Adaptive calibration** - All thresholds are auto-tuned from prompt statistics via `calibrate_on_prompt()`. The `sigma_multiplier` config controls sensitivity.

3. **Hardware tiers:**
   - H100: Full optimization (TF32 + Flash attention)
   - Ampere+: BFloat16 + TF32
   - Older/Mac/CPU: Float16 or Float32 fallback

4. **Config is minimal by design** - `AGSARConfig` exposes only essential tuning knobs (`semantic_layers`, `varentropy_lambda`, `sigma_multiplier`). Most behavior is auto-derived.

## Testing Conventions

- `tests/unit/` - Module-level tests (one file per module)
- `tests/integration/` - Full pipeline and behavior tests
- `tests/experiments/` - Data loader and metrics validation
- All tests use pytest fixtures from `conftest.py` files
