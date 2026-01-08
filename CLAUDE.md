# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a single-pass inference-time hallucination detection framework for LLMs. It detects when generated text contradicts provided source context by analyzing internal attention structure without external models or multiple forward passes. The project is structured as a research library with full benchmarking infrastructure.

## Commands

### Testing
```bash
pytest tests/                          # All tests
pytest tests/unit/                     # Unit tests only
pytest tests/integration/              # Integration tests only
pytest tests/unit/test_config.py -v    # Single test file
pytest -k "test_name"                  # Run specific test by name
pytest tests/ --cov=src/ag_sar         # With coverage
```

### Code Quality
```bash
black src/ experiments/ tests/ --line-length=100
ruff check src/ experiments/ tests/ --select E,F,W,I,UP --line-length=100
mypy src/ag_sar/ --python-version=3.9
```

### Running Experiments
```bash
python -m experiments.main --config experiments/configs/benchmarks/reproduce_main_results.yaml
python -m experiments.main --config <config.yaml> --dry-run   # Validate without running
python smoke_test.py                                          # Installation verification
./reproduce_paper.sh                                          # Full paper reproduction
```

### Installation
```bash
pip install -e ".[all]"    # Full install with dev/eval/h100 deps
pip install -e .           # Core only
```

## Architecture

### Core Library (`src/ag_sar/`)

The library implements a three-pillar mechanism executed in a single forward pass:

1. **Authority Flow** (`measures/authority.py`): Recursive attention tracing that measures how much response attention derives from prompt tokens (0=parametric memory, 1=grounded in context)

2. **Unified Gating** (`measures/stability.py`): Dynamically balances context-derived authority vs parametric confidence based on MLP-attention agreement

3. **Semantic Dispersion** (`measures/semantics.py`): Measures embedding-space consistency of top-k predictions (low=semantically coherent, high=hallucinating)

Key modules:
- `engine.py`: AGSAR orchestrator combining all mechanisms
- `config.py`: AGSARConfig dataclass with validated parameters
- `modeling/adapters.py`: Version-aware transformer adapters (GPT-2, Llama, Mistral, Qwen)
- `modeling/hooks.py`: Attention hook registration and Q/K/V capture
- `ops/torch_functional.py`: Vectorized O(N) tensor operations
- `ops/triton_kernels.py`: Optional H100 GPU kernels

### Experiments Framework (`experiments/`)

- `main.py`: Entry point for all experiments
- `configs/`: YAML configs organized by purpose (benchmarks/, baselines/, ablations/, scaling/, etc.)
- `configs/schema.py`: Pydantic validators for all config fields
- `evaluation/engine.py`: BenchmarkEngine with streaming JSONL output
- `evaluation/metrics.py`: AUROC, AUPRC, F1, ECE with bootstrap CI
- `methods/`: UncertaintyMethod implementations (AG-SAR + baselines)
- `data/`: Dataset loaders (HaluEval, RAGTruth, TruthfulQA, FAVA, WikiText)

### Data Flow

```
Input (prompt, response)
  → AGSAR.compute_uncertainty()
  → Single forward pass with attention hooks
  → Three measures computed in parallel
  → Aggregated to sequence-level score ∈ [0, 1]
```

## Critical Constraints

- **transformers version**: Must use >=4.40.0,<4.45.0 (4.45+ breaks attention hooks)
- **Streaming mode**: Default `streaming_mode=True` enables O(N) memory and Flash Attention compatibility by capturing only the current token's attention row
- **Cleanup**: Always call `agsar.cleanup()` before creating new AGSAR instances to remove hooks
- **Thread safety**: AGSAR instances are NOT thread-safe; each thread needs its own instance

## Key Files by Task

| Task | Files |
|------|-------|
| Add new measure | `src/ag_sar/measures/` + update `__init__.py` |
| Add baseline method | `experiments/methods/` + update `__init__.py` |
| Add dataset | `experiments/data/` + update `schema.py` |
| Debug config | `experiments/configs/schema.py` |
| Paper notation | `paper/notation_mapping.md` (code ↔ paper symbols) |

## Config System

All experiments use YAML configs validated by Pydantic. The canonical config for paper results is `experiments/configs/benchmarks/reproduce_main_results.yaml`. Key AGSARConfig parameters:

- `semantic_layers`: Number of final layers to analyze (default: 4)
- `prompt_authority`: Initial authority assigned to prompt tokens (default: 1.0)
- `enable_unified_gating`: Whether to use Agreement Gate (default: true)
- `enable_semantic_dispersion`: Whether to compute semantic consistency (default: true)
- `aggregation_method`: How to combine per-token scores (default: percentile_10; options: mean, min, percentile_10, percentile_25, importance_weighted)
- `dispersion_method`: Semantic dispersion algorithm (default: nucleus_variance; options: top1_projection, centroid_variance, nucleus_variance)

## Usage Patterns

### Basic Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

agsar = AGSAR(model, tokenizer)
try:
    uncertainty = agsar.compute_uncertainty(prompt="Context...", response="Response...")
finally:
    agsar.cleanup()  # Always cleanup to remove hooks
```

### With Prompt Calibration (Model-Agnostic Thresholds)
```python
config = AGSARConfig(enable_prompt_calibration=True)
agsar = AGSAR(model, tokenizer, config)

# Calibrate baseline statistics from prompt
baseline = agsar.calibrate_on_prompt(prompt)
# Now compute_uncertainty uses adaptive Z-score thresholds
uncertainty = agsar.compute_uncertainty(prompt, response)
```
