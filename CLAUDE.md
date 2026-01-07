# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency hallucination detection framework for LLMs. It analyzes internal attention graph structure without external semantic models or multiple forward passes. Optimized for **extrinsic hallucination detection** (unfaithful to source context) in RAG settings.

**Version**: 0.4.0 (v8.0 SOTA Gold Master for ICML/NeurIPS submission)

## Research Abstract

### Core Problem

Large Language Models generate fluent text but frequently produce **hallucinations**—statements that appear plausible but are unfaithful to the provided source context. Existing detection methods suffer from either:
- **High latency**: Requiring multiple forward passes or external verifier models
- **Semantic blindness**: Relying solely on token probabilities without understanding meaning
- **Wrong target**: Detecting intrinsic hallucinations (factual errors) rather than extrinsic ones (source unfaithfulness)

### Key Insight

The attention mechanism itself encodes whether a generated token is **grounded in context** or **fabricated from parametric memory**. By tracing how "authority" (information provenance) flows through attention layers, we can distinguish faithful generation from hallucination—without any external models or repeated inference.

### Three-Pillar Mechanism

**1. Authority Flow**
Information has a source. When a model generates a token, that token's "authority" should trace back to the prompt/context if grounded, or emerge from internal parameters if hallucinated. AG-SAR computes this recursively:
```
Authority(token) = [attention to prompt tokens] + [attention to prior generated tokens × their authority]
```
Tokens with low prompt-derived authority are suspicious.

**2. Unified Gating**
Not all parametric knowledge is hallucination—sometimes the model legitimately uses learned facts. The gating mechanism dynamically balances:
- Context authority (from Authority Flow)
- Parametric confidence (from model's internal state)

The gate opens toward context when the model attends heavily to source material, and toward parametric memory when generating common knowledge.

**3. Semantic Dispersion**
Raw token probability is misleading—a model might be "uncertain" between synonyms (US, USA, America) or between unrelated alternatives (Paris, London, Tokyo). AG-SAR measures **semantic consistency** of top-k predictions:
- Low dispersion (synonyms) → Grounded
- High dispersion (scattered meanings) → Hallucination

### Technical Contributions

- **O(N) Memory**: Matrix-free eigenvector centrality avoids O(N²) attention matrices
- **Zero Latency**: Single forward pass, no external models
- **Architecture Agnostic**: Works across GPT-2, Llama, Mistral, Qwen via unified hooking
- **Task Adaptivity**: Preset configurations for QA, RAG, summarization, attribution

### Philosophical Position

AG-SAR treats hallucination detection as an **information provenance problem**, not a factuality problem. The question isn't "is this true?" but rather "does this come from the context the user provided?" This framing is particularly suited for RAG systems where faithfulness to retrieved documents matters more than world knowledge.

## Usage

```python
from ag_sar import AGSAR, AGSARConfig
config = AGSARConfig()  # Uses v8.0 defaults
agsar = AGSAR(model, tokenizer, config)

# Compute uncertainty score (0=confident, 1=uncertain)
score = agsar.compute_uncertainty(prompt, response)

# Binary hallucination detection
is_violation, confidence, details = agsar.detect_context_violation(prompt, response)
```

## Commands

### Installation
```bash
pip install -e ".[all]"              # Full install with all extras
pip install -e ".[eval]"             # Evaluation dependencies only
pip install -e ".[h100]"             # H100 optimizations (Linux only)
```

### Testing
```bash
pytest tests/                        # All tests (96 total)
pytest tests/unit/                   # Unit tests only
pytest tests/integration/            # Integration tests
pytest tests/unit/test_ops.py -k "test_authority"  # Specific test pattern
```

### Code Quality
```bash
black src/ tests/                    # Format code
ruff check src/ tests/               # Lint
mypy src/ag_sar/                     # Type check
```

### Running Experiments
```bash
python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml
python -m experiments.main --config experiments/configs/generalization/cross_dataset.yaml --output-dir results/custom
bash reproduce_paper.sh              # Full ICML 2025 reproduction (6 stages)
bash run_final_matrix.sh             # Batch experiment runner
```

### Demo
```bash
python examples/minimal_demo.py      # Quick demo with GPT-2
```

## Architecture

### Core Package (`src/ag_sar/`)

Three-component detection mechanism:

1. **Authority Flow** (`measures/authority.py`) - Tracks signal provenance from prompt to response tokens via recursive attention recharge
2. **Unified Gating** (`measures/stability.py`) - Context-dependent trust switching between context authority and parametric confidence
3. **Semantic Dispersion** (`measures/semantics.py`) - Measures top-k prediction consistency (synonyms = grounded, unrelated = hallucination)

**Master Equation**: `A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × parametric_weight`

### Key Modules

- `engine.py` - AGSAR orchestrator, main entry point (`AGSAR.compute_uncertainty`)
- `config.py` - AGSARConfig with 80+ parameters
- `modeling/hooks.py` - Unified attention extraction hooks
- `modeling/adapters.py` - Architecture-specific adapters (GPT-2, Llama, Mistral, Qwen)
- `ops/torch_functional.py` - Pure PyTorch O(N) implementation
- `ops/triton_kernels.py` - Triton acceleration (Linux only, auto-fallback)
- `presets/` - Task-adaptive configs (qa.yaml, rag.yaml, summarization.yaml)

### Experiments Framework (`experiments/`)

- `main.py` - CLI entry point for running benchmarks
- `evaluation/` - Benchmark engine, metrics (AUROC, AUPRC, ECE), logging infrastructure
- `configs/` - YAML experiment specifications organized by purpose:
  - `benchmarks/` - Main SOTA comparisons
  - `ablations/` - Component ablation studies
  - `scaling/` - Model size sweeps
  - `generalization/` - Cross-dataset evaluation
  - `architecture/` - Architecture tests (Qwen, MoE, FAVA)
  - `mechanism_analysis/` - Mechanism sweeps and knowledge conflict
  - `baselines/` - Parallel baseline comparisons
  - `validation/` - CI smoke tests
- `methods/` - Baseline implementations (LogProb, Entropy, SelfCheckGPT, Semantic Entropy, EigenScore, SAPLMA, LLMCheck variants)
- `data/` - Dataset loaders: HaluEval, RAGTruth, TruthfulQA, WikiText, FAVA
- `scripts/` - Utility scripts for benchmarking, dataset prep, and result formatting
- `analysis/` - Plotting and metrics computation

## Critical Constraints

- **transformers version**: Must be `>=4.40.0,<4.45.0`. Version 4.45+ breaks attention hooks.
- **Triton**: Optional acceleration. Falls back to PyTorch automatically if unavailable.
- **Flash Attention**: Requires CUDA 12+ for H100 optimizations.

## Inference Pipeline

```
Input: (prompt, response)
  ↓ Tokenization
  ↓ Forward pass with attention hooks (extract Q/K/V)
  ↓ Authority Flow (recursive recharge, track prompt authority)
  ↓ Semantic Dispersion (top-k consistency analysis)
  ↓ Unified Gating (blend authority + parametric confidence)
  ↓ Aggregation (mean/percentile/median + temperature calibration)
Output: (is_hallucination, confidence, details)
```

## Configuration Presets

Task-specific tuning in `src/ag_sar/presets/`:
- **QA**: Conservative (percentile_10, T=1.2)
- **RAG**: Context-focused (mean, T=1.5, low parametric_weight)
- **Summarization**: Paraphrase-tolerant (percentile_25, T=2.5, centroid_variance)
- **Attribution**: Fine-grained (percentile_25, T=2.0)

## Supported Models

GPT-2, Llama-3/3.1/3.2 (8B, 70B), Mistral (7B, 8x7B), Qwen (7B, 72B) - any model with standard attention interface.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | HuggingFace token for gated models (Llama, Mistral) | Required for gated models |
| `AG_SAR_USE_TORCH` | Set to `1` to force PyTorch fallback (skip Triton) | `0` (use Triton if available) |
| `PYTHONHASHSEED` | Set by reproducibility utilities for hash determinism | Set by `set_global_seed()` |
| `CUDA_VISIBLE_DEVICES` | Control which GPUs are visible | All GPUs |

## Reproducibility

For deterministic, reproducible experiments:

```bash
# Run with explicit seed and deterministic mode
python -m experiments.main \
    --config experiments/configs/benchmarks/main_sota.yaml \
    --seed 42 \
    --deterministic
```

The `--deterministic` flag enables:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` where supported

**Note**: Deterministic mode may reduce performance by 10-20%.

### Sampling Method Seeds

All sampling-based methods (SelfCheck, EigenScore, SemanticEntropy) accept a `seed` parameter for reproducibility. Seeds are set before each generation call.

## Hardware Requirements

### Minimum
- GPU: Any CUDA-capable GPU with 8GB+ VRAM
- CPU: 4+ cores
- RAM: 16GB

### Recommended (for full reproduction)
- GPU: 2x NVIDIA H100 80GB (or A100 40GB)
- CPU: 24+ cores
- RAM: 256GB

### Model-Specific Requirements

| Model | VRAM Required | Device Map |
|-------|---------------|------------|
| GPT-2 | 2GB | Single GPU |
| Llama-3-8B | 16GB (bf16) | Single GPU |
| Llama-3-70B | 140GB (bf16) | 2x H100 (balanced) |
| Mixtral-8x7B | 90GB (bf16) | 2x H100 (balanced) |

### H100 Optimizations

Automatically enabled on H100/A100:
- TF32 precision (3x faster matmul)
- Flash Attention 2 (when available)
- cuDNN benchmark mode (unless `--deterministic`)

To check if optimizations are active:
```python
from ag_sar.utils import is_tf32_enabled, is_h100
print(f"TF32: {is_tf32_enabled()}, H100: {is_h100()}")
```
