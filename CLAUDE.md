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

## Commands

### Installation
```bash
pip install -r requirements.txt
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
python -m experiments.main --config experiments/configs/01_main_sota.yaml
python -m experiments.main --config experiments/configs/03_generalization.yaml --output-dir results/custom
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

- `main.py` - CLI entry point
- `configs/` - 43 YAML experiment specifications
- `methods/` - 8 baseline implementations (LogProb, Entropy, SelfCheckGPT, Semantic Entropy, EigenScore, LLMCheck)
- `data/` - Dataset loaders (HaluEval, RAGTruth, TruthfulQA, FAVA)
- `analysis/` - Plotting and metrics computation

## Critical Constraints

- **transformers version**: Must be `>=4.40.0,<4.45.0`. Version 4.45+ breaks attention hooks.
- **Triton**: Linux-only. Falls back to PyTorch on Mac/Windows automatically.
- **Flash Attention**: Requires Linux + CUDA 12+ for H100 optimizations.

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
