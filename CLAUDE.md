# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot hallucination detection system for LLaMA 3.1. It computes internal signals during text generation and aggregates them via prompt-anchored normalization + Noisy-OR for polarity-stable risk scoring. This project targets ICML submission and focuses on detecting **contextual hallucinations** in retrieval-augmented generation (RAG) systems.

The codebase is mid-refactor toward a DSG (Decoupled Spectral Grounding) architecture — see TODO comments in `ag_sar/config.py` and the placeholder `ag_sar/icml/` module.

## Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Hugging Face Authentication
Gated Llama 3.1 models require authentication:
```bash
huggingface-cli login
# Or set HF_TOKEN environment variable
```

### Run Tests
```bash
python -m pytest tests/                              # All tests
python -m pytest tests/test_full_pipeline.py         # Single test file
python -m pytest tests/test_full_pipeline.py -k "test_name"  # Single test
```

### Run Benchmarks
```bash
python experiments/main.py --config experiments/configs/benchmarks/sota_confirmation.yaml
```

### Pilot Evaluation
```bash
python scripts/run_pilot_halueval.py      # HaluEval dataset
python scripts/run_pilot_ragtruth.py      # RAGTruth dataset
python scripts/run_pilot_rigorous.py      # Comprehensive evaluation
python scripts/eval_truthfulqa.py         # TruthfulQA evaluation
```

### Verification
```bash
python tests/verify_polarity.py           # Verify signal polarity (higher = riskier)
```

## Architecture

### Pipeline Flow
```
Input → Model(Hooks) → Signal Extraction → Prompt-Anchored Normalization → Noisy-OR Fusion → Risk Score
```

1. **Hook System** (`ag_sar/hooks.py`): Registers PyTorch forward hooks on transformer layers to capture hidden states at 3 points per layer (pre-MLP, post-MLP, post-layer). `EphemeralHiddenBuffer` stores these transiently, cleared after each token. `PrefillContextHook` captures context representations during the prefill pass for context grounding.

2. **Signal Extraction** (`ag_sar/signals/`): Computes per-token signals from captured hidden states. All signals inherit from `BaseSignal` and must implement `compute()`. Currently implemented:
   - `topk_jsd.py` — `CandidateJSDSignal`: MLP-induced distribution shift via Jensen-Shannon divergence on top-k candidate set
   - `context_grounding.py` — `ContextGroundingSignal`: SVD-based projection measuring how well output representations are "explained" by context subspace (flagship mechanistic approach)

3. **Aggregation** (`ag_sar/aggregation/`):
   - `prompt_anchored.py` — `PromptAnchoredAggregator`: Normalizes signals using prompt-token statistics as baseline (z-scores), maps to independent probabilities via sigmoid, fuses via Noisy-OR
   - `span_merger.py` — `SpanMerger`: Groups adjacent high-risk tokens into contiguous risk spans

4. **Classification**: Token → Span → Response level risk scoring (response risk = Noisy-OR over all token risks)

### Core Entry Points

- `ag_sar/engine.py`: **AGSAR class** — main entry point. Also contains `CandidateSetManager` for top-k candidate tracking
- `ag_sar/config.py`: `DetectorConfig` (configuration), `DetectionResult` (output), `TokenSignals`, `SpanRisk` dataclasses
- `ag_sar/numerics.py`: Numerically-safe primitives (`safe_softmax`, `safe_jsd`, `max_cosine_similarity`)
- `ag_sar/ops/triton_kernels.py`: Optional Triton kernel implementations for GPU acceleration

### Evaluation (`ag_sar/evaluation/`)

- `runner.py`: `EvaluationRunner` orchestrates dataset loading, evaluation mode selection, and metric computation
- `modes.py`: `ForcedDecodingEvaluator` (stepwise with ground-truth tokens) and `GenerationEvaluator` (argmax decoding)
- `metrics.py`: `compute_metrics()` returns AUROC, AUPRC, span F1, ECE, Brier, AURC, risk@coverage, TPR@FPR
- `token_aligner.py`: Aligns model tokenization with dataset span annotations
- `data/`: Dataset loaders — `HaluEvalLoader`, `RAGTruthLoader`, `TruthfulQALoader`

### ICML Module (`ag_sar/icml/`)

Currently a placeholder. DSG (Decoupled Spectral Grounding) detector to be added here. See `DSG_ALGORITHM.md` for the planned algorithm.

## Configuration

Config-driven via `DetectorConfig` in `ag_sar/config.py`:
- `layer_subset`: `"last_third"` | `"last_quarter"` | `"all"` | `List[int]`
- `candidate_topk`: Size of candidate set (default 128)
- `prompt_anchored_signals`: Signals for aggregation (default `("jsd",)`)
- `use_prompt_anchored`: Toggle prompt-anchored aggregation (default True)
- `default_eval_mode`: `"forced_decoding"` | `"generation"`
- `multi_gpu_mode`: `"compatible_only"` | `"allow_transfer"` (for 70B models with `device_map='auto'`)

## Key Design Principles

- **Polarity-stable**: All signals follow convention that higher values = higher risk
- **Candidate-set approach**: Signals computed on top-k tokens for efficiency
- **Prompt-anchored normalization**: Uses prompt-token statistics as baseline (z-scores), making scores comparable across inputs
- **Zero-shot**: No training required
- **Single-pass**: No multi-sample generation needed
- **Two modes**: Deployment mode (single-pass with KV-cache) and evaluation mode (forced decoding with ground-truth tokens)

## Context Grounding (Flagship Mechanism)

Located in `ag_sar/signals/context_grounding.py`.

**Core insight**: If a model is grounded in context, its output representations should be "explainable" by the context subspace. Hallucinations produce representations that diverge from the context.

```
Grounding Score = ||proj_context(h_t)|| / ||h_t||
Hallucination Risk = 1 - Grounding Score
```

AG-SAR targets **contextual hallucinations** (output diverges from provided context), not parametric hallucinations (false statements from pretrained knowledge).
