# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot hallucination detection system for LLaMA 3.1. It computes internal signals during text generation and aggregates them via prompt-anchored normalization + Noisy-OR for polarity-stable risk scoring. This project targets ICML submission and focuses on detecting **contextual hallucinations** in retrieval-augmented generation (RAG) systems.

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
python -m pytest tests/
python -m pytest tests/test_full_pipeline.py
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

1. **Hook System** (`ag_sar/hooks.py`): Captures hidden states at 3 points per transformer layer

2. **Signal Extraction** (`ag_sar/signals/`): Computes per-token signals from hidden states

3. **Aggregation** (`ag_sar/aggregation/`): Prompt-anchored normalization + Noisy-OR fusion

4. **Classification**: Token → Span → Response level risk scoring

### Core Entry Points

- `ag_sar/engine.py`: **AGSAR class** - main entry point
- `ag_sar/config.py`: DetectorConfig dataclass
- `ag_sar/icml/`: ICML-ready ensemble implementations

### Signal Categories (`ag_sar/signals/`)

- **Core**: `topk_jsd.py`, `lci.py`, `varlogp.py`, `uncertainty.py` (entropy, inv_margin)
- **SOTA**: `semantic_entropy.py`, `inside.py` (EigenScore), `lsd.py`, `internal_se.py`
- **Context**: `context_support.py`, `context_grounding.py` (flagship mechanistic approach)

### Aggregation (`ag_sar/aggregation/`)

- `prompt_anchored.py`: **PromptAnchoredAggregator** - ICML-ready with z-scores and Noisy-OR
- `span_merger.py`: Merge adjacent high-risk tokens into spans

### Evaluation (`ag_sar/evaluation/`)

- `runner.py`: EvaluationRunner orchestrates evaluation
- `modes.py`: ForcedDecodingEvaluator and GenerationEvaluator
- `metrics.py`: AUROC, AUPRC, span F1, ECE, Brier, AURC metrics
- `data/`: Dataset loaders (RAGTruth, HaluEval, TruthfulQA, FaithEval)

### ICML Module (`ag_sar/icml/`)

- `robust_ensemble.py`: **RobustEnsemble** - cross-task generalizable (recommended)
- `grounding_detector.py`: **ICMLContextGrounding** - pure SVD-based context grounding

## Configuration

Config-driven via `DetectorConfig` in `ag_sar/config.py`:
- `layer_subset`: `"last_third"` | `"last_quarter"` | `"all"` | `List[int]`
- `candidate_topk`: Size of candidate set (default 128)
- `prompt_anchored_signals`: Signals for aggregation
- `eigenscore_enabled`, `semantic_entropy_enabled`, `ise_enabled`, `lsd_enabled`: SOTA toggles

## Usage Example

```python
from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig

config = DetectorConfig(
    layer_subset="last_third",
    candidate_topk=128,
)
detector = AGSAR(model, tokenizer, config)
result = detector.generate("Your question", context="Optional context", max_new_tokens=100)
# result.generated_text, result.token_risks, result.response_risk, result.risky_spans
```

### ICML Ensemble Usage

```python
from ag_sar.icml import RobustEnsemble

ensemble = RobustEnsemble(model, tokenizer)
result = ensemble.compute_risk(hidden_states, logits, prompt_len)
# result.response_risk, result.token_risks
```

## Key Design Principles

- **Polarity-stable**: All signals follow convention that higher values = higher risk
- **Candidate-set approach**: Signals computed on top-k tokens for efficiency
- **Prompt-anchored normalization**: Uses prompt statistics as baseline
- **Zero-shot**: No training required
- **Single-pass**: No multi-sample generation needed

## Context Grounding (Flagship Mechanism)

Located in `ag_sar/signals/context_grounding.py` and `ag_sar/icml/grounding_detector.py`.

### Core Insight
If a model is grounded in context, its output representations should be "explainable" by the context subspace. Hallucinations produce representations that diverge from the context.

### Mathematical Foundation
```
Grounding Score = ||proj_context(h_t)|| / ||h_t||
Hallucination Risk = 1 - Grounding Score
```

### Two Types of Hallucination

1. **Contextual Hallucination** (AG-SAR excels): Output diverges from provided context
2. **Parametric Hallucination** (Out of scope): Model generates false statements from pretrained knowledge
