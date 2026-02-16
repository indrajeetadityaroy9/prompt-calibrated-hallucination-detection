# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot, training-free hallucination detection system for retrieval-augmented LLMs. It intercepts transformer internals during generation to extract five mechanistically-grounded signals and fuses them via entropy-gated aggregation into per-token risk scores — no training data, calibration, or external models required.

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Run all tests
pytest

# Run a single test
pytest tests/test_full_pipeline.py

# Run evaluation on QA datasets
python scripts/run_dsg_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 100
python scripts/run_dsg_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --all --n-samples 100
```

Requires Python >= 3.11, PyTorch >= 2.5.1, Transformers >= 4.57.0. Tests use SmolLM-135M-Instruct as a lightweight model.

## Architecture

### Pipeline Flow

The entry point is `DSGDetector` in `ag_sar/icml/dsg_detector.py`. It runs in two phases:

1. **Prefill** (once per input): Identifies copying heads via Otsu thresholding on attention affinity, computes context subspace (SVD + Marchenko-Pastur rank), precomputes reasoning subspace from the unembedding matrix, derives prompt center (for CGD) from non-context prompt tokens, sets magnitude tau (for DPS gate) from prefill norms, and collects prompt statistics (mean/MAD-sigma) from a tail window of size sqrt(prompt_length). Hidden state hooks are installed once and persist across both phases.

2. **Generation** (per token): Computes five signals, normalizes via prompt-anchored z-scoring (or direct mode for CUS), fuses with entropy-gated weighted mean for per-token risk (span detection), computes response-level risk via signal-first aggregation (mean of per-signal response probabilities), and merges risky spans via Tukey fences.

### Five Signals

Each targets a distinct failure mode along the transformer's causal chain:

- **CUS (Context Utilization Score)** — `ag_sar/signals/copying_heads.py`: Lookback ratio bimodality across all attention heads. Measures whether heads show healthy separation (some attend to context, others don't) vs uniform attention. CUS = 1 - Otsu coefficient. Range [0,1], higher = unimodal = riskier. Uses direct probability mapping (no z-score).

- **POS (Parametric Override Score)** — `ag_sar/signals/topk_jsd.py`: Candidate-set JSD between pre-FFN and post-FFN logit distributions, with directional override (measures if MLP shift avoids context subspace). Detects when MLP overrides what attention found. Uses z-score normalization.

- **DPS (Dual-Subspace Projection Score)** — `ag_sar/signals/context_grounding.py`: Projects hidden states onto context subspace (from prefill SVD) and reasoning subspace (bottom singular vectors of unembedding matrix). DPS = s_rsn / (s_ctx + s_rsn), with magnitude gating that dampens toward 0.5 when ||h_centered|| is small. Middle third of layers. Uses z-score normalization.

- **DoLa (Layer-Contrast Score)** — `ag_sar/signals/topk_jsd.py`: log P_final(token) - log P_premature(token), where premature layer = argmax JSD over early layers. High = model added factual content in late layers. Uses z-score normalization.

- **CGD (Context-Grounding Direction)** — `ag_sar/signals/context_grounding.py`: Cosine between (h_gen - prompt_center) and (context_center - prompt_center). Measures whether generation moves toward or away from context. (1 - cos_sim)/2 in [0,1], higher = away from context = riskier. Uses z-score normalization.

### Hook System

`ag_sar/hooks.py` captures three residual stream points per layer per token: `h_resid_attn` (post-attention), `h_mlp_in` (post-norm, pre-MLP), `h_resid_mlp` (post-MLP). Uses an ephemeral buffer design — cleared after each token's signals are computed to prevent memory accumulation.

### Signal Registry (`ag_sar/config.py`)

`SIGNAL_REGISTRY` is the single source of truth for each signal's properties:

- `NormMode`: `DIRECT` (value IS the probability, used by CUS) or `ZSCORE` (prompt-anchored z-score -> sigmoid, used by POS/DPS/DoLa/CGD)
- `SignalMetadata`: name, norm_mode, bounded (whether signal is in [0,1]), neutral value, higher_is_riskier
- `SignalMetadata.fallback_stats()`: Registry-derived fallback when calibration fails — bounded signals fall back to direct mode (no model-specific constants), unbounded signals fall back to wide sigma (uninformative)
- `SignalMetadata.sigma_floor()`: Data-derived floor = 10% of observed sigma

### Shared Calibration (`ag_sar/calibration.py`)

- `get_layer_indices()`: Layer subset selection from config
- `build_input()`: Tokenize context + question into model input
- `adaptive_window()`: sqrt(prompt_len) clamped to [16, prompt_len//2]
- `self_calibrate()`: Full self-calibration pipeline — computes prompt-anchored statistics for all active signals from prefill hidden states, with registry-derived fallbacks

### Aggregation Pipeline (`ag_sar/aggregation/`)

- `prompt_anchored.py`: Registry-aware signal normalization (direct vs z-score from `SIGNAL_REGISTRY`), entropy-gated fusion for per-token risk (w_i = (1-H_i)^2, uninformative signals at p=0.5 get weight 0), and signal-first response-level aggregation (mean of per-signal response probabilities)
- `span_merger.py`: Groups contiguous high-risk tokens into spans via Tukey fence (Q3 + 0.5 * IQR)

### Key Data Structures (`ag_sar/config.py`)

- `DSGConfig`: Layer selection strategy (`"all"`, `"last_third"`, `"last_quarter"`, or `List[int]`)
- `DSGTokenSignals`: Per-token CUS/POS/DPS/DoLa/CGD values
- `DetectionResult`: Generated text, per-token signals and risks, risky spans, response-level risk, flagged status

### Evaluation (`ag_sar/evaluation/`)

- `metrics.py`: AUROC, AUPRC, TPR@FPR, ECE, Brier, AURC/E-AURC, Risk@Coverage, span precision/recall with IoU, bootstrap CI
- Primary evaluation script: `scripts/run_dsg_evaluation.py`

### Numerics (`ag_sar/numerics.py`)

Safe softmax (max-subtraction + clamping), JSD in bits bounded [0,1], Otsu thresholding (zero-parameter bimodal splitting), and MAD-based robust sigma estimation (1.4826 * median(|x - median(x)|)).

## Design Principles

- **Zero-parameter**: Every threshold derived from input data, model architecture, or information-theoretic principles (Otsu, Marchenko-Pastur, order statistics). No learned weights or hyperparameters.
- **Prompt-anchored calibration**: Signal statistics estimated from the prefill pass itself; generation values z-scored against these.
- **Entropy-gated fusion**: Uninformative signals (p=0.5) get weight 0 via binary entropy gating with kappa=2 quadratic suppression.
- **Robust statistics**: MAD replaces std-dev throughout; sigma floors prevent numerical issues.
- **Candidate-set efficiency**: POS/DoLa use adaptive top-k (95% cumulative probability mass) + previous top-k + emitted token, not full vocabulary.
