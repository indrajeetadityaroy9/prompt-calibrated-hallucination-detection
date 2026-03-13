# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot, training-free hallucination detection system for retrieval-augmented LLMs. It intercepts transformer internals during generation to extract five mechanistically-grounded signals and fuses them via cross-signal precision-weighted entropy-gated aggregation into per-token risk scores — no training data, calibration, or external models required.

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Run evaluation on QA datasets (mode determined by config's run.mode field)
python -m experiments --config experiments/configs/main.yaml

# Run signal ablation study
python -m experiments --config experiments/configs/ablation.yaml

# Alternative CLI entry point (installed via pip install -e .)
ag-sar-eval --config experiments/configs/main.yaml
```

Requires Python >= 3.10, PyTorch >= 2.5.1, Transformers >= 4.57.0.

No test suite, linting, or formatting tools are configured. Validation is done via the experiment evaluation pipeline.

## Architecture

### Pipeline Flow

Entry point: `Detector` in `ag_sar/detector.py`.

1. **Prefill** (once per input): `ModelAdapter` auto-detects architecture (LLaMA/Mistral/Qwen/Gemma, Phi, GPT-2/Neo, Falcon, GPT-NeoX via dot-path traversal). Context capture at midpoint layer. Context subspace (SVD + effective rank), reasoning subspace (bottom SVD of lm_head + effective rank), prompt center, magnitude tau (DPS gate), SPT window size (effective rank of context singular values), prompt statistics (PIT reference values) from tail window sqrt(prompt_len). Hidden states in bfloat16.

2. **Generation** (per token): 5 signals → PIT or direct normalization → cross-signal precision-coupled entropy-gated fusion → response risk → Otsu-adaptive spans.

### Three Entry Points

- `detect(question, context, prompt_template=...)`: Convenience — builds input from strings with customizable template.
- `detect_from_tokens(input_ids, context_mask)`: Format-agnostic detection from pre-tokenized input + boolean context mask.
- `score(prompt, response_text, context_text)`: Score pre-existing text via teacher-forced decode.

### Five Signals

- **CUS** — `signals/cus.py`: Lookback ratio bimodality. CUS = 1 - Otsu coefficient of per-head context attention mass. Direct mode (no PIT).
- **POS** — `signals/_jsd_base.py`: Candidate-set JSD-weighted directional override. Per-layer JSD(pre-MLP, post-MLP) weights the directional override (fraction of MLP delta escaping context subspace). PIT normalized.
- **DPS** — `signals/dps.py`: Dual-subspace projection (s_rsn/(s_ctx+s_rsn)) with magnitude gating. All-layer mean. PIT normalized.
- **SPT** — `signals/spt.py`: Tracy-Widom calibrated spectral phase-transition. SPT = 1 - F_{TW,1}((λ₁ - μ_TW)/σ_TW) where μ_TW = σ²(1+√γ)² is the MP upper edge and σ_TW is the finite-sample TW scaling rate. Smooth probabilistic transition replaces binary BBP clamp. Direct mode.
- **Spectral Gap** — `signals/spt.py`: λ₂/(λ₁+λ₂) ∈ [0, 0.5] capturing directional coherence. Near 0 = clean spike (low risk), near 0.5 = degenerate (high risk). Computed from same SVD as SPT. Direct mode.

### Hook System (`ag_sar/hooks/`)

2-point capture per layer: h_resid_attn, h_resid_mlp.
- `adapter.py`: ModelAdapter — auto-detects architecture via dot-path traversal across 5 model families
- `buffer.py`: EphemeralHiddenBuffer — bfloat16, cleared per token
- `layer_hooks.py`: LayerHooks — 2 hooks per layer
- `prefill_hooks.py`: PrefillContextHook (boolean context_mask)

### Calibration (`ag_sar/calibration.py`)

`self_calibrate()`: prompt-anchored PIT reference values + variance for DPS and POS. CUS, SPT, and spectral gap use direct mode. SPT/gap variance computed incrementally from prompt-tail evaluations. Computes DPS-POS 2×2 precision block embedded in 5×5 diagonal precision matrix to capture their cross-correlation. `adaptive_window()`: sqrt(prompt_len).

### Aggregation (`ag_sar/aggregation/`)

- `fusion.py`: w_i = Σ_j Ω_ij × (1-H_j)^κ — cross-signal precision-coupled entropy-gated fusion (generalized DerSimonian & Laird). Token-level + response-level (signal-first). 5 signals: {cus, pos, dps, spt, spectral_gap}. Falls back to diagonal inverse-variance when precision matrix unavailable.
- `spans.py`: Otsu-adaptive threshold. Expected-gap merging.

### Data Structures (`ag_sar/config.py`)

- `TokenSignals`: CUS/POS/DPS/SPT/spectral_gap per token
- `DetectionResult`: generated_text, token_signals, token_risks, risky_spans, response_risk, is_flagged

### Experiments (`experiments/`)

- `eval.py`: CLI entry point (`--config` dispatch); mode selected by YAML `run.mode` ("evaluation" or "ablation")
- `run_eval.py`: Evaluation orchestration with YAML config support
- `run_ablation.py`: Leave-one-out signal ablation study (delta-AUROC per signal)
- `schema.py`: Typed YAML config dataclasses (`ExperimentConfig.from_yaml()`)
- `common.py`: Model loading, dataset dispatch, output utilities
- `loaders.py`: Dataset loaders (TriviaQA, SQuAD)
- `metrics.py`: AUROC, AUPRC, TPR@FPR, ECE, AURC with bootstrap CI
- `answer_matching.py`: SQuAD-style F1 answer matching
- `configs/main.yaml`: Default config (LLaMA-3.1-8B-Instruct, triviaqa+squad, 200 samples)
- `configs/ablation.yaml`: Ablation study configuration

### Numerics (`ag_sar/numerics.py`)

`jsd` (bits, [0,1]), `effective_rank` (parameter-free via singular value entropy), `otsu_threshold`, `tracy_widom_cdf` (TW β=1 CDF via Cornish-Fisher expansion with exact moments). Single named constant: EPS.

### Public API (`ag_sar/__init__.py`)

Three exports: `Detector`, `TokenSignals`, `DetectionResult`.

## Design Principles

- **Zero-parameter**: Otsu, effective rank, PIT. No learned weights, no magic numbers.
- **Prompt-anchored**: PIT normalization against prefill-tail empirical CDF.
- **Cross-signal precision fusion**: Full Ω = Σ⁻¹ from prompt-tail covariance captures inter-signal dependencies. Entropy-gated: p=0.5 → weight=0. Generalizes DerSimonian & Laird (1986) to correlated estimators (Hartung, Knapp & Sinha, 2008).
- **Tracy-Widom spectral calibration**: SPT uses TW₁ CDF for proper finite-sample phase transition probability. No binary clamp.
- **Architecture portability**: ModelAdapter auto-detects via dot-path traversal (5 model families).
- **Format-agnostic**: Boolean context_mask supports any prompt format and disjoint multi-document contexts.
- **Candidate-set efficiency**: Effective rank of logit distribution determines candidate set size.
