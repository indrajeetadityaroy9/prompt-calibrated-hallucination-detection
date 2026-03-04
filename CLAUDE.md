# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot, training-free hallucination detection system for retrieval-augmented LLMs. It intercepts transformer internals during generation to extract four mechanistically-grounded signals and fuses them via precision-weighted entropy-gated aggregation into per-token risk scores — no training data, calibration, or external models required.

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Run evaluation on QA datasets
python -m experiments run-eval --config experiments/configs/main.yaml

# Run signal ablation study
python -m experiments run-ablation --config experiments/configs/ablation.yaml
```

Requires Python >= 3.10, PyTorch >= 2.5.1, Transformers >= 4.57.0.

## Architecture

### Pipeline Flow

Entry point: `Detector` in `ag_sar/detector.py`.

1. **Prefill** (once per input): `ModelAdapter` auto-detects architecture (LLaMA/Mistral/Qwen/Gemma, Phi, GPT-2/Neo, Falcon, GPT-NeoX via dot-path traversal). Context capture at midpoint layer. Context subspace (SVD + effective rank), reasoning subspace (bottom SVD of lm_head + effective rank), prompt center, magnitude tau (DPS gate), SPT window size (effective rank of context singular values), prompt statistics (PIT reference values) from tail window sqrt(prompt_len). Hidden states in bfloat16.

2. **Generation** (per token): 4 signals → PIT or direct normalization → entropy-gated precision-weighted fusion → response risk → Otsu-adaptive spans.

### Three Entry Points

- `detect(question, context, prompt_template=...)`: Convenience — builds input from strings with customizable template.
- `detect_from_tokens(input_ids, context_mask)`: Format-agnostic detection from pre-tokenized input + boolean context mask.
- `score(prompt, response_text, context_text)`: Score pre-existing text via teacher-forced decode.

### Four Signals

- **CUS** — `signals/cus.py`: Lookback ratio bimodality. CUS = 1 - Otsu coefficient of per-head context attention mass. Direct mode (no PIT).
- **POS** — `signals/_jsd_base.py`: Candidate-set JSD-weighted directional override. Per-layer JSD(pre-MLP, post-MLP) weights the directional override (fraction of MLP delta escaping context subspace). PIT normalized.
- **DPS** — `signals/dps.py`: Dual-subspace projection (s_rsn/(s_ctx+s_rsn)) with magnitude gating. All-layer mean. PIT normalized.
- **SPT** — `signals/spt.py`: Spectral phase-transition via Marchenko-Pastur BBP threshold. Sliding window of midpoint-layer hidden states. SPT = 1 - clamp((λ₁ - λ₊)/λ₊, 0, 1) where λ₊ = σ²(1+√γ)². Direct mode (MP edge is its own null model).

### Hook System (`ag_sar/hooks/`)

2-point capture per layer: h_resid_attn, h_resid_mlp.
- `adapter.py`: ModelAdapter — auto-detects architecture via dot-path traversal across 5 model families
- `buffer.py`: EphemeralHiddenBuffer — bfloat16, cleared per token
- `layer_hooks.py`: LayerHooks — 2 hooks per layer
- `prefill_hooks.py`: PrefillContextHook (boolean context_mask)

### Calibration (`ag_sar/calibration.py`)

`self_calibrate()`: prompt-anchored PIT reference values + variance for DPS and POS. CUS and SPT use direct mode with peer-derived variance. `adaptive_window()`: sqrt(prompt_len).

### Aggregation (`ag_sar/aggregation/`)

- `fusion.py`: w_i = (1/var_i) × (1-H_i)^κ. Token-level + response-level (signal-first). 4 signals: {cus, pos, dps, spt}.
- `spans.py`: Otsu-adaptive threshold. Expected-gap merging.

### Data Structures (`ag_sar/config.py`)

- `TokenSignals`: CUS/POS/DPS/SPT per token
- `DetectionResult`: generated_text, token_signals, token_risks, risky_spans, response_risk, is_flagged

### Experiments (`experiments/`)

- `run_eval.py`: Evaluation orchestration with YAML config support
- `run_ablation.py`: Leave-one-out signal ablation study (delta-AUROC per signal)
- `configs/main.yaml`: Default experiment configuration
- `configs/ablation.yaml`: Ablation study configuration

### Numerics (`ag_sar/numerics.py`)

`jsd` (bits, [0,1]), `effective_rank` (parameter-free via singular value entropy), `otsu_threshold`. Single named constant: EPS.

## Design Principles

- **Zero-parameter**: Otsu, effective rank, PIT. No learned weights, no magic numbers.
- **Prompt-anchored**: PIT normalization against prefill-tail empirical CDF.
- **Entropy-gated fusion**: p=0.5 → weight=0. Inverse-variance weighting (DerSimonian & Laird 1986).
- **Architecture portability**: ModelAdapter auto-detects via dot-path traversal (5 model families).
- **Format-agnostic**: Boolean context_mask supports any prompt format and disjoint multi-document contexts.
- **Candidate-set efficiency**: Effective rank of logit distribution determines candidate set size.
