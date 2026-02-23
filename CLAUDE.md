# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Aggregated Signal Architecture for Risk) is a zero-shot, training-free hallucination detection system for retrieval-augmented LLMs. It intercepts transformer internals during generation to extract six mechanistically-grounded signals and fuses them via conflict-aware precision-weighted entropy-gated aggregation into per-token risk scores — no training data, calibration, or external models required.

## Build & Development Commands

```bash
# Install (editable mode)
pip install -e .

# Run evaluation on QA datasets
python experiments/run_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 100
python experiments/run_evaluation.py --config experiments/configs/default.yaml --all

# Run signal ablation study
python experiments/ablation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 100
python experiments/ablation.py --config experiments/configs/ablation.yaml
```

Requires Python >= 3.10, PyTorch >= 2.5.1, Transformers >= 4.57.0.

## Architecture

### Pipeline Flow

Entry point: `Detector` in `ag_sar/detector.py`.

1. **Prefill** (once per input): `ModelAdapter` auto-detects architecture (LLaMA/Mistral/Qwen/Gemma, Phi, GPT-2/Neo, Falcon, GPT-NeoX via dot-path traversal). Multi-layer context capture at 3 candidate layers with spectral gap selection (S[0]/S[1]). Identifies copying heads (Otsu on attention affinity), computes context subspace (SVD + Marchenko-Pastur rank), reasoning subspace (bottom SVD of lm_head), prompt center (CGD with all-context guard), magnitude tau (DPS gate), informative DPS layers (variance-based Otsu), and prompt statistics (PIT reference values) from tail window sqrt(prompt_len). Hidden states in bfloat16.

2. **Generation** (per token): 6 signals → PIT or direct normalization → entropy-gated precision-weighted fusion → response risk → bimodality-adaptive percentile spans.

### Three Entry Points

- `detect(question, context, prompt_template=...)`: Convenience — builds input from strings with customizable template, delegates to `detect_from_tokens()`.
- `detect_from_tokens(input_ids, context_mask)`: Format-agnostic detection from pre-tokenized input + boolean context mask. Supports disjoint multi-document contexts.
- `score(prompt, response_text, context_text)`: Score pre-existing text via teacher-forced decode. No generation — evaluates existing model output.

### Six Signals

- **CUS** — `signals/cus.py`: Affinity-weighted lookback ratio bimodality. CUS = 1 - weighted Otsu coefficient. Direct mode (no PIT).
- **POS** — `signals/_jsd_base.py`: Candidate-set JSD + directional override decomposition. Otsu-selected active layers. PIT normalized.
- **DPS** — `signals/dps.py`: Dual-subspace projection (s_rsn/(s_ctx+s_rsn)) with magnitude gating. Data-driven layer selection. PIT normalized.
- **DoLa** — `signals/_jsd_base.py`: log P_premature - log P_final (argmax JSD over early layers). Negated at source so higher = riskier. PIT normalized.
- **CGD** — `signals/dps.py`: (1 - cos(h_gen - prompt_center, ctx_center - prompt_center)) / 2. PIT normalized.
- **STD** — `signals/std.py`: Semantic Trajectory Dynamics — context-projected directional inconsistency + divergence asymmetry across layers. Uses h_mlp_in (post-norm) to avoid residual norm growth. Inspired by LSD (arXiv 2510.04933), adapted training-free. PIT normalized.

### Hook System (`ag_sar/hooks/`)

3-point capture per layer: h_resid_attn, h_mlp_in, h_resid_mlp.
- `adapter.py`: ModelAdapter — auto-detects architecture via dot-path traversal across 6 model families
- `buffer.py`: EphemeralHiddenBuffer — bfloat16, cleared per token
- `layer_hooks.py`: LayerHooks — 3 hooks per layer
- `prefill_hooks.py`: PrefillContextHook (boolean context_mask), PrefillStatisticsHook

### Calibration (`ag_sar/calibration.py`)

`self_calibrate()`: prompt-anchored PIT reference values + variance per signal (all 6). `select_informative_dps_layers()`: variance-based Otsu. `adaptive_window()`: sqrt(prompt_len) clamped.

### Aggregation (`ag_sar/aggregation/`)

- `fusion.py`: w_i = (1/var_i) × (1-H_i)^κ. Token-level + response-level (signal-first). Data-driven conflict normalization. 6 signals: {cus, pos, dps, dola, cgd, std}.
- `spans.py`: Bimodality-adaptive percentile threshold (Otsu split ↔ P95 interpolation). Expected-gap merging.

### Data Structures (`ag_sar/config.py`)

- `TokenSignals`: CUS/POS/DPS/DoLa/CGD/STD per token
- `DetectionResult`: generated_text, token_signals, token_risks, risky_spans, response_risk, is_flagged

### Evaluation (`evaluation/`)

Separate from core — `ag_sar/` has zero imports from `evaluation/`. `metrics.py` (AUROC, AUPRC, TPR@FPR, ECE, AURC, bootstrap CI), `answer_matching.py` (F1), `input_builder.py` (tokenization — legacy, retained for backward compat), `loaders/` (TriviaQA, SQuAD).

### Experiments (`experiments/`)

- `run_evaluation.py`: Evaluation orchestration with YAML config support (CLI overrides)
- `ablation.py`: Leave-one-out signal ablation study (delta-AUROC per signal)
- `configs/default.yaml`: Default experiment configuration
- `configs/ablation.yaml`: Ablation study configuration

### Numerics (`ag_sar/numerics.py`)

safe_softmax (dtype-aware clamping), safe_log_softmax, safe_jsd (bits, [0,1]), otsu_threshold, entropy_adaptive_nucleus. Single named constant: EPS.

## Design Principles

- **Zero-parameter**: Otsu, Marchenko-Pastur, PIT, order statistics. No learned weights.
- **Prompt-anchored**: PIT normalization against prefill-tail empirical CDF.
- **Entropy-gated fusion**: p=0.5 → weight=0. Inverse-variance weighting (DerSimonian & Laird 1986).
- **Architecture portability**: ModelAdapter auto-detects via dot-path traversal (6 model families).
- **Format-agnostic**: Boolean context_mask supports any prompt format and disjoint multi-document contexts.
- **Candidate-set efficiency**: Entropy-adaptive nucleus mass (0.95–1.0 based on distribution entropy) + previous top-k + emitted token.
