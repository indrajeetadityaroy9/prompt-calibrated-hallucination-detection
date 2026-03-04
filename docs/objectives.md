# Research Objectives & Canonical Mapping

## 1. Core Research Problem

Detecting hallucinations in retrieval-augmented LLM generation at inference time — without training data, calibration datasets, multiple sampling passes, or external models. AG-SAR intercepts transformer internals during a single forward pass, extracts four mechanistically-grounded signals targeting distinct failure modes along the causal processing chain, and fuses them into per-token and response-level risk scores via a zero-parameter aggregation pipeline.

## 2. Novel Contributions & Mechanisms

### Contribution 1: Causal Signal Decomposition (4 signals)

Each signal targets a distinct failure mode at a specific stage of the transformer's computation:

- **CUS (Context Utilization Score):**
  `ag_sar/signals/cus.py` → `compute_cus()`
  Lookback ratio bimodality across all attention heads. Bimodal = healthy. Unimodal = hallucinating. CUS = 1 - Otsu coefficient. Range [0,1], higher = riskier. Direct mode (no PIT).
  *Mathematical definition:* CUS(t) = 1 - σ²_between(LR) / σ²_total(LR), where LR is the vector of per-head lookback ratios.

- **POS (Parametric Override Score):**
  `ag_sar/signals/_jsd_base.py` → `CandidateJSDSignal.compute_pos()`
  Candidate-set JSD between pre-FFN and post-FFN logit distributions, with JSD-weighted directional override decomposition relative to context subspace. All-layer mean.
  *Mathematical definition:* POS(t) = Σ_l JSD_l · override_l / Σ_l JSD_l, where override = max(0, 1 - ||proj_ctx(δ)|| / (||δ|| · √(k/d))).

- **DPS (Dual-Subspace Projection Score):**
  `ag_sar/signals/dps.py` → `DualSubspaceGrounding.compute_dps()`
  Projects hidden states onto context subspace (SVD of prefill context, effective rank) and reasoning subspace (bottom SVD of unembedding matrix, effective rank). Magnitude-gated. All-layer mean.
  *Mathematical definition:* DPS(t) = gate(t) · s_rsn/(s_ctx + s_rsn) + (1 - gate(t)) · 0.5, where gate = 1 - exp(-||h - μ_ctx||²/τ²).

- **SPT (Spectral Phase-Transition Score):**
  `ag_sar/signals/spt.py` → `SpectralPhaseTransition.compute_spt()`
  Sliding-window covariance spectrum of midpoint-layer hidden states. Measures departure of leading eigenvalue from Marchenko-Pastur upper edge (BBP phase transition). Window size = effective rank of context singular values.
  *Mathematical definition:* SPT(t) = 1 - clamp((λ₁ - λ₊)/λ₊, 0, 1), where λ₊ = σ²(1+√(d/W))².

### Contribution 2: Prompt-Anchored Self-Calibration

- **Self-calibration pipeline:**
  `ag_sar/calibration.py` → `self_calibrate()`
  All normalization statistics derived from the prefill pass itself. DPS and POS use PIT normalization (sorted reference values from prompt tail). CUS and SPT use direct mode with peer-derived variance.

- **Adaptive tail windowing:**
  `ag_sar/calibration.py` → `adaptive_window()`
  Window = √(prompt_len), capped at prompt_len.

### Contribution 3: Entropy-Gated Precision-Weighted Fusion

- **Per-token fusion:**
  `ag_sar/aggregation/fusion.py` → `PromptAnchoredAggregator._entropy_gated_fusion()`
  w_i(t) = (1/var_i) × (1 - H_i(t))^κ. Signals at p=0.5 get weight 0 via entropy gating. κ = 1 + median(prompt decisiveness).

- **Response-level aggregation:**
  `ag_sar/aggregation/fusion.py` → `PromptAnchoredAggregator._response_level_risk()`
  Signal-first: per-signal mean → PIT normalize → precision × entropy weighted fusion.

- **Span detection:**
  `ag_sar/aggregation/spans.py` → `SpanMerger.adaptive()`
  Otsu-adaptive threshold on per-token risks with expected-gap merging.

### Contribution 4: Zero-Parameter Design

Every threshold from data, architecture, or information theory:
- Context/reasoning subspace rank: effective rank (Shannon entropy of singular values)
- SPT window size: effective rank of context singular values
- SPT noise boundary: Marchenko-Pastur upper edge λ₊ = σ²(1+√γ)²
- Span threshold: Otsu's method on token risks
- Entropy gating exponent: data-driven from prompt decisiveness

## 3. Primary Execution Path

### 3.1 Data

- **Datasets:** TriviaQA (rc split, validation) and SQuAD v2 (validation)
- **Loading:** `experiments/loaders.py` → `load_triviaqa()`, `load_squad()`
- **Ground truth:** F1 token overlap. Adaptive Otsu threshold on F1 scores separates correct from hallucinated.

### 3.2 Inference

Three entry points in `ag_sar/detector.py`, all converging to `_generation_loop()`:

1. **`Detector.detect(question, context)`**: Builds input from strings, delegates to `detect_from_tokens()`.
2. **`Detector.detect_from_tokens(input_ids, context_mask)`**: Format-agnostic detection.
3. **`Detector.score(prompt, response_text, context_text)`**: Teacher-forced scoring.

Pipeline:
1. **Prefill** (`_prefill()`): Context capture at midpoint layer → context/reasoning subspace SVD → SPT window size → prompt center + magnitude tau → self_calibrate() → seed SPT window
2. **Generation** (per token): `compute_token_signals()` → CUS + POS + DPS + SPT
3. **Aggregation** (`_aggregate_results()`): `compute_risk()` → token risks + response risk → `SpanMerger.adaptive()` → spans
4. **Output:** `DetectionResult`

### 3.3 Evaluation

Entry point: `experiments/eval.py` → dispatches to `run_eval.py` or `run_ablation.py` based on config mode.

## 4. Module Classification

### Core Research Logic
| Module | Role |
|--------|------|
| `ag_sar/signals/cus.py` | CUS signal (Contribution 1) |
| `ag_sar/signals/_jsd_base.py` | POS signal (Contribution 1) |
| `ag_sar/signals/dps.py` | DPS signal (Contribution 1) |
| `ag_sar/signals/spt.py` | SPT signal (Contribution 1) |
| `ag_sar/aggregation/fusion.py` | Entropy-gated fusion (Contribution 3) |
| `ag_sar/aggregation/spans.py` | Span detection (Contribution 3) |
| `ag_sar/calibration.py` | Self-calibration (Contribution 2) |
| `ag_sar/config.py` | TokenSignals, DetectionResult (Contribution 4) |
| `ag_sar/numerics.py` | jsd, effective_rank, otsu_threshold (Contribution 4) |
| `ag_sar/hooks/` | 2-point hidden state capture, ModelAdapter (Infrastructure) |
| `ag_sar/detector.py` | Orchestrator: prefill → detect → aggregate (Entry point) |

### Evaluation Infrastructure
| Module | Role |
|--------|------|
| `experiments/metrics.py` | Metric computation (sklearn-backed) |
| `experiments/answer_matching.py` | SQuAD-style F1 matching |
| `experiments/loaders.py` | Dataset loaders (TriviaQA, SQuAD) |
| `experiments/run_eval.py` | Evaluation orchestration |
| `experiments/run_ablation.py` | Leave-one-out signal ablation |
