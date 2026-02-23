# Research Objectives & Canonical Mapping

## 1. Core Research Problem

Detecting hallucinations in retrieval-augmented LLM generation at inference time — without training data, calibration datasets, multiple sampling passes, or external models. Existing methods require at least one of these; AG-SAR requires none. The system intercepts transformer internals during a single forward pass, extracts six mechanistically-grounded signals targeting distinct failure modes along the causal processing chain, and fuses them into per-token and response-level risk scores via a zero-parameter aggregation pipeline.

## 2. Novel Contributions & Mechanisms

### Contribution 1: Causal Signal Decomposition (6 signals)

Each signal targets a distinct failure mode at a specific stage of the transformer's computation:

- **CUS (Context Utilization Score):**
  `ag_sar/signals/cus.py` → `ContextUtilizationSignal.compute_lookback_ratio_signal()`
  Affinity-weighted lookback ratio bimodality across all attention heads. Bimodal = healthy (some heads attend context, others don't). Unimodal = hallucinating. CUS = 1 - weighted Otsu coefficient. Range [0,1], higher = riskier. Direct probability mapping (no z-score).
  *Mathematical definition:* CUS(t) = 1 - σ²_between(LR) / σ²_total(LR), where LR is the vector of per-head lookback ratios weighted by copying affinity from prefill.

- **POS (Parametric Override Score):**
  `ag_sar/signals/_jsd_base.py` → `CandidateJSDSignal.compute_pos()`
  Candidate-set JSD between pre-FFN and post-FFN logit distributions, with directional override decomposition relative to context subspace. Detects when MLP overrides context-grounded attention output.
  *Mathematical definition:* POS(t) = mean over active layers of max(0, 1 - ||proj_ctx(δ)|| / (||δ|| · √(k/d))), where δ = norm(h_post) - norm(h_pre), active layers selected via Otsu on per-layer JSD.

- **DPS (Dual-Subspace Projection Score):**
  `ag_sar/signals/dps.py` → `DualSubspaceGrounding.compute_dps()`
  Projects hidden states onto context subspace (SVD of prefill context, Marchenko-Pastur rank) and reasoning subspace (bottom SVD of unembedding matrix). Magnitude-gated to suppress unreliable ratios near centroid.
  *Mathematical definition:* DPS(t) = gate(t) · s_rsn/(s_ctx + s_rsn) + (1 - gate(t)) · 0.5, where s_* = ||proj_*(h - μ_ctx)|| / ||h - μ_ctx||, gate = 1 - exp(-||h - μ_ctx||²/τ²). Layer selection via variance-based Otsu thresholding.

- **DoLa (Layer-Contrast Score):**
  `ag_sar/signals/_jsd_base.py` → `CandidateJSDSignal.compute_dola_score()`
  log P_final(token) - log P_premature(token), where premature layer = argmax JSD over early layers. High DoLa = factual content added in late layers = safe. Polarity inverted during aggregation (higher_is_riskier=False).
  *Mathematical definition:* DoLa(t) = log P_L(y_t | C_t) - log P_j*(y_t | C_t), where j* = argmax_{j ∈ early} JSD(q_L, q_j) on candidate set.

- **CGD (Context-Grounding Direction):**
  `ag_sar/signals/dps.py` → `DualSubspaceGrounding.compute_grounding_direction()`
  Cosine between generation drift vector and context direction vector. Measures whether generation moves toward or away from context.
  *Mathematical definition:* CGD(t) = (1 - cos(h_t - μ_prompt, μ_ctx - μ_prompt)) / 2. Range [0,1], higher = away from context = riskier.

- **STD (Semantic Trajectory Dynamics):**
  `ag_sar/signals/std.py` → `SemanticTrajectoryDynamics.compute_std()`
  Context-projected directional inconsistency and divergence asymmetry across transformer layers. Uses h_mlp_in (post-norm) to avoid residual norm growth artifacts. Factual tokens show smooth convergent trajectories; hallucinated tokens show oscillatory divergent dynamics. Inspired by LSD (arXiv 2510.04933), adapted training-free.
  *Mathematical definition:* STD(t) = mean(directional_inconsistency, divergence_asymmetry), where directional_inconsistency = (1 - mean cos(d_proj^l, d_proj^(l+1))) / 2 and divergence_asymmetry = sigmoid(late_vel / early_vel - 1).

### Contribution 2: Prompt-Anchored Self-Calibration

- **Self-calibration pipeline:**
  `ag_sar/calibration.py` → `self_calibrate()`
  All normalization statistics derived from the prefill pass itself. No external calibration data.

- **Adaptive tail windowing:**
  `ag_sar/calibration.py` → `adaptive_window()`
  Window = √(prompt_len), clamped to [4, prompt_len//2].

- **Data-driven DPS layer selection:**
  `ag_sar/calibration.py` → `select_informative_dps_layers()`
  Per-layer DPS variance + Otsu threshold selects discriminative layers.

### Contribution 3: Conflict-Aware Precision-Weighted Entropy-Gated Fusion

- **Per-token fusion:**
  `ag_sar/aggregation/fusion.py` → `PromptAnchoredAggregator._conflict_aware_fusion()`
  w_i(t) = precision_i × (1 - H_i(t))^κ, R(t) = Σ w_i·p_i / Σ w_i. Signals at p=0.5 (uninformative) get weight 0 via entropy gating. Signals with tight prompt variance get higher precision weight.

- **Response-level aggregation:**
  `ag_sar/aggregation/fusion.py` → `PromptAnchoredAggregator._response_level_risk()`
  Precision-weighted signal-first aggregation. Per-signal mean → normalize → weight by precision × entropy_weight.

- **Span detection:**
  `ag_sar/aggregation/spans.py` → `SpanMerger.adaptive()`
  Bimodality-adaptive Tukey fence (multiplier = 1.5 × (1 - Otsu bimodality coefficient)) on per-token risks for span boundary detection, with expected-gap-based max_gap.

### Contribution 4: Zero-Parameter Design

Every threshold from data, architecture, or information theory:
- Copying head identification: Otsu on attention affinity (`ag_sar/signals/cus.py` → `identify_copying_heads()`)
- Context subspace rank: Marchenko-Pastur boundary (`ag_sar/signals/dps.py` → `_find_spectral_gap_rank()`)
- Active POS layers: Otsu on per-token JSD (`ag_sar/signals/_jsd_base.py` → `compute_pos()`)
- DPS layer selection: Otsu on per-layer variance (`ag_sar/calibration.py` → `select_informative_dps_layers()`)
- Span threshold: Tukey fence (`ag_sar/aggregation/spans.py` → `SpanMerger.adaptive()`)

## 3. Primary Execution Path

### 3.1 Data

- **Datasets:** TriviaQA (rc split, validation) and SQuAD v2 (validation)
- **Loading:** `evaluation/loaders/` → `load_triviaqa()`, `load_squad()`
- **Format:** Each sample = {question, context, answers}. Context truncated to 2000 chars for TriviaQA.
- **Ground truth:** F1 token overlap between generated answer and reference answers. F1 < 0.3 = hallucination label.
- **Library:** HuggingFace `datasets`

### 3.2 Inference

Three entry points in `ag_sar/detector.py`:

1. **`Detector.detect(question, context, prompt_template=...)`**: Convenience — builds input from strings with customizable template, delegates to `detect_from_tokens()`.
2. **`Detector.detect_from_tokens(input_ids, context_mask)`**: Format-agnostic detection from pre-tokenized input + boolean context mask.
3. **`Detector.score(prompt, response_text, context_text)`**: Teacher-forced scoring of pre-existing text.

Pipeline:
1. **Prefill:** `Detector._prefill()` — single forward pass producing:
   - Multi-layer context capture at 3 candidate layers (spectral gap selection)
   - Copying heads via Otsu on attention affinity
   - Context subspace V_ctx via SVD + Marchenko-Pastur
   - Reasoning subspace V_rsn via bottom SVD of lm_head.weight (cached per model)
   - Prompt statistics (PIT reference values + variance per signal) via `self_calibrate()`
   - DPS layer selection via variance-based Otsu
   - Prompt center (CGD with all-context guard), magnitude tau (DPS gate)
   - Context basis for STD signal
2. **Autoregressive generation loop:** Per token:
   - `compute_token_signals()` → CUS, POS, DPS, DoLa, CGD, STD
   - Hidden states from `EphemeralHiddenBuffer` (cleared per token)
3. **Aggregation:**
   - `PromptAnchoredAggregator.compute_risk()` → per-token risks + response risk
   - `SpanMerger.adaptive()` → risky spans
4. **Output:** `DetectionResult` with generated_text, token_signals, token_risks, risky_spans, response_risk, is_flagged

### 3.3 Evaluation

Entry point: `experiments/run_evaluation.py` → `run_evaluation()`

**Primary metrics:**
| Metric | Definition | Library |
|--------|-----------|---------|
| AUROC | Area under ROC curve (trapezoidal) | `sklearn.metrics.roc_auc_score` |
| AUPRC | Area under precision-recall curve | `sklearn.metrics.average_precision_score` |
| TPR@5%FPR | True positive rate at 5% false positive rate | `sklearn.metrics.roc_curve` + interpolation |

**Calibration metrics:**
| Metric | Definition | Library |
|--------|-----------|---------|
| ECE | Expected calibration error (10-bin) | Custom (standard formula) |
| Brier | Mean squared error of probabilities | `sklearn.metrics.brier_score_loss` |

**Selective prediction metrics:**
| Metric | Definition | Library |
|--------|-----------|---------|
| AURC | Area under risk-coverage curve | Custom |
| E-AURC | Excess AURC (AURC - AURC_optimal) | Custom (depends on AURC) |
| Risk@90% | Error rate at 90% coverage | Custom |

## 4. Constraint Checklist

- [x] **Zero-parameter design:** All thresholds derived from data, architecture, or information theory. No learned weights.
- [x] **Prompt-anchored calibration:** PIT normalization against prefill-tail empirical CDF. No external calibration data.
- [x] **SOTA-by-default execution:** `python experiments/run_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --all` runs the full pipeline with all 6 signals.
- [x] **Architecture portability:** ModelAdapter auto-detects via dot-path traversal (6 model families).
- [x] **Format-agnostic:** Boolean context_mask supports any prompt format and disjoint multi-document contexts.

## 5. Module Classification

### Core Research Logic
| Module | Role |
|--------|------|
| `ag_sar/signals/cus.py` | CUS signal (Contribution 1) |
| `ag_sar/signals/_jsd_base.py` | POS + DoLa signals (Contribution 1) |
| `ag_sar/signals/dps.py` | DPS + CGD signals (Contribution 1) |
| `ag_sar/signals/std.py` | STD signal (Contribution 1) |
| `ag_sar/aggregation/fusion.py` | Entropy-gated fusion (Contribution 3) |
| `ag_sar/aggregation/spans.py` | Span detection (Contribution 3) |
| `ag_sar/calibration.py` | Self-calibration pipeline (Contribution 2) |
| `ag_sar/config.py` | Data structures: TokenSignals, DetectionResult (Contribution 4) |
| `ag_sar/numerics.py` | Numerical primitives: safe_jsd, otsu_threshold (Contribution 4) |
| `ag_sar/hooks/` | 3-point hidden state capture, ModelAdapter (Infrastructure) |
| `ag_sar/detector.py` | Orchestrator: prefill → detect → aggregate (Entry point) |

### Evaluation Infrastructure
| Module | Role |
|--------|------|
| `evaluation/metrics.py` | Metric computation (sklearn-backed) |
| `evaluation/answer_matching.py` | SQuAD-style F1 matching |
| `evaluation/loaders/` | Dataset loaders (TriviaQA, SQuAD) |
| `experiments/run_evaluation.py` | Evaluation orchestration with YAML config |
| `experiments/ablation.py` | Leave-one-out signal ablation study |
