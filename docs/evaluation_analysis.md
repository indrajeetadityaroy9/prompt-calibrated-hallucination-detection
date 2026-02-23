# AG-SAR: Technical Evaluation Audit & SOTA Gap Analysis

**Audit Date:** 2026-02-23
**Scope:** Comprehensive mapping of AG-SAR's technical contributions against the 2025-2026 hallucination detection research landscape
**Auditor Role:** Senior ML Researcher — Technical Audit & Benchmarking

---

## Executive Summary

AG-SAR (Aggregated Signal Architecture for Risk) presents a **genuinely novel contribution** in zero-shot, training-free hallucination detection via five mechanistically-grounded internal signals fused through entropy-gated aggregation. The system's zero-parameter design is a meaningful differentiator in a field dominated by probe-based and training-dependent methods.

However, the **evaluation infrastructure has significant gaps** relative to 2025-2026 peer-review standards. The current evaluation covers only 2 QA datasets with F1-based labeling, lacks comparison against any external baseline, and does not evaluate on the field's gold-standard RAG benchmark (RAGTruth with human annotations). Existing experimental results in `results/` show promising but inconsistent performance (AUROC 0.52-0.75 on RAGTruth depending on signal/task), with unresolved length confounding and weak cross-task transfer.

**Verdict:** Strong conceptual contribution with substantial evaluation debt. Addressing the gaps identified below is necessary before credible peer-review submission.

---

## Phase 1: Objective Mapping

### 1.1 Claimed Research Contributions

| # | Contribution | Type | Novelty Assessment |
|---|-------------|------|-------------------|
| 1 | **Causal Signal Decomposition** — 5 independent signals (CUS, POS, DPS, DoLa, CGD) targeting distinct transformer failure modes | Architectural Innovation | **High.** No published method combines five mechanistically-distinct internal signals. ReDeEP (ICLR 2025) uses 2 signals; CHARM (ICLR 2026) uses graph-based attention but requires GNN training. The decomposition along the causal processing chain (attention → FFN → subspace → layer-contrast → direction) is original. |
| 2 | **Prompt-Anchored Self-Calibration** — PIT normalization against prefill-tail empirical CDF, MAD-based robust sigma, adaptive tail windowing | Methodological Innovation | **Moderate-High.** PIT for distribution-free normalization in this context is novel. However, the adaptive window heuristic (sqrt(prompt_len)) lacks theoretical justification beyond "balances accuracy vs. locality." |
| 3 | **Conflict-Aware Entropy-Gated Fusion** — w_i = (1/var_i) x (1-H_i)^kappa with data-driven conflict normalization | Methodological Innovation | **Moderate.** Inverse-variance weighting (DerSimonian & Laird 1986) is well-established in meta-analysis. The entropy gating mechanism is a reasonable adaptation to signal fusion, but the adaptive kappa derivation from prompt decisiveness is somewhat ad hoc. |
| 4 | **Zero-Parameter Design** — All thresholds from Otsu, Marchenko-Pastur, order statistics | Efficiency / Deployment Innovation | **High for deployment.** No learned weights, no calibration data, no hyperparameters. This is the primary differentiator against ReDeEP (regression), HARP (classifier), HaMI (MIL), and Lookback Lens (linear probe). |

### 1.2 Distinction Between Innovation Types

- **Architectural innovations:** Signal decomposition (CUS, POS, DPS, CGD are novel; DoLa is adapted from Chuang et al. 2024)
- **Efficiency gains:** Single-pass inference, zero-parameter design, no sampling
- **Novel task application:** Applying mechanistic interpretability to RAG hallucination detection at deployment time without supervision

### 1.3 Signal Originality Assessment

| Signal | Novelty | Closest Prior Work |
|--------|---------|-------------------|
| CUS | Novel formulation | Lookback Lens (Chuang et al. 2024) uses lookback ratio but trains a linear classifier; CUS uses Otsu bimodality coefficient — training-free |
| POS | Novel | ReDeEP's parametric knowledge score targets similar phenomenon but uses different mechanism (knowledge FFN identification) |
| DPS | Partially novel | HARP (2025) uses reasoning subspace projection from unembedding SVD — conceptually very close. AG-SAR adds context subspace and magnitude gating |
| DoLa | Adapted | Direct adaptation of Chuang et al. (ICLR 2024) with candidate-set restriction |
| CGD | Novel | No direct prior work on context-grounding direction cosine |

---

## Phase 2: Evaluation Forensics

### 2.1 Implemented Evaluation Metrics

**Location:** `evaluation/metrics.py` (239 lines)

| Metric | Implementation | Library | Status |
|--------|---------------|---------|--------|
| AUROC | `sklearn.metrics.roc_auc_score` | sklearn | Standard |
| AUPRC | `sklearn.metrics.average_precision_score` | sklearn | Standard |
| TPR@5%FPR | Custom via `sklearn.metrics.roc_curve` | sklearn | Standard |
| F1 (at t=0.5) | Custom | — | Standard |
| ECE (10-bin) | Custom implementation | — | **Needs validation** — should use `torchmetrics.BinaryCalibrationError` or validated reference |
| Brier Score | Inline MSE computation | — | Standard |
| AURC | Custom implementation | — | **Needs validation** — should validate against Geifman & El-Yaniv (2017) reference |
| E-AURC | Custom (AURC - optimal) | — | Depends on AURC correctness |
| Risk@90% Coverage | Custom | — | Standard |
| Bootstrap AUROC CI | Custom (1000 resamples) | — | Standard; could use `scipy.stats.bootstrap` |
| Per-signal AUROC | `sklearn.metrics.roc_auc_score` per signal | sklearn | Standard (diagnostic) |

**Assessment:** The metric suite is comprehensive and exceeds what most 2025-2026 papers report. The 10 metrics cover detection (AUROC, AUPRC), safety (TPR@FPR), calibration (ECE, Brier), selective prediction (AURC, E-AURC, Risk@90%), and uncertainty (bootstrap CI). Two custom implementations (ECE, AURC) should be validated against reference implementations.

### 2.2 Implemented Datasets & Loaders

| Dataset | Loader | Split | Samples | Ground Truth | Status |
|---------|--------|-------|---------|-------------|--------|
| TriviaQA (rc) | `evaluation/loaders/triviaqa.py` | validation | variable (--n-samples) | F1 token overlap < 0.3 | Implemented |
| SQuAD v2 | `evaluation/loaders/squad.py` | validation | variable (--n-samples) | F1 token overlap < 0.3 | Implemented |

**Ground truth labeling mechanism:**
- Generated answer compared against reference answers via SQuAD-style F1 (normalize articles/punctuation/whitespace, token-level precision/recall)
- `is_hallucination = (max_f1 < threshold)`, default threshold 0.3
- Adaptive threshold via Otsu on collected F1 distribution (fallback to 0.3)

### 2.3 Implemented Baselines

**None.** The evaluation script (`scripts/run_evaluation.py`) runs only AG-SAR. No external baseline methods are implemented for comparison. The `results/` directory contains experiments with internal signal variants (entropy, inv_margin, hidden_norm, eigenscore, etc.) but no external published baselines (ReDeEP, SelfCheckGPT, Lookback Lens, DoLa-standalone, perplexity, etc.).

### 2.4 Existing Experimental Results (from `results/`)

| Experiment | Key Finding | AUROC Range |
|-----------|-------------|-------------|
| RAGTruth processed | Overall: 0.689 (any hallucination), QA: 0.699, Summ: 0.586, Data2txt: 0.525 | 0.52-0.70 |
| HaluEval multi-dataset | QA: 0.999 (inflated by length), Summ: 0.723, Dialogue: 0.540 | 0.54-0.99 |
| TruthfulQA | 0.575 (parametric hallucination — outside AG-SAR's scope) | 0.58 |
| Generalization experiments | Context grounding most consistent (std 0.134), but negative delta vs length baseline across all signals | All negative vs length |
| ICML table1 (length-controlled) | Ensemble: 0.751 on RAGTruth QA, residual: 0.715 | 0.68-0.75 |
| Standard model eval | AUROC: 0.570 [CI: 0.496-0.643] | 0.57 |

**Critical finding from existing results:** The generalization experiments reveal that **no single signal consistently beats the length baseline across all datasets**. HaluEval QA is unusable (0.988 AUROC from length alone). Context grounding is the most generalizable signal (lowest variance), but all signals have negative mean delta above the length baseline.

### 2.5 Model Coverage

| Model | Status |
|-------|--------|
| LLaMA-3.1-8B-Instruct | Primary (implemented) |
| LLaMA-2-7B-Chat | Not implemented |
| LLaMA-2-13B-Chat | Not implemented |
| Mistral-7B-Instruct | Not implemented |
| Other families | Not implemented |

---

## Phase 3: SOTA Gap Analysis (2025-2026)

### 3.1 Benchmarking Standards Gap

#### 3.1.1 Missing Gold-Standard Benchmarks

| Benchmark | Status in Field | AG-SAR Status | Priority |
|-----------|----------------|---------------|----------|
| **RAGTruth** (ACL 2024) | **Mandatory** for RAG hallucination work. 18K responses, human-annotated span labels, 3 task types (QA/summarization/data-to-text) | Partially evaluated in `results/` with 200-300 samples. **Not integrated into `evaluation/loaders/`.** No span-level evaluation against human annotations | **Critical** |
| **HaluEval** (EMNLP 2023) | Common baseline, though dated; known length confounding | Partially evaluated in `results/` (multi_dataset_evaluation.json). Severe length bias detected (0.988 AUROC from length on QA). **Not in standard eval pipeline** | Medium |
| **Natural Questions (NQ Open)** | Standard QA benchmark for internal-signal methods (used by HaMI, HARP) | **Not implemented** | High |
| **TruthfulQA** | Standard for parametric hallucination (generation subset) | Partially evaluated (AUROC 0.575). AG-SAR targets contextual hallucination — weak performance expected | Low (out of scope) |
| **HaluBench** (Stanford) | Emerging RAG benchmark, 15K samples, diverse domains | **Not implemented** | Medium |
| **HalluRAG** (2025) | Newer RAG-specific benchmark used by LUMINA | **Not implemented** | Medium-High |
| **FaithBench/FaithJudge** (EMNLP 2025) | LLM-as-judge framework for RAG faithfulness | **Not implemented** | Medium |

#### 3.1.2 Missing Baseline Comparisons

The following baselines are **expected** in a 2025-2026 submission on internal hallucination detection:

| Baseline | Category | Used By | AG-SAR Status |
|----------|----------|---------|---------------|
| **Perplexity** | Probability | All papers | **Not implemented** |
| **LN-Entropy** | Probability | ReDeEP, LUMINA, HaMI | **Not implemented** |
| **SelfCheckGPT** | Sampling-based | ReDeEP, LUMINA, Freq-Attn | **Not implemented** |
| **EigenScore** | Sampling-based | LUMINA, HARP | **Not implemented** |
| **Semantic Entropy** | Sampling-based | HaMI, HARP | **Not implemented** |
| **Lookback Lens** | Attention-based | Freq-Attn, ReDeEP | **Not implemented** |
| **ReDeEP** (ICLR 2025) | Mechanistic | LUMINA, Freq-Attn | **Not implemented** |
| **DoLa** (standalone) | Layer-contrast | HaMI, HARP, LUMINA | Adapted as signal but not compared as standalone baseline |
| **INSIDE** | Internal states | ReDeEP | **Not implemented** |
| **P(True)** | Verbalization | LUMINA | **Not implemented** |

**Minimum credible baseline set for submission:** Perplexity, LN-Entropy, SelfCheckGPT, Lookback Lens, ReDeEP, DoLa (standalone). AG-SAR implements **zero** of these.

#### 3.1.3 Missing Model Diversity

2025-2026 papers evaluate across multiple model families and sizes:

| Paper | Models Evaluated |
|-------|-----------------|
| ReDeEP (ICLR 2025) | LLaMA-2-7B, LLaMA-2-13B, LLaMA-3-8B |
| HaMI (NeurIPS 2025) | LLaMA-2-7B, LLaMA-2-13B, LLaMA-3.1-8B, Mistral-7B |
| LUMINA (2025) | LLaMA-2-7B, LLaMA-2-13B, LLaMA-3-8B, Mistral-7B |
| HARP (2025) | Qwen-2.5-7B, LLaMA-3.1-8B |
| **AG-SAR** | **LLaMA-3.1-8B only** |

**Minimum expectation:** 2 sizes from one family (7B + 13B) + 1 cross-family model. AG-SAR evaluates on **1 model only**.

### 3.2 Evaluation Protocol Gaps

#### 3.2.1 Ground Truth Methodology

| Aspect | Community Standard | AG-SAR Implementation | Gap |
|--------|-------------------|----------------------|-----|
| QA labeling | F1 or ROUGE-L > threshold, with Otsu/data-driven threshold | F1 < 0.3 (default), Otsu adaptive | Adequate for QA |
| RAG labeling | Human annotation (RAGTruth) or NLI-based verification | F1 matching only | **Missing human-annotated ground truth** |
| Label quality | Multi-annotator agreement > 90% | Automatic F1 only | **No human validation of labels** |
| Span-level ground truth | RAGTruth provides word-level hallucination annotations | No span-level ground truth evaluation | **Critical gap** — AG-SAR produces span predictions but never evaluates span quality |

#### 3.2.2 Granularity Levels

| Level | Community Expectation | AG-SAR Status |
|-------|----------------------|---------------|
| Response-level | Mandatory (AUROC, AUPRC) | Implemented |
| Span-level | Highly valued when benchmark supports it (RAGTruth) | **Implemented in code** (`spans.py`) but **never evaluated against ground truth** |
| Token-level | Valued for mechanistic methods | Computed but not evaluated against annotations |

#### 3.2.3 Statistical Rigor

| Aspect | Community Standard | AG-SAR Status |
|--------|-------------------|---------------|
| Bootstrap CI | Recommended | Implemented (1000 resamples, 95% CI) |
| Multi-seed/prompt evaluation | Mean +/- std across runs | Not implemented |
| Significance testing | t-test or paired bootstrap (strengthens paper) | Not implemented |
| Ablation studies | Signal-by-signal contribution | Per-signal AUROC implemented; full ablation (leave-one-out) not in standard pipeline |

### 3.3 Evaluation Debt: Comprehensive Inventory

Below is a prioritized list of evaluation deficiencies that would undermine credibility in 2025-2026 peer review:

#### Critical (Must Address for Any Venue)

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 1 | **Zero external baselines** — No comparison against any published method | Automatic desk reject at top venues | High — requires implementing or adapting 6+ baselines |
| 2 | **Single model only** (LLaMA-3.1-8B) | Reviewers will question generalizability | Medium — ModelAdapter claims portability but untested |
| 3 | **No RAGTruth evaluation with human annotations** | Missing the field's gold-standard RAG benchmark | Medium — data exists in `data/ragtruth_repo/`, partially evaluated but not in standard pipeline |
| 4 | **Span-level evaluation never tested** | AG-SAR's core output (risky spans) is never validated | Medium — requires RAGTruth span annotations |

#### High Priority (Expected at Top Venues)

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 5 | **Length confounding unresolved** — All signals show negative delta vs length baseline | Undermines signal validity claims | Medium-High — needs principled deconfounding or length-residualized evaluation |
| 6 | **No cross-task transfer evaluation** | Limits practical applicability claims | Medium — RAGTruth provides QA/summarization/data-to-text splits |
| 7 | **No ablation study in standard pipeline** | Cannot justify 5-signal design vs simpler alternatives | Low — per-signal AUROC exists; needs leave-one-out and signal-pair analysis |
| 8 | **Missing Natural Questions dataset** | Standard benchmark missing | Low — similar to TriviaQA loader |
| 9 | **No runtime/efficiency analysis** | Cannot compare against multi-sample methods (SelfCheckGPT) | Low — just timing |

#### Medium Priority (Strengthens Paper)

| # | Gap | Impact | Effort |
|---|-----|--------|--------|
| 10 | **No cross-model evaluation** | Cannot claim architecture portability | Medium — needs Mistral/Qwen support testing |
| 11 | **ECE and AURC implementations unvalidated** | Custom metric implementations may have bugs | Low — replace with torchmetrics |
| 12 | **No Pearson Correlation Coefficient** | Common metric in field (ReDeEP, LUMINA report it) | Low — trivial to add |
| 13 | **No qualitative examples** | Reviewers expect case studies showing span detection | Low — select illustrative examples |
| 14 | **Weak performance on summarization/data-to-text** | RAGTruth summarization AUROC ~0.59, data-to-text ~0.53 | Fundamental — may indicate signal design biased toward QA |

---

## Phase 3.5: External Verification — Key Competitor Analysis

### ReDeEP (ICLR 2025) — Primary Competitor

- **Signals:** 2 (context utilization via copying heads + parametric knowledge via knowledge FFNs)
- **Evaluation:** RAGTruth + Dolly; 3 models; 25+ baselines; AUROC, PCC, Acc, Recall, F1
- **Key results:** AUROC 0.75-0.85 on RAGTruth depending on task type
- **Comparison to AG-SAR:** AG-SAR uses 5 signals vs 2, but ReDeEP's reported RAGTruth AUROC (0.75+) exceeds AG-SAR's current RAGTruth results (0.52-0.75 depending on task). ReDeEP requires multivariate regression (not training-free).

### HARP (arXiv 2025) — Closest Mechanistic Competitor

- **Signals:** Reasoning subspace projection (SVD of unembedding matrix)
- **Key result:** 92.8% AUROC on TriviaQA
- **Comparison to AG-SAR:** HARP's reasoning subspace is conceptually identical to AG-SAR's DPS reasoning basis. AG-SAR adds context subspace and magnitude gating. HARP requires a trained classifier.

### HaMI (NeurIPS 2025)

- **Approach:** Multiple Instance Learning over token representations
- **Key result:** 8-12% AUROC improvement over MARS-SE across 4 datasets, 4 models
- **Comparison to AG-SAR:** Requires labeled training data. AG-SAR's zero-shot design is a clear advantage, but HaMI's multi-dataset/multi-model evaluation sets the standard.

### LUMINA (arXiv 2025)

- **Signals:** Context utilization (distributional distance) + internal knowledge (layer evolution)
- **Key result:** >0.9 AUROC on HalluRAG, +13% over prior SOTA
- **Comparison to AG-SAR:** Similar signal decomposition philosophy but different mechanisms. LUMINA evaluates on RAGTruth + HalluRAG with 4 models and 10+ baselines — substantially more thorough evaluation.

### Frequency-Aware Attention (arXiv 2026)

- **Key result:** Substantially better cross-domain transfer than Lookback Lens
- **Relevance:** Demonstrates that attention-based hallucination detection (AG-SAR's CUS signal) benefits from frequency-domain features — potential enhancement direction.

---

## Summary Scorecard

| Evaluation Dimension | Community Standard (2025-2026) | AG-SAR Current State | Score |
|---------------------|-------------------------------|---------------------|-------|
| **Metric Suite** | AUROC + AUPRC + PCC + F1 | AUROC, AUPRC, TPR@FPR, ECE, Brier, AURC, E-AURC, Risk@90%, Bootstrap CI | **A** (exceeds standard) |
| **Primary Benchmark** | RAGTruth (human-annotated) | TriviaQA + SQuAD (F1-labeled) | **D** (wrong benchmark) |
| **Baseline Comparisons** | 6+ baselines spanning categories | Zero baselines | **F** |
| **Model Diversity** | 3+ models, 2+ families | 1 model | **F** |
| **Granularity** | Response + span/token | Response only (evaluated); span produced but not validated | **C-** |
| **Statistical Rigor** | Bootstrap CI + multi-run std | Bootstrap CI implemented | **B-** |
| **Cross-task Transfer** | Multiple task types | QA only (standard pipeline) | **D** |
| **Ablation Studies** | Signal-by-signal, leave-one-out | Per-signal AUROC only | **C** |
| **Length Confounding** | Controlled or residualized | Identified but unresolved | **D+** |
| **Code Quality** | Reproducible, documented | Well-structured, 2818 LOC, zero-parameter | **A-** |

**Overall Evaluation Readiness:** The metric infrastructure and code quality are strong, but the experimental evaluation falls critically short of 2025-2026 standards. The primary deficits are: (1) zero external baselines, (2) single model, (3) wrong primary benchmark, and (4) unvalidated span detection.

---

## Recommendations: Priority-Ordered Action Plan

### Tier 1: Critical Path to Submission

1. **Implement RAGTruth loader with human annotations** — Integrate the dataset already in `data/ragtruth_repo/`. Evaluate at both response-level and span-level against human annotations. This is the single most impactful change.

2. **Implement minimum baseline suite** — Perplexity (trivial), LN-Entropy (trivial), DoLa standalone (already have the signal), Lookback Lens (attention-based, close to CUS), ReDeEP (use their public code), SelfCheckGPT (public code available).

3. **Multi-model evaluation** — Add LLaMA-2-7B-Chat, LLaMA-2-13B-Chat, and Mistral-7B-Instruct. The ModelAdapter should handle LLaMA variants; Mistral may need adapter extension.

4. **Span-level evaluation** — Implement IoU-based span matching against RAGTruth annotations. Compute span-level precision/recall/F1.

### Tier 2: Strengthening

5. **Address length confounding** — Report delta-above-length-baseline as a standard metric. Consider length-residualized AUROC following the approach in `results/generalization_experiments/`.

6. **Full ablation study** — Leave-one-out signal ablation, signal-pair analysis, fusion mechanism comparison (entropy-gated vs simple mean vs max).

7. **Add Natural Questions** and optionally HalluRAG to the evaluation suite.

8. **Add Pearson Correlation Coefficient** to metrics (standard in ReDeEP, LUMINA).

### Tier 3: Polish

9. **Replace custom ECE/AURC with torchmetrics** implementations for validation.

10. **Add multi-seed evaluation** with mean +/- std reporting.

11. **Runtime analysis** — Report per-sample inference time; compare against SelfCheckGPT (multi-sample) to quantify efficiency advantage.

12. **Qualitative analysis** — Select 5-10 illustrative examples showing correct span detection, missed hallucinations, and false positives.

---

## Appendix A: Existing Results Interpretation

The `results/` directory contains extensive preliminary experiments from an earlier development phase. Key observations:

- **RAGTruth QA (best case):** Ensemble AUROC 0.751, context_grounding 0.690, hidden_norm 0.699
- **RAGTruth Summarization:** All signals weak (0.52-0.63 AUROC). This is a fundamental challenge — the signal architecture may be QA-biased.
- **HaluEval QA:** Inflated (0.999 AUROC) due to severe length confounding (0.988 AUROC from length alone). Unusable for fair evaluation.
- **TruthfulQA:** Weak (0.575). Expected — AG-SAR targets contextual hallucination, TruthfulQA tests parametric knowledge.
- **Length residualization:** Context grounding and hidden_norm are genuinely length-independent (+0.008 and -0.002 AUROC drop after residualization), confirming they capture real signal.
- **Signal combinations:** Best combination (jsd_cand + hidden_norm + context_grounding) marginally beats individual signals (-0.110 vs -0.112 avg delta). Adding more signals hurts (ensemble degradation).

**Warning:** The existing results use earlier signal implementations (not the current 5-signal pipeline). The `results/` data may not reflect the current codebase's performance.

## Appendix B: Reference Papers in Repository

The `papers/` directory contains 8 reference papers with markdown conversions:

| Paper | Relevance |
|-------|-----------|
| Chuang et al. 2024 — Lookback Lens | CUS signal inspiration |
| Chuang et al. 2024 — DoLa | DoLa signal source |
| Sun et al. 2025 — ReDeEP | Primary competitor |
| Hu et al. 2025 — HARP | DPS signal overlap |
| Wu et al. 2024 — Retrieval Heads | Copying head mechanism |
| Li et al. 2024 — Inference-time hallucination | Survey context |
| Orgad et al. 2024 — LLMs know when hallucinating | Theoretical grounding |
| Su et al. 2024 — Unsupervised hallucination | Alternative approach |
| Shi et al. 2023 — Context faithfulness | Problem framing |
| Yuksekgonul et al. 2024 — Attention satisfies | Attention analysis |

**Missing from reference collection:** LUMINA (2025), HaMI (NeurIPS 2025), CHARM (ICLR 2026), Frequency-Aware Attention (2026), FaithJudge (EMNLP 2025).
