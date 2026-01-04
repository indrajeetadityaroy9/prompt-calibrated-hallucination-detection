# ICML / NeurIPS Evaluation Readiness Checklist

**(Final · Auditor-Grade · Crash-Resilient · Reproducible)**

---

## Phase 0 — Environment, Determinism & Dependency Preflight

**Failure Modes:** Non-reproducible numbers, drifting baselines, import-time crashes

### 0.1 Dependency Sanity

* [x] All core libraries import without error:

  ```bash
  python -c "
  import numpy, scipy, sklearn, torch, transformers, sentence_transformers
  from sklearn.metrics import roc_auc_score, average_precision_score
  from sklearn.calibration import calibration_curve
  print('All Science Libraries Importable.')
  "
  ```

* [x] Implemented in `experiments/core/determinism.py:verify_dependencies()`
* [ ] CUDA devices visible and sufficient VRAM confirmed for planned models.

### 0.2 Global Determinism

* [x] **Global seed set at process entry point** (before model or data loading):
  - Implemented in `experiments/core/determinism.py:set_global_seed()`
  - Sets: random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all

* [x] SelfCheckGPT and any sampling-based method run with **fixed seed**.
  - See `experiments/methods/selfcheck.py:SelfCheckNLIMethod.__init__`

* [x] No per-batch reseeding unless explicitly documented.

---

## Phase 1 — Metric Implementation Integrity

**Verify:** `experiments/core/metrics.py`

### 1.1 Required Imports (Exact)

* [x] File contains **exactly**:
  ```python
  from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, auc
  from sklearn.calibration import calibration_curve
  from scipy.stats import spearmanr, bootstrap
  import numpy as np
  ```

### 1.2 Metric Correctness Constraints

* [x] **No custom numerical integration** for AUROC or AUPRC.

* [x] **AUPRC uses**: `average_precision_score(y_true, y_score)` (**not** `auc(recall, precision)`)

* [x] **ECE**: Uses `calibration_curve` + **weighted mean absolute confidence–accuracy gap**

### 1.3 Tensor & Device Safety

* [x] Torch tensors converted safely via `_to_numpy()`:
  ```python
  if isinstance(y_scores, torch.Tensor):
      y_scores = y_scores.detach().cpu().numpy()
  ```

### 1.4 Custom Metric: AURC (Risk–Coverage)

* [x] `_aurc()` matches the **exact vectorized logic**:
  ```python
  desc_indices = np.argsort(y_scores)[::-1]
  cumulative_hallucinations = np.cumsum(y_true_sorted[::-1])[::-1]
  aurc = auc(coverages, risks)
  ```
* [x] No Python loops over samples.

### 1.5 Robustness Logic (NaN & Single-Class Guards)

* [x] **NaN Drop Strategy implemented** before any metric computation:
  ```python
  valid_mask = ~np.isnan(y_scores) & ~np.isinf(y_scores)
  if drop_rate > self.max_nan_rate:
      raise ValueError("Excessive NaNs in metric inputs")
  ```

* [x] **Single-Class Protection**: `_check_single_class()` returns None for degenerate batches

---

## Phase 2 — Baseline Method Fidelity

**Verify:** `experiments/methods/`

### 2.1 SelfCheckGPT (`selfcheck.py`)

* [x] Imports `sentence_transformers.CrossEncoder`
* [x] Uses **pretrained NLI only** (`cross-encoder/nli-deberta-v3-base`)
* [x] ❌ No training loops
* [x] ❌ No custom weight loading
* [x] ✅ Inherits **global seed** from Phase 0

### 2.2 Semantic Entropy (`entropy.py`)

* [x] Uses normalized Shannon entropy via torch
* [x] NaN handling verified (returns NaN for empty response tokens)

### 2.3 Log Probability Baseline (`logprob.py`)

* [x] Uses `torch.log_softmax` (equivalent to `torch.nn.functional.log_softmax`)
* [x] Followed by explicit log-prob gathering
* [x] NaN handling verified (returns NaN for empty response tokens)

---

## Phase 3 — Dataset Configuration, Versioning & Schema

**Verify:** `experiments/data/`

### 3.1 Dataset Logic Integrity

* [x] **HaluEval Balance**: Each row → **two samples** (Fact → 0, Hallucination → 1)

* [x] **RAGTruth Label Mapping**: `hallucination_labels empty → 0`, `non-empty → 1`

* [x] **Refusal Filtering**: Implemented with regex patterns

### 3.2 Dataset Source Integrity

**HaluEval**

* [x] Repo: `pminervini/HaluEval`
* [x] Config: `qa` or `summarization`
* [x] Split: `data`
* [x] Required columns: `knowledge, question, right_answer, hallucinated_answer`

**RAGTruth**

* [x] Repo: `wandb/RAGTruth-processed`
* [x] Split: `test`
* [x] Required columns: `input_str/context, output, hallucination_labels, task_type`

### 3.3 Dataset Verification Script (Hard Gate)

* [x] `scripts/verify_datasets.py` created
* [x] Script exits non-zero on schema mismatch
* [x] Streaming mode used for schema verification

---

## Phase 4 — Model Matrix & Scaling Law Coverage

**Verify:** `src/ag_sar/modeling/hooks.py`

### 4.1 Scaling Law (Model Size)

* [x] `meta-llama/Llama-3.1-8B-Instruct` - supported
* [x] `meta-llama/Llama-3.1-70B-Instruct` - supported with `device_map="balanced"`
* [x] `Qwen/Qwen2.5-72B-Instruct` - supported

### 4.2 Architecture Generalization

* [x] `Qwen/Qwen2.5-32B-Instruct` - supported
* [x] `mistralai/Mistral-Nemo-Instruct-2407` - supported

### 4.3 Mixture-of-Experts

* [x] `mistralai/Mixtral-8x7B-Instruct-v0.1` - supported
* [x] `"Mixtral"` maps to `"llama"` hook in `_detect_architecture()`

---

## Phase 5 — Output Integrity, Streaming & Statistical Rigor

**Verify:** `experiments/core/engine.py`, `experiments/core/logging.py`

### 5.1 Required Metrics (Per-Sample)

* [x] `auroc`, `auprc`, `ece`, `brier`, `aurc` - all implemented

### 5.2 Confidence Intervals

* [x] `ci_lower`, `ci_upper` computed via bootstrap percentile method
* [x] Uses `scipy.stats` for statistical functions

### 5.3 Crash Resilience

* [x] **Output format is JSONL**, not a single JSON array
* [x] Output file is writable incrementally with `flush()` after each write
* [x] **Resume Logic**: `BenchmarkEngine` supports `resume_from` parameter

---

## Phase 6 — Final Pre-Run Gate

**Proceed only if ALL checks above pass**

* [ ] Dataset verification script passes: `python scripts/verify_datasets.py`
* [ ] Determinism confirmed
* [ ] Metrics validated with NaN and single-class guards
* [ ] JSONL streaming verified
* [ ] One end-to-end dry run completes successfully

🚀 **Status:** Pending verification run
🛑 **Otherwise:** Fix the failing phase before execution

---

## How to Verify

```bash
# 1. Run dataset verification
python scripts/verify_datasets.py

# 2. Run metrics tests
pytest tests/experiments/test_metrics_correctness.py -v

# 3. Dry run on small sample
python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml --dry-run

# 4. If all pass, run full experiments
./reproduce_paper.sh
```

---

## Implementation Summary

| Phase | Component | Status |
|-------|-----------|--------|
| 0.1 | Dependency verification | ✅ `experiments/core/determinism.py` |
| 0.2 | Global seed | ✅ `set_global_seed()` |
| 1.1-1.5 | Metrics | ✅ `experiments/core/metrics.py` |
| 2.1 | SelfCheckGPT | ✅ `experiments/methods/selfcheck.py` |
| 2.2 | Entropy | ✅ `experiments/methods/entropy.py` |
| 2.3 | LogProb | ✅ `experiments/methods/logprob.py` |
| 3.1-3.2 | Datasets | ✅ `experiments/data/{halueval,ragtruth}.py` |
| 3.3 | Verification | ✅ `scripts/verify_datasets.py` |
| 4.1-4.3 | Model hooks | ✅ `src/ag_sar/modeling/hooks.py` |
| 5.1-5.3 | JSONL streaming | ✅ `experiments/core/{engine,logging}.py` |
