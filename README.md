# AG-SAR: Aggregated Signal Architecture for Risk

Zero-shot, training-free hallucination detection for retrieval-augmented LLMs via internal signal decomposition.

## Problem

When LLMs generate text grounded in retrieved context, they hallucinate — fabricating facts, misrepresenting sources, or ignoring provided evidence. Existing detection methods require (a) labeled training data, (b) multiple sampling passes, or (c) external NLI models. None work at inference time with a single forward pass and no prior supervision.

AG-SAR detects hallucinations by intercepting the model's internal computation during generation, extracting four independent signals that target distinct failure modes along the transformer's processing chain, and fusing them into per-token and response-level risk scores — all without training, calibration data, or external models.

## Core Research Contributions

### 1. Causal Signal Decomposition

AG-SAR decomposes hallucination risk along the transformer's processing chain into four signals, each targeting a distinct failure mode:

**Context Utilization Score (CUS)** — *Is the model looking at the context?*
Measures bimodality of per-head lookback ratios (fraction of attention mass on context tokens) across all attention heads. A bimodal distribution indicates grounded generation (some heads attend to context, others don't); a unimodal distribution indicates hallucination. CUS = 1 - inter-class variance ratio. Range [0,1], higher = riskier.

**Parametric Override Score (POS)** — *Is the FFN overriding what attention found?*
Measures the MLP-induced distribution shift via candidate-set Jensen-Shannon divergence (JSD) between pre-MLP and post-MLP residual stream projections, with directional override decomposition relative to the context subspace. When the MLP pushes the representation away from context-grounded directions, parametric knowledge is overriding retrieved evidence.

**Dual-Subspace Projection Score (DPS)** — *Does the representation live in context space or reasoning space?*
Projects output hidden states onto two reference subspaces: a context subspace (SVD of prefill context hidden states, rank selected via effective rank) and a reasoning subspace (bottom singular vectors of the unembedding matrix, rank selected via effective rank). DPS = s_rsn / (s_ctx + s_rsn), with magnitude gating that blends toward 0.5 for low-magnitude states. All-layer mean.

**Spectral Phase-Transition Score (SPT)** — *Has the representation manifold collapsed toward noise?*
Sliding-window covariance spectrum of midpoint-layer hidden states. Measures how far the leading eigenvalue departs from the theoretical noise ceiling derived from random matrix theory. When the leading eigenvalue reabsorbs into the noise bulk, the model has lost its signal structure. SPT = 1 - clamp((lambda_1 - lambda_+) / lambda_+, 0, 1) where lambda_+ = sigma^2 * (1 + sqrt(d/W))^2. Window size is data-driven via effective rank of context singular values.

### 2. Prompt-Anchored Self-Calibration

All signal normalization is derived from the input itself — no hardcoded thresholds, no calibration dataset, no learned parameters:

- **Empirical CDF normalization**: Generation-time signal values are normalized against sorted prompt-tail reference values via rank-based probability integral transform with continuity correction, producing distribution-free probability estimates. No Gaussian assumption required.
- **Adaptive tail windowing**: The calibration window scales as sqrt(prompt_length), clamped to prompt_len.
- **Signal-specific normalization**: DPS and POS use empirical CDF normalization (their raw scales are input-dependent). CUS and SPT use direct mode (CUS has a semantic [0,1] range; SPT uses its own theoretical noise boundary as a null model). Peer-derived variance for precision weighting.

### 3. Training-Free Fusion and Aggregation

Signals are fused into per-token risk scores and aggregated into response-level decisions:

- **Entropy-gated precision-weighted fusion**: w_i(t) = (1/var_i) x (1 - H_i(t))^kappa, where H_i is binary entropy and 1/var_i is inverse-variance weighting. Signals at p=0.5 (uninformative) receive weight 0 via entropy gating. kappa = 1 + median(prompt decisiveness).
- **Signal-first response-level aggregation**: Per-signal mean of raw values, then empirical CDF normalize, then precision x entropy weighted fusion.
- **Span detection**: Adaptive bimodal threshold with expected-gap merging for contiguous high-risk span identification.

### 4. Zero-Parameter Design

Every threshold, normalization constant, and aggregation parameter in AG-SAR is derived from either:
- The input data (prompt statistics, context SVD rank)
- Model architecture (unembedding matrix SVD, layer count)
- Information-theoretic and statistical principles (bimodal thresholding, random matrix theory noise bounds, effective rank via singular value entropy)

There are no learned weights, no hyperparameters requiring tuning, and no calibration datasets required for operation.

## Architecture

```
Prefill Phase (once per input)
├── Context capture at midpoint layer
├── Compute context subspace V_ctx (SVD + effective rank)
├── Compute reasoning subspace V_rsn (bottom SVD of unembedding matrix + effective rank)
├── Compute SPT window size (effective rank of context singular values)
├── Set prompt center (non-context tokens), magnitude tau (DPS gate)
├── Collect prompt statistics (reference values + variance via self_calibrate)
├── Seed SPT generation window from prompt tail
└── Install generation hooks (architecture-adaptive via ModelAdapter)

Generation Phase (per token)
├── CUS: lookback ratio bimodality (inter-class variance ratio)
├── POS: candidate-set JSD + directional override decomposition
├── DPS: dual-subspace projection ratio (all-layer mean)
├── SPT: spectral phase-transition (random matrix theory noise boundary)
├── Normalize: empirical CDF (DPS/POS) or direct (CUS/SPT)
├── Fuse: entropy-gated precision-weighted mean
└── Aggregate: signal-first response risk

Post-Processing
└── Span merger (adaptive bimodal threshold + expected-gap merging)
```

## Project Structure

```
ag_sar/                            # Core library
├── __init__.py                    # Public API: Detector, TokenSignals, DetectionResult
├── config.py                      # TokenSignals (4 signals), DetectionResult
├── numerics.py                    # jsd, effective_rank, otsu_threshold, EPS
├── detector.py                    # Detector: prefill → 4 signals per token → aggregate
├── calibration.py                 # Self-calibration: empirical CDF stats, adaptive window
├── hooks/
│   ├── adapter.py                 # ModelAdapter: architecture auto-detection (5 model families)
│   ├── buffer.py                  # EphemeralHiddenBuffer (bfloat16, cleared per token)
│   ├── layer_hooks.py             # 2-point per-layer capture (h_resid_attn, h_resid_mlp)
│   └── prefill_hooks.py           # PrefillContextHook (boolean context_mask)
├── signals/
│   ├── cus.py                     # CUS: lookback ratio bimodality
│   ├── _jsd_base.py               # POS: candidate-set JSD + directional override
│   ├── dps.py                     # DPS: dual-subspace projection
│   └── spt.py                     # SPT: spectral phase-transition (random matrix theory)
└── aggregation/
    ├── fusion.py                  # Entropy-gated precision-weighted fusion
    └── spans.py                   # Adaptive span detection + expected-gap merging

experiments/                       # Experiment orchestration
├── eval.py                        # CLI entry point (--config dispatch)
├── run_eval.py                    # Evaluation orchestration with YAML config
├── run_ablation.py                # Leave-one-out signal ablation study
├── schema.py                      # Typed YAML config dataclasses
├── common.py                      # Model loading, dataset dispatch, output
├── loaders.py                     # Dataset loaders (TriviaQA, SQuAD)
├── metrics.py                     # AUROC, AUPRC, TPR@FPR, ECE, AURC, bootstrap CI
├── answer_matching.py             # SQuAD-style F1 answer matching
└── configs/
    ├── main.yaml                  # Default: all 4 signals, LLaMA-3.1-8B
    └── ablation.yaml              # Leave-one-out signal ablation config
```

## Installation

```bash
pip install -e .
```

Requires Python >= 3.10, PyTorch >= 2.5.1, and Transformers >= 4.57.0.

## Usage

### Standard QA Detection

```python
from ag_sar import Detector
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

detector = Detector(model, tokenizer)

result = detector.detect(
    question="What is the capital of France?",
    context="France is a country in Western Europe. Its capital is Paris.",
    max_new_tokens=64,
)

print(f"Response: {result.generated_text}")
print(f"Risk: {result.response_risk:.3f}")
print(f"Flagged: {result.is_flagged}")
for span in result.risky_spans:
    print(f"  Risky span [{span.start}:{span.end}]: {span.mean_risk:.3f}")
```

### Format-Agnostic Detection

```python
import torch

# Any prompt format — just provide input_ids and a boolean context mask
input_ids = tokenizer.encode("...", return_tensors="pt")
context_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
context_mask[start:end] = True  # Mark context token positions

result = detector.detect_from_tokens(input_ids, context_mask, max_new_tokens=64)
```

### Score Pre-Existing Text

```python
# Teacher-forced scoring of existing model output (no generation)
result = detector.score(
    prompt="Context: ... Question: ... Answer:",
    response_text="Paris is the capital of France.",
    context_text="France is a country in Western Europe. Its capital is Paris.",
)
print(f"Risk: {result.response_risk:.3f}")
```

### Running Experiments

```bash
# Evaluation on QA datasets
python -m experiments --config experiments/configs/main.yaml

# Signal ablation study
python -m experiments --config experiments/configs/ablation.yaml
```

## References

Key works that inform AG-SAR's design:

- Wu et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality." ICLR 2025.
- Chuang et al. (2024). "Lookback Lens: Detecting and Mitigating Contextual Hallucinations." EMNLP 2024.
- Sun et al. (2025). "ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability." ICLR 2025.
- Ettori et al. (2026). "Spectral Analysis of Transformer Hidden States." arXiv:2601.17357.
- Ali et al. (2025). "Spectral Filtering for LLM Representations." arXiv:2511.12220.
- Bloemendal et al. (2026). "Random Matrix Theory for Large Language Models." arXiv:2602.22345.
- DerSimonian & Laird (1986). "Meta-analysis in clinical trials." Controlled Clinical Trials.
- Otsu (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. SMC.
- Roy (2007). "Effective Rank: A Measure of Effective Dimensionality." European Signal Processing Conference.

## License

Research use only.
