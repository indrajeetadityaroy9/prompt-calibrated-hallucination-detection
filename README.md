# AG-SAR: Aggregated Signal Architecture for Risk

Zero-shot, training-free hallucination detection for retrieval-augmented LLMs via internal signal decomposition.

## Problem

When LLMs generate text grounded in retrieved context, they hallucinate — fabricating facts, misrepresenting sources, or ignoring provided evidence. Existing detection methods require (a) labeled training data, (b) multiple sampling passes, or (c) external NLI models. None work at inference time with a single forward pass and no prior supervision.

AG-SAR detects hallucinations by intercepting the model's internal computation during generation, extracting five mechanistically-grounded signals that target distinct failure modes along the transformer's processing chain, and fusing them via precision-weighted entropy-gated aggregation into per-token and response-level risk scores — all without training, calibration data, or external models.

## Core Research Contributions

### 1. Causal Signal Decomposition

AG-SAR decomposes hallucination risk along the transformer's processing chain into five signals, each targeting a distinct failure mode:

**Context Utilization Score (CUS)** — *Is the model looking at the context?*
Measures bimodality of per-head lookback ratios (fraction of attention mass on context tokens) across all attention heads. A bimodal distribution indicates grounded generation (some heads attend to context, others don't); a unimodal distribution indicates hallucination. CUS = 1 - inter-class variance ratio. Range [0,1], higher = riskier.

**Parametric Override Score (POS)** — *Is the FFN overriding what attention found?*
Measures the MLP-induced distribution shift via candidate-set Jensen-Shannon divergence (JSD) between pre-MLP and post-MLP residual stream projections, with directional override decomposition relative to the context subspace. When the MLP pushes the representation away from context-grounded directions, parametric knowledge is overriding retrieved evidence.

**Dual-Subspace Projection Score (DPS)** — *Does the representation live in context space or reasoning space?*
Projects output hidden states onto two reference subspaces: a context subspace (SVD of prefill context hidden states, rank selected via effective rank) and a reasoning subspace (bottom singular vectors of the unembedding matrix, rank selected via effective rank). DPS = s_rsn / (s_ctx + s_rsn), with magnitude gating that blends toward 0.5 for low-magnitude states. All-layer mean.

**Spectral Phase-Transition Score (SPT)** — *Has the representation manifold collapsed toward noise?*
Sliding-window covariance spectrum of midpoint-layer hidden states. Uses Tracy-Widom calibrated phase-transition detection: SPT(t) = 1 - F_{TW,1}(z_TW), where z_TW = (lambda_1 - mu_TW) / sigma_TW is the standardized statistic with Marchenko-Pastur centering mu_TW = sigma^2 * (1 + sqrt(gamma))^2 and finite-sample TW scaling sigma_TW = sigma^2 * (1 + sqrt(gamma)) * (1/sqrt(W) + 1/sqrt(d))^{1/3} (Johnstone, 2001). The TW CDF is computed via Cornish-Fisher expansion with exact TW_1 moments (mu ~ -1.2065, sigma ~ 1.2680, skewness ~ 0.2935), providing a smooth probabilistic transition instead of a binary clamp. Window size is data-driven via effective rank of context singular values.

**Spectral Gap** — *How coherent is the signal structure?*
The ratio lambda_2 / (lambda_1 + lambda_2 + eps), bounded in [0, 0.5], computed from the same eigenvalue decomposition as SPT. Near 0 = clean spike separation (single dominant signal direction, low risk). Near 0.5 = degenerate eigenvalues (no preferred direction, high risk). Complementary to SPT: SPT measures whether signal exists above noise; spectral gap measures whether that signal has directional coherence.

### 2. Prompt-Anchored Self-Calibration

All signal normalization is derived from the input itself — no hardcoded thresholds, no calibration dataset, no learned parameters:

- **Empirical CDF normalization**: Generation-time signal values are normalized against sorted prompt-tail reference values via rank-based probability integral transform with continuity correction, producing distribution-free probability estimates. No Gaussian assumption required.
- **Adaptive tail windowing**: The calibration window scales as sqrt(prompt_length), clamped to prompt_len.
- **Signal-specific normalization**: DPS and POS use empirical CDF normalization (their raw scales are input-dependent). CUS, SPT, and spectral gap use direct mode (CUS has a semantic [0,1] range; SPT uses its own TW null model; spectral gap has a natural [0, 0.5] range). SPT/gap variance is computed incrementally from prompt-tail evaluations; falls back to peer-derived variance when insufficient samples exist.
- **Cross-signal precision matrix**: A hybrid 5×5 precision matrix Omega = Sigma^{-1} is computed from prompt-tail statistics. DPS-POS coupling uses a regularized 2×2 covariance block (signals with genuine per-position variation). CUS, SPT, and spectral gap retain diagonal precision (no per-position calibration data available). Tikhonov regularization ensures numerical stability for short prompt tails (Hartung, Knapp & Sinha, 2008).

### 3. Training-Free Fusion and Aggregation

Signals are fused into per-token risk scores and aggregated into response-level decisions:

- **Cross-signal precision-weighted fusion**: For each signal i, the effective weight incorporates off-diagonal coupling: w_i(t) = sum_j Omega_ij * e_j(t), where Omega is the cross-signal precision matrix and e_j = (1 - H_j)^kappa is the entropy modulation. When signals are correlated, the precision matrix naturally downweights redundant information via negative off-diagonal entries. Reduces to diagonal inverse-variance weighting when signals are uncorrelated.
- **Entropy gating**: Signals at p=0.5 (uninformative) receive weight 0 via binary entropy gating. kappa = 1 + median(prompt decisiveness).
- **Signal-first response-level aggregation**: Per-signal mean of raw values, then empirical CDF normalize, then cross-signal precision-weighted entropy-gated fusion.
- **Span detection**: Adaptive bimodal threshold with expected-gap merging for contiguous high-risk span identification.

### 4. Zero-Parameter Design

Every threshold, normalization constant, and aggregation parameter in AG-SAR is derived from either:
- The input data (prompt statistics, context SVD rank, cross-signal covariance)
- Model architecture (unembedding matrix SVD, layer count)
- Information-theoretic and statistical principles (bimodal thresholding, Tracy-Widom distribution, Marchenko-Pastur law, effective rank via singular value entropy)
- Universal random matrix constants (TW_1 moments, Cornish-Fisher coefficients)

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
│   ├── DPS/POS: per-position PIT reference values from prompt tail
│   ├── SPT/spectral gap: incremental variance from prompt-tail evaluations
│   └── Cross-signal precision matrix (hybrid 5×5: diagonal + DPS-POS 2×2 block)
├── Seed SPT generation window from prompt tail
└── Install generation hooks (architecture-adaptive via ModelAdapter)

Generation Phase (per token)
├── CUS: lookback ratio bimodality (inter-class variance ratio)
├── POS: candidate-set JSD + directional override decomposition
├── DPS: dual-subspace projection ratio (all-layer mean)
├── SPT: Tracy-Widom calibrated phase-transition probability
├── Spectral gap: λ₂/(λ₁+λ₂) directional coherence ratio
├── Normalize: empirical CDF (DPS/POS) or direct (CUS/SPT/spectral gap)
├── Fuse: cross-signal precision-weighted entropy-gated mean
└── Aggregate: signal-first response risk

Post-Processing
└── Span merger (adaptive bimodal threshold + expected-gap merging)
```

## Project Structure

```
ag_sar/                            # Core library
├── __init__.py                    # Public API: Detector, TokenSignals, DetectionResult
├── config.py                      # TokenSignals (5 signals), DetectionResult
├── numerics.py                    # jsd, effective_rank, otsu_threshold, tracy_widom_cdf, EPS
├── detector.py                    # Detector: prefill → 5 signals per token → aggregate
├── calibration.py                 # Self-calibration: PIT stats, cross-signal precision matrix
├── hooks/
│   ├── adapter.py                 # ModelAdapter: architecture auto-detection (5 model families)
│   ├── buffer.py                  # EphemeralHiddenBuffer (bfloat16, cleared per token)
│   ├── layer_hooks.py             # 2-point per-layer capture (h_resid_attn, h_resid_mlp)
│   └── prefill_hooks.py           # PrefillContextHook (boolean context_mask)
├── signals/
│   ├── cus.py                     # CUS: lookback ratio bimodality
│   ├── _jsd_base.py               # POS: candidate-set JSD + directional override
│   ├── dps.py                     # DPS: dual-subspace projection
│   └── spt.py                     # SPT: TW-calibrated phase-transition + spectral gap
└── aggregation/
    ├── fusion.py                  # Cross-signal precision-weighted entropy-gated fusion
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
    ├── main.yaml                  # Default: all 5 signals, LLaMA-3.1-8B
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

- Tracy, C. A., & Widom, H. (1994). "Level-spacing distributions and the Airy kernel." Communications in Mathematical Physics.
- Johnstone, I. M. (2001). "On the distribution of the largest eigenvalue in principal components analysis." Annals of Statistics.
- Baik, J., Ben Arous, G., & Péché, S. (2005). "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices." Annals of Probability.
- DerSimonian & Laird (1986). "Meta-analysis in clinical trials." Controlled Clinical Trials.
- Hartung, J., Knapp, G., & Sinha, B. K. (2008). Statistical Meta-Analysis with Applications. Wiley.
- Cornish, E. A., & Fisher, R. A. (1938). "Moments and cumulants in the specification of distributions." Revue de l'Institut International de Statistique.
- Wu et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality." ICLR 2025.
- Chuang et al. (2024). "Lookback Lens: Detecting and Mitigating Contextual Hallucinations." EMNLP 2024.
- Sun et al. (2025). "ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability." ICLR 2025.
- Ettori et al. (2026). "Spectral Geometry for Deep Learning: Compression and Hallucination Detection via Random Matrix Theory." arXiv:2601.17357.
- Ali et al. (2025). "Spectral Filtering for LLM Representations." arXiv:2511.12220.
- Bloemendal et al. (2026). "Random Matrix Theory for Large Language Models." arXiv:2602.22345.
- Otsu (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. SMC.
- Roy (2007). "Effective Rank: A Measure of Effective Dimensionality." European Signal Processing Conference.

## License

Research use only.
