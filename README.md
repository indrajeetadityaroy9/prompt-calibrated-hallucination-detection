# AG-SAR: Aggregated Signal Architecture for Risk

Zero-shot, training-free hallucination detection for decoder-based autoregressive LLMs via internal signal decomposition.

## Problem

When LLMs generate text, they hallucinate — fabricating facts, misrepresenting sources, or producing content ungrounded in the prompt. Existing detection methods require (a) labeled training data, (b) multiple sampling passes, or (c) external NLI models. None work at inference time with a single forward pass and no prior supervision.

AG-SAR detects hallucinations by intercepting the model's internal computation during generation, extracting five mechanistically-grounded signals that target distinct failure modes along the transformer's processing chain, and fusing them via precision-weighted entropy-gated aggregation into per-token and response-level risk scores — all without training, calibration data, or external models. All signals are context-free: no retrieved documents or context masks required.

## Core Research Contributions

### 1. Causal Signal Decomposition

AG-SAR decomposes hallucination risk along the transformer's processing chain into five signals, each targeting a distinct failure mode:

**Attention Entropy Dispersion (ENT)** — *How specialized are the attention heads?*
Measures bimodality of per-head normalized Shannon entropies across all attention heads. During factual generation, heads develop functional specialization (positional, syntactic, semantic) creating bimodal entropy. During hallucination, this specialization degrades. ENT = 1 - Otsu coefficient of normalized entropies. Range [0,1], higher = riskier.

**MLP Transformation Magnitude (MLP)** — *How much is the MLP changing what attention found?*
All-layer mean of Jensen-Shannon divergence between pre-MLP and post-MLP logit distributions on the candidate token set. Measures parametric intervention magnitude — MLP interventions larger than the model's baseline behavior on a given prompt are suspicious. PIT-normalized against prompt-tail reference values.

**Prompt Subspace Projection (PSP)** — *Is the generation grounded in the prompt?*
Projects output hidden states onto the prompt subspace (SVD of centered prompt hidden states at midpoint layer, rank selected via effective rank). PSP = 1 - ||V_prompt @ h_centered|| / ||h_centered||, magnitude-gated via Rayleigh survival function: gate = 1 - exp(-||h||^2 / tau^2) where tau = median(prompt norms). All-layer mean, PIT-normalized.

**Spectral Phase-Transition Score (SPT)** — *Has the representation manifold collapsed toward noise?*
Sliding-window covariance spectrum of midpoint-layer hidden states. Uses Tracy-Widom calibrated phase-transition detection: SPT(t) = 1 - F_{TW,1}(z_TW), where z_TW = (lambda_1 - mu_TW) / sigma_TW with Marchenko-Pastur centering and finite-sample TW scaling (Johnstone, 2001). Window size derived from effective rank of prompt singular values. Direct mode.

**Spectral Gap** — *How coherent is the signal structure?*
The ratio lambda_2 / (lambda_1 + lambda_2 + eps), bounded in [0, 0.5], computed from the same eigenvalue decomposition as SPT. Near 0 = clean spike separation (single dominant signal direction, low risk). Near 0.5 = degenerate eigenvalues (no preferred direction, high risk). Direct mode.

### 2. Prompt-Anchored Self-Calibration

All signal normalization is derived from the input itself — no hardcoded thresholds, no calibration dataset, no learned parameters:

- **Empirical CDF normalization**: Generation-time signal values are normalized against sorted prompt-tail reference values via rank-based probability integral transform with Haldane-Anscombe continuity correction, producing distribution-free probability estimates.
- **Adaptive tail windowing**: The calibration window scales as sqrt(prompt_length), clamped to prompt_len.
- **Signal-specific normalization**: PSP and MLP use empirical CDF normalization (their raw scales are input-dependent). ENT, SPT, and spectral gap use direct mode (ENT has a semantic [0,1] range; SPT uses its own TW null model; spectral gap has a natural [0, 0.5] range).
- **Cross-signal precision matrix**: A hybrid 5x5 precision matrix Omega = Sigma^{-1} is computed from prompt-tail statistics. PSP-MLP coupling uses a Ledoit-Wolf regularized 2x2 covariance block (signals with genuine per-position variation). ENT, SPT, and spectral gap retain diagonal precision (no per-position calibration data available).

### 3. Training-Free Fusion and Aggregation

Signals are fused into per-token risk scores and aggregated into response-level decisions:

- **Cross-signal precision-weighted fusion**: For each signal i, the effective weight incorporates off-diagonal coupling: w_i(t) = sum_j Omega_ij * e_j(t), where Omega is the cross-signal precision matrix and e_j = (1 - H_j)^kappa is the entropy modulation.
- **Entropy gating**: Signals at p=0.5 (uninformative) receive weight 0 via binary entropy gating. kappa = 1 + median(prompt decisiveness).
- **Signal-first response-level aggregation**: Per-signal mean of raw values, then empirical CDF normalize, then cross-signal precision-weighted entropy-gated fusion.
- **Span detection**: Adaptive bimodal threshold with expected-gap merging for contiguous high-risk span identification.

### 4. Zero-Parameter Design

Every threshold, normalization constant, and aggregation parameter in AG-SAR is derived from either:
- The input data (prompt statistics, prompt SVD rank, cross-signal covariance)
- Model architecture (layer count, hidden dimension)
- Information-theoretic and statistical principles (bimodal thresholding, Tracy-Widom distribution, Marchenko-Pastur law, effective rank via singular value entropy)
- Universal random matrix constants (TW_1 moments, Cornish-Fisher coefficients)

There are no learned weights, no hyperparameters requiring tuning, and no calibration datasets required for operation.

## Architecture

```
Prefill Phase (once per input)
├── Prompt hidden states captured at midpoint layer
├── Compute prompt subspace V_prompt (SVD + effective rank)
├── Compute prompt center, magnitude tau (PSP gate)
├── Compute SPT window size (effective rank of prompt singular values)
├── Collect prompt statistics (reference values + variance via self_calibrate)
│   ├── PSP/MLP: per-position PIT reference values from prompt tail
│   ├── SPT/spectral gap: incremental variance from prompt-tail evaluations
│   └── Cross-signal precision matrix (hybrid 5×5: diagonal + PSP-MLP 2×2 block)
├── Seed SPT generation window from prompt tail
└── Install generation hooks (architecture-adaptive via ModelAdapter)

Generation Phase (per token)
├── ENT: attention entropy dispersion (head specialization bimodality)
├── MLP: candidate-set JSD transformation magnitude (all-layer mean)
├── PSP: prompt subspace projection (all-layer mean)
├── SPT: Tracy-Widom calibrated phase-transition probability
├── Spectral gap: λ₂/(λ₁+λ₂) directional coherence ratio
├── Normalize: empirical CDF (PSP/MLP) or direct (ENT/SPT/spectral gap)
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
│   └── layer_hooks.py             # 2-point per-layer capture (h_resid_attn, h_resid_mlp)
├── signals/
│   ├── ent.py                     # ENT: attention entropy dispersion
│   ├── _jsd_base.py               # MLP: candidate-set JSD transformation magnitude
│   ├── psp.py                     # PSP: prompt subspace projection
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

## Supported Models

AG-SAR targets decoder-based autoregressive language models. The `ModelAdapter` auto-detects architecture via `nn.Module.get_submodule()` probing across five architecture patterns. No retrieved context or context masks are required — all signals derive from prompt conditioning and generation dynamics.

### Verified Model Families

| Model Family | Layers Path | Final Norm | LM Head | Post-Attn Norm | Status |
|---|---|---|---|---|---|
| **LLaMA / Mistral / Qwen / Gemma** | `model.layers` | `model.norm` | `lm_head` | `post_attention_layernorm` | Verified |
| **Phi** | `model.layers` | `model.final_layernorm` | `lm_head` | `post_attention_layernorm` | Supported |
| **GPT-2 / GPT-Neo** | `transformer.h` | `transformer.ln_f` | `lm_head` | `ln_2` | Verified |
| **Falcon** | `transformer.h` | `transformer.ln_f` | `lm_head` | `post_attention_layernorm` | Supported |
| **GPT-NeoX / Pythia** | `gpt_neox.layers` | `gpt_neox.final_layer_norm` | `embed_out` | `post_attention_layernorm` | Supported |

### Hook Point Verification

AG-SAR captures two residual stream points per layer — the state after the attention residual add (`h_resid_attn`) and after the MLP residual add (`h_resid_mlp`). These hook points have been verified against the HuggingFace Transformers source code for each architecture:

**LLaMA 3 / Qwen2** (Pre-Norm architecture, identical residual stream):
```
residual = hidden_states
hidden_states = input_layernorm(hidden_states)
hidden_states = self_attn(hidden_states)
hidden_states = residual + hidden_states           ← h_resid_attn (pre-hook on post_attention_layernorm)
residual = hidden_states
hidden_states = post_attention_layernorm(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states           ← h_resid_mlp (layer output hook)
```

**GPT-2** (`ln_2` serves the same architectural role as `post_attention_layernorm`):
```
residual = hidden_states
hidden_states = ln_1(hidden_states)
hidden_states = attn(hidden_states)
hidden_states = residual + attn_output             ← h_resid_attn (pre-hook on ln_2)
residual = hidden_states
hidden_states = ln_2(hidden_states)
hidden_states = mlp(hidden_states)
hidden_states = residual + hidden_states           ← h_resid_mlp (layer output hook)
```

### Signal Compatibility

| Signal | Requirement | LLaMA 3 | Qwen2 | GPT-2 |
|---|---|---|---|---|
| **ENT** | `output_attentions=True` → `(batch, heads, seq, seq)` per layer | Yes | Yes | Yes |
| **MLP** | 2-point hidden capture + `lm_head` + `final_norm` for logit projection | Yes | Yes | Yes |
| **PSP** | Prompt hidden states at midpoint layer via layer output hook | Yes | Yes | Yes |
| **SPT** | Midpoint layer `h_resid_mlp` via layer output hook | Yes | Yes | Yes |
| **Spectral Gap** | Same SVD as SPT | Yes | Yes | Yes |

### Architecture Notes

- **Attention implementation**: ENT requires raw attention weights via `output_attentions=True`, which is incompatible with Flash Attention 2 (it never materializes the full attention matrix). All models must use `attn_implementation="eager"` (enforced in experiment configs). SDPA falls back to eager internally when `output_attentions=True`.
- **Qwen2 sliding window**: Lower layers use sliding window attention (`max_window_layers`). This concentrates attention distributions on local windows but does not affect the hook system or attention weight return format. ENT correctly captures the different entropy profiles of windowed vs. full-attention layers.
- **GPT-2 weight tying**: `lm_head.weight` is tied to `transformer.wte.weight`, but `lm_head` exists as an `nn.Linear` submodule accessible via `get_submodule("lm_head")`. Weight tying is handled by HF internals, transparent to AG-SAR.
- **GPT-2 Conv1D**: GPT-2 uses `Conv1D` for internal projections instead of `nn.Linear`, but this does not affect hook-level hidden state capture (hooks operate on layer inputs/outputs, not internal projection weights).

## Usage

### Standard Detection

```python
from ag_sar import Detector
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

detector = Detector(model, tokenizer)

result = detector.detect(
    prompt="What is the capital of France? Answer:",
    max_new_tokens=64,
)

print(f"Response: {result.generated_text}")
print(f"Risk: {result.response_risk:.3f}")
print(f"Flagged: {result.is_flagged}")
for span in result.risky_spans:
    print(f"  Risky span [{span.start}:{span.end}]: {span.mean_risk:.3f}")
```

### Detection from Tokens

```python
import torch

input_ids = tokenizer.encode("...", return_tensors="pt").to("cuda")
result = detector.detect_from_tokens(input_ids, max_new_tokens=64)
```

### Score Pre-Existing Text

```python
# Teacher-forced scoring of existing model output (no generation)
result = detector.score(
    prompt="Context: ... Question: ... Answer:",
    response_text="Paris is the capital of France.",
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
- Ledoit, O., & Wolf, M. (2004). "A well-conditioned estimator for large-dimensional covariance matrices." Journal of Multivariate Analysis.
- DerSimonian & Laird (1986). "Meta-analysis in clinical trials." Controlled Clinical Trials.
- Hartung, J., Knapp, G., & Sinha, B. K. (2008). Statistical Meta-Analysis with Applications. Wiley.
- Cornish, E. A., & Fisher, R. A. (1938). "Moments and cumulants in the specification of distributions." Revue de l'Institut International de Statistique.
- Otsu (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. SMC.
- Roy (2007). "Effective Rank: A Measure of Effective Dimensionality." European Signal Processing Conference.

## License

Research use only.
