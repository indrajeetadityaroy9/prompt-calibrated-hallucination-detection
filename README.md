# AG-SAR: Aggregated Signal Architecture for Risk

Zero-shot, training-free hallucination detection for retrieval-augmented LLMs via mechanistic signal decomposition.

## Problem

When LLMs generate text grounded in retrieved context, they hallucinate --- fabricating facts, misrepresenting sources, or ignoring provided evidence. Existing detection methods either require (a) labeled training data, (b) multiple sampling passes, or (c) external NLI models. None of these work at inference time with a single forward pass and no prior supervision.

AG-SAR detects hallucinations by intercepting the model's internal computation during generation, extracting six mechanistically-grounded signals, and fusing them into a per-token risk score --- all without training, calibration data, or external models.

## Core Research Contributions

### 1. Causal Signal Decomposition

AG-SAR decomposes hallucination risk along the transformer's causal processing chain into six independent signals, each targeting a distinct failure mode:

**Context Utilization Score (CUS)** --- *Is the model looking at the context?*
Measures bimodality of per-head lookback ratios across all attention heads, weighted by copying affinity from prefill. Bimodal distribution (some heads attend to context, others don't) indicates grounded generation. Unimodal distribution indicates hallucination. CUS = 1 - weighted Otsu coefficient. Range [0,1], higher = riskier.

**Parametric Override Score (POS)** --- *Is the FFN overriding what attention found?*
Measures MLP-induced distribution shift via candidate-set JSD between pre-FFN and post-FFN residual stream projections, with directional override decomposition relative to the context subspace. When the FFN pushes the representation away from context-grounded directions, parametric knowledge is overriding retrieved evidence.

**Dual-Subspace Projection Score (DPS)** --- *Does the representation live in context space or reasoning space?*
Projects output hidden states onto two orthogonal reference subspaces: a context subspace (SVD of prefill context hidden states, rank selected via Marchenko-Pastur boundary) and a reasoning subspace (bottom singular vectors of the unembedding matrix). DPS = s_rsn / (s_ctx + s_rsn), with magnitude gating. Layer selection is data-driven via variance-based Otsu thresholding.

**DoLa (Layer-Contrast Score)** --- *Did late layers add factual content?*
log P_final(token) - log P_premature(token), where premature layer = argmax JSD over early layers. High DoLa = model added factual content in late layers = safe (lower_is_riskier). Polarity is inverted in the aggregation pipeline.

**Context-Grounding Direction (CGD)** --- *Is generation moving toward or away from context?*
Cosine between (h_gen - prompt_center) and (context_center - prompt_center). Measures whether generation moves toward or away from context representation. (1 - cos_sim)/2 in [0,1], higher = away from context = riskier.

**Semantic Trajectory Dynamics (STD)** --- *Are hidden states evolving consistently across layers?*
Measures directional consistency and velocity asymmetry of context-projected hidden state trajectories across transformer layers. Factual tokens exhibit smooth convergent trajectories; hallucinated tokens show oscillatory divergent dynamics. Uses post-norm hidden states (h_mlp_in) to avoid residual norm growth artifacts. Inspired by LSD (arXiv 2510.04933), adapted training-free.

### 2. Prompt-Anchored Self-Calibration

All signal normalization is derived from the input itself --- no hardcoded thresholds, no calibration dataset, no learned parameters:

- **Probability Integral Transform (PIT)**: Generation-time signal values are normalized via empirical CDF against sorted prompt-tail reference values, producing distribution-free probability estimates with Haldane-Anscombe correction. No Gaussian assumption required.
- **MAD-based robust estimation**: Median Absolute Deviation replaces standard deviation for outlier-resilient variance estimates.
- **Adaptive tail windowing**: The calibration window scales as sqrt(prompt_length), clamped to [4, prompt_len//2], balancing estimation accuracy against locality.
- **Signal-specific normalization**: CUS uses direct probability mapping (its [0,1] range has semantic meaning); POS, DPS, DoLa, CGD, and STD use PIT normalization (their raw scales are input-dependent). DoLa polarity is inverted (`higher_is_riskier=False`) since high layer-contrast indicates factual content.
- **Data-driven DPS layer selection**: Variance-based Otsu thresholding selects discriminative layers per input, replacing fixed heuristics.

### 3. Training-Free Fusion and Aggregation

Signals are fused into per-token risk scores and aggregated into response-level decisions:

- **Conflict-aware precision-weighted entropy-gated fusion**: w_i(t) = precision_i x (1 - H_i(t))^kappa, where H_i = binary entropy and precision_i = 1/variance_i (inverse-variance weighting, DerSimonian & Laird 1986). Signals at p=0.5 (uninformative) get weight 0 via entropy gating. Signals with low prompt variance get higher precision weight. Conflict coefficient measures inter-signal disagreement as a diagnostic.
- **Precision-weighted response-level aggregation**: Signal-first aggregation (per-signal mean of raw values, then PIT normalize, then precision x entropy weighted fusion), capturing diffuse mean shifts that per-token entropy gating suppresses.
- **Span detection**: Contiguous high-risk tokens are merged into spans via bimodality-adaptive Tukey fences (multiplier = 1.5 x (1 - Otsu bimodality coefficient)), with expected-gap-based merging. Strongly bimodal risk distributions yield aggressive thresholds; unimodal distributions yield conservative (standard Tukey) thresholds.

### 4. Zero-Parameter Design

Every threshold, normalization constant, and aggregation parameter in AG-SAR is derived from either:
- The input data (prompt statistics, context SVD rank, copying head identification)
- Model architecture (unembedding matrix SVD, layer count)
- Information-theoretic principles (Otsu's method, Marchenko-Pastur law, order statistics)

There are no learned weights, no hyperparameters requiring tuning, and no calibration datasets required for operation.

## Architecture

```
Prefill Phase (once per input)
├── Multi-layer context capture at 3 candidate layers (spectral gap selection)
├── Identify copying heads (Otsu on context-to-context attention affinity)
├── Compute context subspace V_ctx (SVD + Marchenko-Pastur rank selection)
├── Compute reasoning subspace V_rsn (bottom SVD of unembedding matrix)
├── Collect prompt statistics (sorted reference values + variance per signal via PIT)
├── Data-driven DPS layer selection (variance-based Otsu)
├── Set prompt center (CGD), magnitude tau (DPS gate)
├── Set context basis for STD signal (context-projected trajectories)
└── Set up generation hooks (architecture-adaptive via ModelAdapter)

Generation Phase (per token)
├── CUS: affinity-weighted lookback ratio bimodality (Otsu coefficient)
├── POS: candidate-set JSD + directional override decomposition
├── DPS: dual-subspace projection ratio (data-selected layers)
├── DoLa: layer-contrast score (polarity-inverted)
├── CGD: context-grounding direction cosine
├── STD: context-projected directional inconsistency + divergence asymmetry
├── Normalize: PIT against prompt-tail reference (with polarity flip for DoLa)
├── Fuse: conflict-aware precision-weighted entropy-gated mean
└── Aggregate: precision-weighted signal-first response risk

Post-Processing
└── Span merger (bimodality-adaptive Tukey fence on token risks)
```

## Project Structure

```
ag_sar/                            # Core library (novel contribution)
├── __init__.py                    # Public API: Detector, TokenSignals, DetectionResult, RiskySpan
├── config.py                      # TokenSignals (6 signals), DetectionResult
├── numerics.py                    # safe_jsd, otsu_threshold, safe_softmax, named constants
├── detector.py                    # Detector: prefill → 6 signals per token → aggregate
├── calibration.py                 # Self-calibration: PIT stats, adaptive window, DPS layer selection
├── hooks/
│   ├── adapter.py                 # ModelAdapter: architecture auto-detection (6 model families)
│   ├── buffer.py                  # EphemeralHiddenBuffer (bfloat16, cleared per token)
│   ├── layer_hooks.py             # 3-point per-layer capture (h_resid_attn, h_mlp_in, h_resid_mlp)
│   └── prefill_hooks.py           # PrefillContextHook (boolean mask), PrefillStatisticsHook
├── signals/
│   ├── cus.py                     # CUS: affinity-weighted lookback ratio bimodality
│   ├── _jsd_base.py               # POS + DoLa: candidate-set JSD, directional override, layer contrast
│   ├── dps.py                     # DPS + CGD: dual-subspace projection + grounding direction
│   └── std.py                     # STD: semantic trajectory dynamics (context-projected)
└── aggregation/
    ├── fusion.py                  # Precision-weighted entropy-gated fusion (DerSimonian & Laird)
    └── spans.py                   # Bimodality-adaptive Tukey fence span detection

evaluation/                        # Evaluation harness (separate from core)
├── metrics.py                     # AUROC, AUPRC, TPR@FPR, ECE, AURC, bootstrap CI
├── answer_matching.py             # SQuAD-style F1 answer matching
├── input_builder.py               # Input tokenization (legacy, retained for backward compat)
└── loaders/                       # Dataset loaders (TriviaQA, SQuAD)

experiments/                       # Experiment orchestration
├── configs/                       # YAML experiment configurations
│   ├── default.yaml               # Default: all 6 signals, LLaMA-3.1-8B
│   └── ablation.yaml              # Leave-one-out signal ablation config
├── run_evaluation.py              # Evaluation with YAML config + CLI override support
└── ablation.py                    # Signal ablation study (delta-AUROC per signal)

docs/                              # Research documentation
├── objectives.md                  # Research objectives & canonical code mapping
└── evaluation_analysis.md         # SOTA gap analysis & evaluation audit
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
    print(f"  Risky span [{span.start_token}:{span.end_token}]: {span.risk_score:.3f}")
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

## Related Work

AG-SAR occupies a unique position in the hallucination detection landscape: **zero-shot, training-free, multi-signal internal detection**. The table below situates AG-SAR against the closest published methods as of early 2026.

| Method | Venue | Signals | Training-Free? | Zero-Shot? |
|--------|-------|---------|----------------|------------|
| **AG-SAR** | — | 6 (CUS, POS, DPS, DoLa, CGD, STD) | Yes | Yes |
| ReDeEP (Sun et al.) | ICLR 2025 | 2 (CKS, PKS) | No (regression) | No |
| CHARM (Frasca et al.) | ICLR 2026 | Attention + activation graphs | No (GNN) | No |
| HARP (Hu et al.) | arXiv 2025 | Subspace projection | No (classifier) | No |
| HaMI (Niu et al.) | NeurIPS 2025 | Hidden-state MIL | No (MIL) | No |
| Lookback Lens (Chuang et al.) | EMNLP 2024 | Attention (lookback ratio) | No (linear) | No |
| DoLa (Chuang et al.) | ICLR 2024 | Layer contrast | Yes | Yes |
| SelfCheckGPT (Manakul et al.) | EMNLP 2023 | Black-box sampling | Yes | Yes |

**Key differentiators**: No published method combines six mechanistically-distinct internal signals in a training-free framework. ReDeEP (closest mechanistic competitor) uses two signals but requires multivariate regression. CHARM (ICLR 2026) subsumes attention-based heuristics via graph neural networks but requires training. HARP's reasoning subspace projection is conceptually close to AG-SAR's DPS signal but requires a trained classifier. AG-SAR's value proposition is deployment without labeled hallucination data — it works on any new input, model, or domain immediately.

## References

Key works that inform AG-SAR's design:

- Wu et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality." ICLR 2025.
- Chuang et al. (2024). "DoLa: Decoding by Contrasting Layers Improves Factuality." ICLR 2024.
- Chuang et al. (2024). "Lookback Lens: Detecting and Mitigating Contextual Hallucinations." EMNLP 2024.
- Sun et al. (2025). "ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability." ICLR 2025.
- Hu et al. (2025). "HARP: Hallucination Detection via Reasoning Subspace Projection." arXiv:2509.11536.
- Frasca et al. (2026). "CHARM: Graph-Based Hallucination Detection via Attributed Attention Maps." ICLR 2026.
- Niu et al. (2025). "HaMI: Hallucination Detection via Multiple Instance Learning." NeurIPS 2025.
- Chen et al. (2025). "Layer-wise Semantic Divergence for Hallucination Detection." arXiv:2510.04933.
- DerSimonian & Laird (1986). "Meta-analysis in clinical trials." Controlled Clinical Trials.

## License

Research use only.
