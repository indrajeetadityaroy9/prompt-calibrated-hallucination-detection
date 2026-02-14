# AG-SAR: Aggregated Signal Architecture for Risk

Zero-shot, training-free hallucination detection for retrieval-augmented LLMs via mechanistic signal decomposition.

## Problem

When LLMs generate text grounded in retrieved context, they hallucinate --- fabricating facts, misrepresenting sources, or ignoring provided evidence. Existing detection methods either require (a) labeled training data, (b) multiple sampling passes, or (c) external NLI models. None of these work at inference time with a single forward pass and no prior supervision.

AG-SAR detects hallucinations by intercepting the model's internal computation during generation, extracting three mechanistically-grounded signals, and fusing them into a per-token risk score --- all without training, calibration data, or external models.

## Core Research Contributions

### 1. Causal Signal Decomposition

AG-SAR decomposes hallucination risk along the transformer's causal processing chain into three independent signals, each targeting a distinct failure mode:

**Context Utilization Score (CUS)** --- *Is the model looking at the context?*
Identifies copying heads during prefill via Otsu-thresholded attention affinity, then tracks their context attention mass during generation. Built on the retrieval head mechanism (Wu et al., 2024): fewer than 5% of attention heads are responsible for faithfully retrieving context, and their attention patterns are directly measurable.

**Parametric Override Score (POS)** --- *Is the FFN overriding what attention found?*
Measures MLP-induced distribution shift via candidate-set JSD between pre-FFN and post-FFN residual stream projections, then decomposes the shift direction relative to the context subspace. When the FFN pushes the representation away from context-grounded directions, parametric knowledge is overriding retrieved evidence.

**Dual-Subspace Projection Score (DPS)** --- *Does the representation live in context space or reasoning space?*
Projects output hidden states onto two orthogonal reference subspaces: a context subspace (SVD of prefill context hidden states, rank selected via Marchenko-Pastur boundary) and a reasoning subspace (bottom singular vectors of the unembedding matrix, following the HARP finding that ~5% of representation dimensions encode reasoning). DPS = s_rsn / (s_ctx + s_rsn), measuring the balance between contextual grounding and parametric reasoning.

### 2. Prompt-Anchored Self-Calibration

All signal normalization is derived from the input itself --- no hardcoded thresholds, no calibration dataset, no learned parameters:

- **Per-input z-scoring**: Each signal's mean and variance are estimated from the prefill pass (the model processing the prompt). Generation-time values are z-scored against these prompt statistics, then mapped to probabilities via sigmoid.
- **MAD-based robust estimation**: Median Absolute Deviation replaces standard deviation for outlier-resilient variance estimates.
- **Adaptive tail windowing**: The calibration window scales as sqrt(prompt_length), balancing estimation accuracy against locality.
- **Signal-specific normalization**: CUS uses direct probability mapping (its [0,1] range has semantic meaning); POS and DPS use z-score normalization (their raw scales are input-dependent).

### 3. Training-Free Fusion and Aggregation

Signals are fused into per-token risk scores and aggregated into response-level decisions:

- **Noisy-OR fusion**: R(t) = 1 - product(1 - p_i(t)), interpreting risk as "probability that at least one signal indicates hallucination" under an independence assumption.
- **Adaptive quantile aggregation**: Response-level risk uses the q(n) = 1 - 1/(n+1) quantile of token risks, an order-statistics approach that naturally adapts to response length.
- **Span detection**: Contiguous high-risk tokens are merged into spans via Tukey fences (Q3 + 0.5 x IQR), identifying localized hallucination regions.
- **Conformal calibration** (optional): Split conformal prediction provides finite-sample coverage guarantees for the binary detection threshold when a small calibration set is available.

### 4. Zero-Parameter Design

Every threshold, normalization constant, and aggregation parameter in AG-SAR is derived from either:
- The input data (prompt statistics, context SVD rank, copying head identification)
- Model architecture (unembedding matrix SVD, layer count)
- Information-theoretic principles (Otsu's method, Marchenko-Pastur law, order statistics)

There are no learned weights, no hyperparameters requiring tuning, and no calibration datasets required for operation. The only user-facing configuration is the architectural choice of which layers to hook.

## Architecture

```
Prefill Phase (once per input)
├── Identify copying heads (Otsu on context-to-context attention affinity)
├── Compute context subspace V_ctx (SVD + Marchenko-Pastur rank selection)
├── Compute reasoning subspace V_rsn (bottom SVD of unembedding matrix)
├── Collect prompt statistics (mu, sigma per signal via MAD)
└── Set up generation hooks

Generation Phase (per token)
├── CUS: context attention mass over identified copying heads
├── POS: candidate-set JSD + directional override decomposition
├── DPS: dual-subspace projection ratio
├── Normalize: z-score against prompt stats → sigmoid
├── Fuse: Noisy-OR across signals
└── Aggregate: adaptive quantile → response risk

Post-Processing
├── Span merger (Tukey fence on token risks)
└── Conformal threshold (optional, if calibration data available)
```

## Project Structure

```
ag_sar/
├── __init__.py                    # Package exports
├── config.py                      # DSGConfig, DSGTokenSignals, DetectionResult
├── numerics.py                    # Numerical primitives (safe_jsd, otsu_threshold, mad_sigma)
├── hooks.py                       # PyTorch forward hooks for hidden state capture
├── signals/
│   ├── copying_heads.py           # CUS: copying head identification + context attention
│   ├── topk_jsd.py                # POS: candidate-set JSD + directional override
│   └── context_grounding.py       # DPS: dual-subspace projection score
├── aggregation/
│   ├── prompt_anchored.py         # Noisy-OR fusion + adaptive quantile aggregation
│   ├── span_merger.py             # Token-to-span risk grouping
│   └── conformal.py               # Split conformal prediction calibration
├── evaluation/
│   ├── modes.py                   # Forced decoding + generation evaluation modes
│   ├── runner.py                  # Evaluation pipeline orchestration
│   ├── metrics.py                 # AUROC, AUPRC, TPR@FPR, ECE, AURC, bootstrap CI
│   └── token_aligner.py           # Align token-level signals with span-level labels
└── icml/
    └── dsg_detector.py            # Main DSGDetector: prefill → detect → aggregate
```

## Installation

```bash
pip install -e .
```

Requires Python >= 3.11, PyTorch >= 2.5.1, and Transformers >= 4.57.0.

## Usage

```python
from ag_sar.icml.dsg_detector import DSGDetector
from ag_sar.config import DSGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

detector = DSGDetector(model, tokenizer, DSGConfig())

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

## Evaluation

AG-SAR supports two evaluation modes:

- **Forced decoding**: Feeds ground-truth response tokens to compute signals at each position, enabling alignment with span-level hallucination labels (e.g., RAGTruth).
- **Generation**: Standard autoregressive generation with integrated detection, for response-level evaluation (e.g., HaluEval).

Metrics include AUROC, AUPRC, TPR@5%FPR, Expected Calibration Error, Brier score, AURC/E-AURC (selective prediction), span-level precision/recall with IoU matching, and bootstrap confidence intervals.

## References

Key works that inform AG-SAR's design:

- Wu et al. (2024). "Retrieval Head Mechanistically Explains Long-Context Factuality." ICLR 2025.
- Chuang et al. (2024). "DoLa: Decoding by Contrasting Layers Improves Factuality." ICLR 2024.
- Chuang et al. (2024). "Lookback Lens: Detecting and Mitigating Contextual Hallucinations." EMNLP 2024.
- Sun et al. (2024). "ReDeEP: Detecting Hallucination in RAG via Mechanistic Interpretability." ICLR 2025.
- Orgad et al. (2024). "LLMs Know More Than They Show." ICLR 2025.
- Angelopoulos et al. (2024). "Conformal Risk Control." ICLR 2024.

## License

Research use only.
