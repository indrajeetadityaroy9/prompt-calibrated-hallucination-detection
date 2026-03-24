# AG-SAR: Aggregated Signal Architecture for Risk

**Domain**: LLM reliability and trustworthiness at inference time.

**Problem**: Large language models hallucinate — fabricating facts, misattributing sources, or generating content ungrounded in the input. Existing detection methods require labeled training data, multiple sampling passes, or external NLI models. None operate within a single forward pass without prior supervision.

**Approach**: AG-SAR intercepts transformer internals during autoregressive generation to extract five mechanistically-grounded signals targeting distinct failure modes along the processing chain. These signals are fused via cross-signal precision-weighted entropy-gated aggregation into per-token and response-level risk scores.

### Causal Signal Decomposition

Five signals, each probing a different point in the transformer's computation:

| Signal | Question it answers | Mechanism |
|--------|-------------------|-----------|
| **ENT** (Attention Entropy Dispersion) | Are attention heads specialized or diffuse? | Bimodality of per-head normalized entropies via Otsu coefficient |
| **MLP** (Transformation Magnitude) | How much is the MLP revising what attention found? | All-layer mean JSD between pre-MLP and post-MLP logit distributions |
| **PSP** (Prompt Subspace Projection) | Is the generation grounded in the prompt? | Projection onto prompt subspace (SVD + effective rank), magnitude-gated |
| **SPT** (Spectral Phase-Transition) | Has the representation manifold collapsed toward noise? | Tracy-Widom calibrated eigenvalue test on sliding-window covariance spectrum |
| **Spectral Gap** | How coherent is the signal structure? | Ratio of top two eigenvalues from the same decomposition as SPT |

### 2. Prompt-Anchored Self-Calibration

Each signal is normalized using only the prompt itself — no external data:

- **PSP, MLP**: Ranked against prompt-tail values to produce percentile scores.
- **ENT, SPT, spectral gap**: Used directly (already on meaningful scales).
- **Calibration window**: sqrt(prompt_length) tokens from the prompt tail.
- **Signal covariance**: A 5x5 precision matrix (inverse covariance) estimated from prompt-tail signal values captures how signals co-vary for a given input.

### 3. Training-Free Fusion

- **Weighted combination**: Each signal's weight comes from the precision matrix, so correlated signals share influence rather than double-counting.
- **Entropy gating**: Signals near p = 0.5 (uncertain) get downweighted to zero; decisive signals dominate.
- **Span detection**: Otsu thresholding on token risks, then merging nearby flagged tokens into contiguous spans.

## Pipeline

```
Prefill (once per input)
  Capture prompt hidden states at the middle layer
  Build prompt subspace from SVD of prompt representations
  Collect baseline signal values from the prompt tail for normalization
  Estimate signal covariance to set fusion weights

Generation (per token)
  Compute all 5 signals from hidden states and attention weights
  Normalize each signal against prompt baselines
  Combine into a single token risk score using weighted fusion
  Aggregate token risks into a response-level risk score

Post-processing
  Threshold token risks and merge nearby flagged tokens into spans
```
