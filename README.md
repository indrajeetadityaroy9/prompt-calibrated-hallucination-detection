# AG-SAR: Aggregated Signal Architecture for Risk

**Training-free, zero-shot hallucination detection for large language models via spectral information flow analysis and sequential change-point detection.**

Large language models hallucinate.Existing detection methods require labeled training data, multiple sampling passes, or external NLI models. AG-SAR operates within a single forward pass without prior supervision by intercepting transformer internals during autoregressive generation to extract five spectral signals and fusing them through a CUSUM (Cumulative Sum) change-point detector calibrated entirely from the input prompt.

The core theoretical insight is that hallucination manifests as a **phase transition in the spectral structure of layer-wise hidden representations** — detectable online via cumulative divergence from a prompt-calibrated reference distribution. AG-SAR models each token's cross-layer hidden states as a spiked random matrix, using the Marchenko-Pastur law to separate semantic signal eigenvalues from noise, and applies sequential hypothesis testing to detect when the spectral structure departs from the prompt-established baseline.

## Method

### Spectral Signal Extraction

For each generated token, hidden states are stacked across all L transformer layers to form a matrix H ∈ R^{L×d}. The **dual covariance** C = (1/d) H H^T ∈ R^{L×L} is analyzed spectrally. This formulation operates correctly in the d >> L regime typical of modern LLMs (e.g., L=32, d=4096) where standard covariance methods would be ill-conditioned.

Five signals, each grounded in a distinct theoretical framework:

| Signal | Definition | Theoretical basis |
|--------|-----------|-------------------|
| **Spike Excess** (rho) | Largest eigenvalue exceedance above the Marchenko-Pastur edge, normalized | Spiked covariance model; BBP phase transition (Baik, Ben Arous & Peche, 2005) |
| **Info Flow Regularity** (phi) | Smoothness of the layer-wise Fisher information profile: L1 / (L1 + TV) | Fisher information flow in neural networks (Weimar et al., PRX 2025) |
| **Spectral Projection Fidelity** (spf) | Fraction of token's dual covariance variance captured by prompt eigenvectors | Spectral subspace alignment; EigenShield (Ettori, 2025) |
| **MLP Divergence** (mlp) | All-layer mean Jensen-Shannon divergence between pre-MLP and post-MLP logit distributions | Information-theoretic transformation analysis |
| **Attention Entropy** (ent) | Bimodality of per-head normalized attention entropies via Otsu coefficient | Attention head specialization analysis |

### CUSUM Sequential Fusion

Rather than fusing signals independently per token, AG-SAR treats the 5-dimensional signal vector as a multivariate time series and applies a **CUSUM change-point detector** to identify hallucination onset.

**Calibration** (prompt tail, all parameters data-derived):
```
mu    = mean of signal vectors over calibration window
Omega = inverse of Ledoit-Wolf shrinkage covariance (5x5 precision matrix)
tau   = mean Mahalanobis distance under the null (drift allowance)
h     = maximum CUSUM excursion on calibration data (detection threshold)
```

**Per-token detection**:
```
d(t) = (s(t) - mu)^T Omega (s(t) - mu)       Mahalanobis distance from null
C(t) = max(0, C(t-1) + d(t) - tau)            CUSUM accumulator
risk(t) = C(t) / (C(t) + h)                   Risk score in [0, 1)
```

The CUSUM statistic accumulates evidence of distributional shift over tokens. Hallucination spans are contiguous regions where C(t) exceeds the threshold h. This provides minimax-optimal detection delay (Lorden, 1971) and replaces ad-hoc per-token thresholding with a principled sequential test.

### Prompt-Anchored Self-Calibration

All normalization uses only the input prompt — no external data or learned parameters:

- **Calibration window**: min(num_layers, prompt_length) tokens from the prompt tail. This is the natural scale for the L×L dual covariance analysis.
- **Spectral reference**: Prompt-tail eigenvectors above the BBP phase transition threshold define the "expected" spectral structure.
- **Signal covariance**: Full 5×5 Ledoit-Wolf shrinkage estimate from prompt-tail signal vectors, inverted to obtain the precision matrix for Mahalanobis scoring.
- **CUSUM parameters**: tau (drift allowance) and h (detection threshold) derived from running the CUSUM on the calibration window itself.

## Pipeline

```
Prefill (once per input)
  Register hooks on all transformer layers
  Forward pass with output_attentions=True
  Capture tail hidden states across all layers
  Calibrate spectral analyzer (prompt dual covariance eigenvectors above MP edge)
  Compute all 5 signals over prompt tail
  Fit CUSUM null distribution (mu, precision, tau, h)

Generation (per token)
  Capture layer-wise hidden states via hooks
  Compute dual covariance eigendecomposition → rho, spf
  Compute layer-wise Fisher information profile → phi
  Compute pre/post-MLP JSD across all layers → mlp
  Compute attention head entropy bimodality → ent
  Accumulate Mahalanobis distance into CUSUM statistic

Post-processing
  Map CUSUM values to risk scores
  Extract hallucination spans from CUSUM threshold crossings
  Flag response if max CUSUM exceeds calibrated threshold
```
