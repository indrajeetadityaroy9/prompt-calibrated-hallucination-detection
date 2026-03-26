# AG-SAR: Prompt-Anchored Spectral Signals for Zero-Shot Hallucination Detection in Language Models

AG-SAR hooks into transformer internals during autoregressive generation, extracts five spectral signals from layer-wise hidden states, and fuses them with a CUSUM change-point detector calibrated from the prompt itself.

Hallucination is a phase transition in the spectral structure of cross-layer representations. We model each token's hidden states as a spiked random matrix, use the Marchenko-Pastur law to separate signal from noise eigenvalues, and run sequential hypothesis testing to detect when generation departs from the prompt baseline.

### Signals

For each token, stack hidden states across L layers into H in R^{L x d}. Compute the dual covariance C = (1/d) H H^T in R^{L x L}. This handles the d >> L regime (e.g., L=32, d=4096) where standard covariance is ill-conditioned.

| Signal | What it measures | How |
|--------|-----------------|-----|
| **rho** (spike excess) | Semantic signal strength above noise floor | (lambda_1 - MP edge) / MP edge; BBP phase transition |
| **phi** (info flow regularity) | Smoothness of layer-wise updates | L1 / (L1 + TV) on Fisher information profile |
| **spf** (spectral projection fidelity) | Alignment with prompt's spectral structure | Variance fraction along prompt eigenvectors |
| **mlp** (MLP divergence) | How much MLP revises attention output | Mean JSD between pre/post-MLP logits across layers |
| **ent** (attention entropy) | Head specialization vs diffusion | Otsu coefficient on per-head normalized entropies |

### Fusion
The 5 signals form a multivariate time series. Instead of scoring tokens independently, CUSUM accumulates evidence of distributional shift:

```
d(t) = (s(t) - mu)^T Omega (s(t) - mu)    
C(t) = max(0, C(t-1) + d(t) - tau)         
risk(t) = C(t) / (C(t) + h)         
```

### Calibration

Everything is prompt-anchored. No external data, no learned parameters:
- **Window**: min(num_layers, prompt_length) tail tokens
- **Reference spectrum**: Prompt eigenvectors above BBP threshold
- **Precision matrix**: 5x5 Ledoit-Wolf shrinkage inverse covariance
- **CUSUM thresholds**: tau = mean calibration distance, h = max calibration excursion

## Pipeline

```
Prefill
  Hook all layers, forward pass, capture tail hidden states
  Calibrate spectral analyzer + fit CUSUM null distribution

Generation (per token)
  Dual covariance eigendecomposition -> rho, spf
  Fisher information profile -> phi
  Pre/post-MLP JSD -> mlp
  Attention entropy bimodality -> ent
  Accumulate Mahalanobis distance into CUSUM

Post-processing
  Map CUSUM to risk scores, extract spans, flag response
```
