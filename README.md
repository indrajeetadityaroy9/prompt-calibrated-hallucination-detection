# Prompt-Calibrated Spectral Signals for Zero-Shot Hallucination Detection

Zero-shot, training-free hallucination detector for language models. Hooks into transformer internals during autoregressive generation, extracts five spectral signals from cross-layer hidden states, and fuses them with a CUSUM change-point detector calibrated from the prompt itself. No external data or learned parameters.

## Core Idea

Hidden states across $L$ layers are stacked into $H \in \mathbb{R}^{L \times d}$. The dual covariance $C = \frac{1}{d} H_c H_c^\top \in \mathbb{R}^{L \times L}$ is eigendecomposed cheaply ($L \ll d$). Eigenvalues above the Marchenko-Pastur edge $\lambda_+ = \sigma^2(1+\sqrt{L/d})^2$ are signal; the rest is noise. Hallucination shifts this spectral structure relative to the prompt's baseline.

## Signals

| Signal | What it measures |
|--------|-----------------|
| **rho** | Spike excess above MP noise floor: $(\lambda_1 - \lambda_+) / \lambda_+$ |
| **phi** | Layer-wise information flow smoothness: $\|f\|_1 / (\|f\|_1 + \text{TV}(f))$ |
| **spf** | Variance fraction along prompt eigenvectors |
| **mlp** | Mean JSD between pre/post-MLP logit distributions (logit lens) |
| **ent** | Attention entropy bimodality via Otsu coefficient |

## Fusion

The five signals form a time series over generated tokens. A CUSUM detector accumulates Mahalanobis distance from the prompt-calibrated null:

$$d(t) = (s(t) - \mu)^\top \Omega\, (s(t) - \mu), \quad C(t) = \max(0, C(t{-}1) + d(t) - \tau), \quad \text{risk}(t) = C(t) / (C(t) + h)$$

where $\mu$, $\Omega$ (Ledoit-Wolf precision), $\tau$, and $h$ are all estimated from prompt tokens.
