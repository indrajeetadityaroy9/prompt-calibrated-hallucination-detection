# Prompt-Calibrated Spectral Signals for Zero-Shot Hallucination Detection in Language Models

LLM's hallucinate -- generating fluent text that is unfaithful to the provided context. Existing detection methods require either supervised training on labeled hallucination data or access to external knowledge bases, limiting their applicability. This work presents a zero-shot, training-free hallucination detector that operates entirely within the transformer's own representations. The method hooks into the autoregressive generation process, extracts five mechanistic signals from cross-layer hidden states at each token, and fuses them through a CUSUM sequential change-point detector calibrated from the prompt itself. The key insight is that hallucination manifests as a phase transition in the spectral structure of layer-wise representations: when generation departs from the prompt's semantic structure, the dominant eigenvalues of the cross-layer covariance matrix shift relative to the Marchenko-Pastur noise floor. Modeling each token's hidden states as a spiked random matrix and running sequential hypothesis testing against a prompt-derived null distribution detects this transition without any learned parameters or external data.

## Signals

For each generated token, hidden states across $L$ layers are stacked into $H \in \mathbb{R}^{L \times d}$ and the dual covariance $C = \frac{1}{d} H_c H_c^\top \in \mathbb{R}^{L \times L}$ is computed, exploiting the $d \gg L$ regime to make eigendecomposition tractable.

| Signal | Description | Computation |
|--------|-------------|-------------|
| **rho** | Spike excess above noise floor | $(\lambda_1 - \lambda_+) / \lambda_+$ where $\lambda_+$ is the Marchenko-Pastur edge |
| **phi** | Layer-wise information flow regularity | $\|f\|_1 / (\|f\|_1 + \text{TV}(f))$ on the Fisher information profile |
| **spf** | Alignment with prompt spectral structure | Variance fraction along prompt eigenvectors above the BBP threshold |
| **mlp** | MLP revision magnitude | Mean Jensen-Shannon divergence between pre- and post-MLP logit distributions |
| **ent** | Attention head specialization | $1 - \eta_{\text{Otsu}}$ on per-head normalized entropies |

## Fusion

The five signals form a multivariate time series over generated tokens. Rather than scoring tokens independently, evidence of distributional shift is accumulated via a CUSUM detector operating on Mahalanobis distances from the prompt-calibrated null:

$$d(t) = (s(t) - \mu)^\top \Omega\, (s(t) - \mu)$$

$$C(t) = \max(0,\; C(t-1) + d(t) - \tau)$$

$$\text{risk}(t) = C(t)\, /\, (C(t) + h)$$

where $\mu$ and $\Omega$ (Ledoit-Wolf shrinkage precision) are estimated from prompt tail tokens, $\tau$ is the mean calibration distance, and $h$ is the maximum calibration excursion.

## Calibration

All parameters are prompt-anchored — no external data, no training:
- **Window**: $\min(L,\, \text{prompt length})$ tail tokens
- **Reference spectrum**: Prompt eigenvectors above the BBP phase transition threshold
- **Precision matrix**: $5 \times 5$ Ledoit-Wolf shrinkage inverse covariance
- **CUSUM thresholds**: $\tau$ from mean calibration distance, $h$ from maximum calibration CUSUM excursion

## Pipeline

```
Prefill
  Hook all layers, forward pass, capture tail hidden states
  Calibrate spectral analyzer from prompt covariance eigenvectors
  Fit CUSUM null distribution (mu, precision, tau, h)

Generation (per token)
  Dual covariance eigendecomposition  ->  rho, spf
  Fisher information profile          ->  phi
  Pre/post-MLP JSD via logit lens     ->  mlp
  Attention entropy bimodality        ->  ent
  Accumulate Mahalanobis distance into CUSUM

Post-processing
  Map CUSUM to risk scores, extract risky spans, flag response
```

## Evaluation

Evaluation is conducted on TriviaQA and SQuAD v2 using adaptive Otsu thresholding on token-level F1 to define hallucination labels, reporting AUROC, AUPRC, FPR@95%TPR, AURC, E-AURC, ECE, and Brier score with BCa bootstrap confidence intervals. Leave-one-out signal ablation quantifies each signal's marginal contribution by replacing it with its calibration mean and recomputing CUSUM risks.
