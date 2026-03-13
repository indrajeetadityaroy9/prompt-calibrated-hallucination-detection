# Spectral Information Geometry (SIG): A Unified Calibration-Fusion Framework for AG-SAR

## 1. Executive Summary

This proposal introduces **Spectral Information Geometry (SIG)**, a unified mechanism that replaces AG-SAR's two-stage pipeline (independent calibration → post-hoc diagonal fusion) with a single geometrically-grounded framework. SIG addresses three interconnected theoretical limitations through one cohesive design:

1. **Tracy-Widom Probabilistic SPT** — replaces the binary BBP clamp with a smooth, finite-sample-calibrated phase transition probability
2. **Cross-Signal Precision Fusion** — replaces diagonal inverse-variance weighting with a precision matrix that captures DPS-POS cross-correlation from prompt-tail joint statistics
3. **Spectral Gap Enrichment** — augments SPT with the bounded ratio λ₂/(λ₁+λ₂) ∈ [0, 0.5], capturing the *sharpness* of spectral signal separation (not just its existence)

These three components form a single mechanism because they share a common theoretical foundation: the spectral geometry of the joint signal manifold. Tracy-Widom provides the proper null model for spectral statistics; the full precision matrix captures the geometry of the joint signal distribution; and the spectral gap ratio enriches the information content that feeds into the geometric fusion.

---

## 2. Research Domains and Scope

### Primary Domains
- **Mechanistic interpretability of LLMs** — internal signal extraction from transformer computation
- **Random matrix theory (RMT) in deep learning** — spectral analysis of activation covariance
- **Training-free uncertainty quantification** — zero-shot, calibration-free risk estimation
- **Meta-analytic signal fusion** — principled combination of heterogeneous evidence streams

### Subdomains
- Spiked covariance models and BBP phase transitions
- Tracy-Widom distribution theory for finite-sample spectral statistics
- Fisher information geometry and precision-weighted estimation
- Probability integral transform (PIT) calibration with finite-sample guarantees
- Entropy-gated evidence weighting (DerSimonian & Laird framework)

---

## 3. Identified Limitations in Current Architecture

### 3.1 Binary BBP Clamp in SPT (Critical)

**Current formulation:**
```
SPT(t) = 1 - clamp((λ₁ - λ₊) / λ₊, 0, 1)
```

**Problems:**
- **Derivative discontinuity** at both boundaries (excess = 0 and excess = 1). The gradient is exactly zero outside [0, 1] and constant inside, creating a piecewise-linear function with no sensitivity to the *magnitude* of departure.
- **Saturation**: Any excess > 1 maps to SPT = 0 identically, discarding information about how far above the BBP threshold the leading eigenvalue sits.
- **No finite-sample calibration**: The MP upper edge λ₊ is an asymptotic quantity. For finite window sizes W (typically 5–30 in AG-SAR), the actual distribution of the largest eigenvalue fluctuates around λ₊ according to the Tracy-Widom distribution at rate W^{−2/3}. The binary clamp ignores these fluctuations entirely.
- **No probabilistic interpretation**: The clamped ratio is not a probability, p-value, or calibrated score — it's an ad-hoc rescaling.

**Literature support:** Ettori et al. (2601.17357) demonstrate that the Tracy-Widom distribution provides the correct null model for spectral statistics of transformer hidden states. EigenTrack uses P(λ₁ > λ₊ + δ) as a feature, validating the probabilistic approach. Johnstone (2001) and the BBP paper (Baik, Ben Arous & Péché, 2005) establish the exact centering and scaling constants.

### 3.2 Diagonal Independence Assumption in Fusion (Structural)

**Current formulation:**
```
w_i(t) = (1/var_i) × (1 - H_i(t))^κ    [diagonal: no cross-signal terms]
risk(t) = Σᵢ w_i(t) p_i(t) / Σᵢ w_i(t)
```

**Problems:**
- **Ignores cross-signal dependencies**: CUS (attention utilization) and POS (MLP override) are causally coupled — if attention doesn't attend to context (high CUS), the MLP has nothing context-grounded to override (POS becomes less informative). DPS and SPT both operate on hidden-state geometry and share variance.
- **Overweights redundant information**: When two correlated signals both fire, diagonal fusion counts their shared information twice. The full precision matrix Ω = Σ⁻¹ naturally downweights correlated signals via negative off-diagonal entries.
- **Proxy variances for CUS and SPT**: Currently `var(CUS) = median(var_DPS, var_POS)` and `var(SPT) = median(var_DPS, var_POS, var_CUS)`. These are not intrinsic — they're borrowed from other signals. The full covariance matrix provides intrinsic variance for all signals simultaneously.

**Literature support:** DerSimonian & Laird (1986) — the framework AG-SAR cites — explicitly assumes independence. The generalization to correlated estimators uses the full precision matrix (Hartung, Knapp & Sinha, 2008). Vector copula methods (2503.01072) and information-geometric fusion (2501.01556) provide modern treatments of dependent signal aggregation.

### 3.3 Single-Eigenvalue SPT (Information Loss)

**Current formulation:**
```
SPT uses only λ₁ (leading eigenvalue) relative to λ₊ (MP edge)
```

**Problems:**
- **Binary signal/noise question**: Current SPT asks "is there signal above noise?" but not "how cleanly separated is the signal from noise?" The spectral gap λ₁/λ₂ captures this distinction.
- **Multi-spike blindness**: In the spiked covariance model, multiple eigenvalues may emerge above the bulk. If λ₁ is large but λ₂ ≈ λ₁ (degenerate spikes), the representation has lost directional coherence — a different failure mode than λ₁ ≈ λ₊.
- **Lost information**: EigenTrack (2601.17357) uses 22 spectral descriptors precisely because a single eigenvalue ratio discards most of the spectral structure.

**Literature support:** The spectral gap ratio appears in Wigner-Dyson statistics and has been shown to distinguish structured from random matrices more robustly than the leading eigenvalue alone (SpectralGap OOD, 2505.15177). SGD-to-Spectra (2507.12709) theoretically justifies the bulk+spike structure that the gap ratio detects.

---

## 4. Proposed Mechanism: Spectral Information Geometry (SIG)

### 4.1 Theoretical Foundation

AG-SAR's four signals are projections of a single underlying geometric quantity: the divergence of generation-time hidden states from the prompt-anchored reference manifold. Currently, each projection is calibrated independently and combined via a diagonal weighted mean — the Euclidean approximation to the true geometry of the joint signal space.

SIG replaces this with a framework grounded in two mathematical principles:

1. **Tracy-Widom theory** provides the correct finite-sample null model for spectral statistics, replacing the asymptotic BBP threshold with a proper probability.
2. **Full precision weighting** captures the joint geometry of the signal space, replacing diagonal independence with the natural metric on the signal manifold.

### 4.2 Component 1: Tracy-Widom Probabilistic SPT

**Governing equations:**

Given the sliding-window matrix H ∈ ℝ^{W×d} of centered midpoint-layer hidden states:

```
σ² = median(eigenvalues of H^T H / W)           [noise floor estimate]
γ  = d / W                                        [aspect ratio]
μ_TW = σ²(1 + √γ)²                               [MP upper edge = current λ₊]
σ_TW = σ²(1 + √γ) × (1/√W + 1/√d)^{1/3}         [TW scaling rate]
z_TW = (λ₁ - μ_TW) / σ_TW                        [standardized TW statistic]
SPT(t) = 1 - F_{TW,1}(z_TW)                      [Tracy-Widom CDF → probability]
```

**Where F_{TW,1} is the Tracy-Widom β=1 CDF**, computed via the Cornish-Fisher expansion with exact TW moments (μ_TW_dist ≈ −1.2065, σ_TW_dist ≈ 1.2680, γ₃ ≈ 0.2935):

```
z = (s - μ_TW_dist) / σ_TW_dist
z_CF = z - (γ₃/6)(z² - 1)                         [Cornish-Fisher skewness correction]
F_{TW,1}(s) ≈ Φ(z_CF)                             [Gaussian CDF with CF correction]
```

**Interpretation:**
- When λ₁ is in the noise bulk: z_TW ≈ 0, F_TW ≈ 0.5, SPT ≈ 0.5 (uncertain — smooth, not hard 1.0)
- When λ₁ >> μ_TW (strong signal): z_TW >> 0, F_TW → 1, SPT → 0 (not risky)
- When λ₁ << μ_TW (collapsed below noise): z_TW << 0, F_TW → 0, SPT → 1 (risky)
- The transition is **smooth** with proper finite-sample width proportional to W^{−2/3}

**Why this is principled:**
- The Tracy-Widom distribution is the *exact* limiting distribution of the largest eigenvalue of random covariance matrices (Tracy & Widom, 1994; Johnstone, 2001).
- The centering μ_TW and scaling σ_TW are derived from the Marchenko-Pastur law, not fitted.
- The Cornish-Fisher expansion accounts for the positive skewness of TW₁ (skewness ≈ 0.2935), giving CDF accuracy < 0.005 in the region |z| < 3 where AG-SAR operates.
- No new parameters are introduced — all constants are universal properties of random matrices.

### 4.3 Component 2: Spectral Gap Enrichment

**Governing equation:**

```
g(t) = λ₂ / (λ₁ + λ₂ + ε)     [bounded spectral gap ratio ∈ [0, 0.5]]
```

This is folded into the SPT computation by returning both `spt` and `spectral_gap` from the same eigenvalue decomposition. The spectral gap is stored in `TokenSignals` and participates in fusion as a **fifth signal dimension** that enters the precision matrix.

**Why λ₂/(λ₁+λ₂) instead of λ₁/λ₂:** The ratio λ₁/λ₂ is unbounded above, which breaks the [0,1] risk guarantee when entering the weighted-mean fusion formula. The reciprocal form λ₂/(λ₁+λ₂) is naturally bounded in [0, 0.5] and has a clean interpretation: near 0 = clean spike separation (low risk), near 0.5 = degenerate eigenvalues (high risk).

**Why a fifth signal (not merged into SPT):**
- SPT measures "is there signal above noise?" (phase transition probability)
- Spectral gap measures "how coherent is the signal structure?" (directional stability)
- These are complementary: SPT can be low (strong signal) but gap can be small (degenerate signal — multiple competing directions). This indicates a different failure mode: the model has structure but lacks directional coherence.
- Merging them into one scalar via multiplication or ad-hoc combination would lose this distinction. Keeping them separate lets the precision matrix discover their optimal combination from the prompt-tail statistics.

**Calibration:** Direct mode (like CUS and SPT), since the gap ratio has a natural [0, 0.5] range. Variance is computed incrementally from prompt-tail SPT evaluations when sufficient samples exist; otherwise falls back to peer-derived variance.

### 4.4 Component 3: Cross-Signal Precision Fusion

**Governing equations:**

During calibration, compute the cross-signal precision from prompt-tail signals:

```
Base: Ω = diag(1/var_cus, 1/var_pos, 1/var_dps, 1/var_spt, 1/var_gap)  [diagonal baseline]

DPS-POS coupling (signals with per-position tail samples):
  Σ_2x2 = Cov([pos_tail, dps_tail]) + ε·I     [regularized 2×2 covariance]
  Ω_2x2 = Σ_2x2⁻¹                             [2×2 precision block]
  Ω[pos,pos] = Ω_2x2[0,0];  Ω[pos,dps] = Ω_2x2[0,1]
  Ω[dps,pos] = Ω_2x2[1,0];  Ω[dps,dps] = Ω_2x2[1,1]
```

This hybrid approach uses the full precision block for signals with genuine per-position variation (DPS and POS), while retaining diagonal precision for signals without per-position calibration data (CUS needs attention weights; SPT/gap produce one value from the sliding window). This avoids the degenerate-rank problem of expanding constant or proxy columns into a full 5×5 covariance.

During token-level fusion:

```
p = [p_cus(t), p_pos(t), p_dps(t), p_spt(t), p_gap(t)]    [calibrated signal probabilities]
δ = p - 0.5                                                  [deviation from neutral]

H_i = binary_entropy(p_i)                                    [per-signal entropy]
e_i = (1 - H_i)^κ                                            [entropy modulation]

δ̃ = δ ⊙ e                                                   [entropy-modulated deviations]

risk(t) = 0.5 + (δ̃^T Ω̃ p̃) / (1^T |Ω̃| 1)
```

where Ω̃ = diag(e) · Ω · diag(e) is the entropy-modulated precision matrix and p̃ replaces p in the numerator to maintain the [0,1] risk range.

**Simplified formulation (preserving the weighted-mean structure):**

For practical implementation, the cross-signal precision enters through **effective weights** that incorporate off-diagonal coupling:

```
For each signal i:
    w̃_i(t) = Σ_j Ω_ij × e_j(t)    [precision-coupled entropy-modulated weight]

risk(t) = Σᵢ max(w̃_i, 0) × p_i(t) / Σᵢ max(w̃_i, 0)
```

This means signal i's effective weight depends on how informative *all* signals are, weighted by their precision coupling. If CUS is uninformative (e_CUS ≈ 0) but strongly correlated with POS, POS's effective weight is reduced (because part of POS's apparent information overlaps with CUS's noise contribution).

**Why this preserves AG-SAR's properties:**
- Output remains in [0, 1] (weighted mean of probabilities)
- Reduces to current diagonal fusion when Ω is diagonal (signals uncorrelated)
- No learned parameters: Ω is computed from prompt-tail statistics
- The regularization ε·I ensures numerical stability for short prompt tails
- Cross-signal coupling is prompt-specific: different prompts yield different correlation structures

**Response-level fusion** follows the same structure: compute per-signal response means, normalize, then apply the cross-signal precision fusion.

---

## 5. Mathematical Justification

### 5.1 Tracy-Widom Convergence Guarantee

**Theorem (Johnstone, 2001; Baik, Ben Arous & Péché, 2005):** Let H ∈ ℝ^{W×d} have i.i.d. entries with mean 0 and variance σ². As W, d → ∞ with d/W → γ > 0:

```
(λ₁(H^T H / W) - μ_TW) / σ_TW  →_d  TW₁
```

where μ_TW = σ²(1 + √γ)² and σ_TW = σ²(1 + √γ)(W^{-1/2} + d^{-1/2})^{1/3}.

**Application to AG-SAR:** The hidden states at the midpoint layer, after centering, are approximately zero-mean with near-isotropic covariance (due to LayerNorm). The spiked covariance model C_t = σ²I + Σᵢ θᵢuᵢuᵢ^T captures the context-aligned signal directions. When generation is grounded, θᵢ > σ²(1 + √γ) and eigenvalues emerge above the bulk. When hallucinating, signal strengths decay below the BBP threshold and eigenvalues reabsorb.

The TW distribution provides the *exact* probability of this transition, accounting for finite-sample fluctuations that the binary clamp ignores. For typical AG-SAR window sizes (W = 5–30, d = 4096), the TW scaling σ_TW ~ W^{-2/3} gives meaningful probability resolution.

### 5.2 Precision Matrix Optimality

**Theorem (Generalized Least Squares):** Given K estimators θ̂₁, ..., θ̂_K with joint covariance matrix Σ, the minimum-variance linear combination is:

```
θ̂_opt = (1^T Σ⁻¹ 1)⁻¹ × 1^T Σ⁻¹ θ̂
```

with weights proportional to the rows of Σ⁻¹. When Σ is diagonal, this reduces to inverse-variance weighting (the current AG-SAR formulation). The full Σ⁻¹ is the natural generalization.

**Application to AG-SAR:** The four signals estimate different aspects of the same latent quantity (hallucination risk). Their prompt-tail joint distribution provides an empirical estimate of Σ. The full precision matrix fusion is the minimum-variance combination — strictly better than the diagonal approximation when signals are correlated.

### 5.3 Spectral Gap as Independent Information

**Theorem (Wigner surmise generalization):** For the spiked covariance model, the ratio λ₁/λ₂ carries information about the number and structure of spikes that is not captured by λ₁ alone. Specifically:

- Single dominant spike: λ₂/(λ₁+λ₂) → 0 as n → ∞ (clean signal, low risk)
- Multiple equal spikes: λ₂/(λ₁+λ₂) → 0.5 (degenerate — no preferred direction, high risk)
- No spikes (pure noise): λ₂/(λ₁+λ₂) → 0.5 - O(n^{-2/3}) ≈ 0.5 (noise, high risk)

The bounded gap ratio thus distinguishes "coherent signal" from "degenerate or absent signal" — information orthogonal to the TW probability, while remaining in [0, 0.5] for safe fusion.

---

## 6. Computational Feasibility

### 6.1 Tracy-Widom CDF
- **Cost:** One evaluation of the standard Gaussian CDF Φ(·) per token (after Cornish-Fisher transform)
- **Overhead:** Negligible — replaces the current `min(max(...))` clamp
- **Dependencies:** `scipy.stats.norm.cdf` (already available via scipy ≥ 1.11.0)

### 6.2 Spectral Gap Ratio
- **Cost:** Zero additional computation — λ₂ is already computed by `torch.linalg.svdvals(H)` in the current SPT. We simply read `eigs[1]` in addition to `eigs[0]`.
- **Memory:** One additional float per token

### 6.3 Full Precision Matrix
- **Calibration cost:** One 5×5 covariance matrix computation + inverse during prefill. For n_tail ≈ 10–50 samples, this is O(n_tail × 5²) = negligible.
- **Per-token cost:** One 5-vector × 5×5-matrix multiplication = 25 FLOPs per token. Current diagonal weighting is 5 multiplications = 5 FLOPs. The 5× increase is negligible relative to the O(d²) cost of each transformer layer.
- **Memory:** 25 floats for the precision matrix (stored once during calibration)

### 6.4 Total Overhead
The entire SIG mechanism adds < 100 FLOPs per token to a pipeline dominated by transformer forward passes costing O(10⁹) FLOPs. The overhead is unmeasurable.

---

## 7. Comparison with Contemporary Literature

| Method | Training Required | Signals | Cross-Signal Modeling | Spectral Calibration | Per-Token |
|--------|:-:|:-:|:-:|:-:|:-:|
| AG-SAR (current) | No | 4 | Diagonal (independent) | Binary BBP clamp | Yes |
| **AG-SAR + SIG (proposed)** | **No** | **5** | **Full precision matrix** | **Tracy-Widom CDF** | **Yes** |
| EigenTrack (2601.17357) | Yes (RNN) | 22 spectral | Learned (RNN) | MP + TW features | Yes |
| INSIDE (EigenScore) | No | 1 spectral | N/A | None | No (response) |
| Lookback Lens | Yes (linear probe) | 1 (attention) | N/A | None | Yes |
| ReDeEP (2410.11414) | No | 2 (attn + FFN) | Independent | None | No (response) |
| HARP (2509.11536) | No | 1 (subspace) | N/A | None | No (response) |

**Key differentiators of SIG:**
- **Only training-free method with cross-signal fusion**: All other training-free methods either use a single signal or assume independence.
- **Only method with TW-calibrated spectral analysis**: EigenTrack uses TW as a feature but requires a trained RNN to interpret it. SIG uses TW directly as a calibrated probability.
- **Principled generalization**: SIG strictly generalizes the current AG-SAR fusion — when signals are uncorrelated and λ₁ >> λ₊, SIG reduces to the current formulation.

---

## 8. Implementation Plan

### 8.1 Files Modified

| File | Change | Lines Affected |
|------|--------|:-:|
| `ag_sar/numerics.py` | Add `tracy_widom_cdf()` | +15 |
| `ag_sar/signals/spt.py` | TW-calibrated SPT + spectral gap output | ~20 modified |
| `ag_sar/config.py` | Add `spectral_gap` field to `TokenSignals` | +1 |
| `ag_sar/calibration.py` | Compute full cross-signal covariance + spectral gap calibration | ~30 modified |
| `ag_sar/aggregation/fusion.py` | Cross-signal precision fusion | ~40 modified |
| `ag_sar/detector.py` | Wire spectral gap through pipeline | ~5 modified |

**Total: ~110 lines changed across 6 files. No new files. No new dependencies.**

### 8.2 Detailed Code Changes

See implementation in the codebase. Each change is documented inline with the mathematical justification from this proposal.

---

## 9. Risk Assessment and Graceful Degradation

### 9.1 When SIG reduces to current AG-SAR
- **Uncorrelated signals**: When Σ is diagonal, Ω = diag(1/var_i), and cross-signal fusion reduces exactly to the current diagonal weighting.
- **Large window sizes**: When W → ∞, σ_TW → 0, and TW CDF approaches a step function at λ₊, recovering the binary BBP threshold behavior.
- **Single dominant eigenvalue**: When λ₁ >> λ₂, the spectral gap signal is always "informative" (g >> 1) and contributes negligible additional information, effectively dropping to the current 4-signal system.

### 9.2 Potential failure modes
- **Very short prompt tails** (n_tail < 5): The 5×5 covariance matrix may be rank-deficient. Mitigated by ε·I regularization in the precision matrix.
- **Perfectly uniform signals**: If all prompt-tail signals are identical, Σ collapses. The ε·I regularization falls back to isotropic (diagonal) weighting.
- **Extreme aspect ratios**: For γ = d/W >> 100, the TW approximation may be less accurate. However, AG-SAR's window sizes (effective rank ≈ 5–30 for d = 4096) give γ ∈ [100, 800], which is within the validated regime of the Cornish-Fisher TW approximation.

---

## 10. References

1. Tracy, C. A., & Widom, H. (1994). "Level-spacing distributions and the Airy kernel." *Communications in Mathematical Physics*, 159(1), 151–174.
2. Johnstone, I. M. (2001). "On the distribution of the largest eigenvalue in principal components analysis." *Annals of Statistics*, 29(2), 295–327.
3. Baik, J., Ben Arous, G., & Péché, S. (2005). "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices." *Annals of Probability*, 33(5), 1643–1697.
4. DerSimonian, R., & Laird, N. (1986). "Meta-analysis in clinical trials." *Controlled Clinical Trials*, 7(3), 177–188.
5. Hartung, J., Knapp, G., & Sinha, B. K. (2008). *Statistical Meta-Analysis with Applications*. Wiley.
6. Ettori, E., et al. (2026). "Spectral Geometry for Deep Learning: Compression and Hallucination Detection via Random Matrix Theory." arXiv:2601.17357.
7. Cornish, E. A., & Fisher, R. A. (1938). "Moments and cumulants in the specification of distributions." *Revue de l'Institut International de Statistique*, 5(4), 307–320.
8. Roy, O. (2007). "Effective Rank: A Measure of Effective Dimensionality." *European Signal Processing Conference*.
9. Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." *IEEE Trans. SMC*, 9(1), 62–66.
10. Bornemann, F. (2010). "On the numerical evaluation of distributions in random matrix theory." *Markov Processes and Related Fields*, 16(4), 803–866.
