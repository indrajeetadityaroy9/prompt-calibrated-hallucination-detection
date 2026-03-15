# AG-SAR Research Audit: Code-Level Analysis & Literature Alignment

## 1. Core Research Objectives & Claimed Contributions

AG-SAR (Aggregated Signal Architecture for Risk) claims to be a **zero-shot, training-free hallucination detection system** for autoregressive LLMs. Its stated contributions:

1. **Five mechanistically-grounded signals** extracted from transformer internals during generation (ENT, MLP, PSP, SPT, Spectral Gap)
2. **Cross-signal precision-weighted entropy-gated aggregation** — a novel fusion mechanism generalizing DerSimonian & Laird (1986) to correlated estimators via Hartung, Knapp & Sinha (2008)
3. **Zero-parameter design** — all thresholds derived from data (Otsu, effective rank, PIT, Tracy-Widom CDF)
4. **Prompt-anchored calibration** — PIT normalization against prefill-tail empirical CDF, no external data
5. **Context-free operation** — no retrieved context required, works from prompt conditioning and generation dynamics alone
6. **Architecture portability** — auto-detection across 5 model families

---

## 2. Research Domains & Subdomains

| Domain | Subdomain | AG-SAR Component |
|---|---|---|
| NLP / LLM Safety | Hallucination detection | Core objective |
| Uncertainty Quantification | Training-free uncertainty estimation | PIT, entropy gating |
| Random Matrix Theory | Marchenko-Pastur law, Tracy-Widom distribution, BBP phase transition | SPT signal |
| Representation Geometry | Subspace projection, hidden state analysis | PSP signal |
| Information Theory | Attention entropy, JSD, binary entropy gating | ENT, MLP signals |
| Statistical Meta-Analysis | Precision-weighted fusion, inverse-covariance weighting | Aggregation mechanism |
| Covariance Estimation | Ledoit-Wolf shrinkage | Cross-signal precision matrix |

---

## 3. Deep Code-Level Analysis

### 3.1 ENT — Attention Entropy Dispersion (`signals/ent.py`)

**Algorithm:** `ENT = 1 - otsu_coefficient(per_head_normalized_entropies)`

**Implementation:**
```
a = attn_tensor.float().clamp(min=EPS)
H = -(a * a.log2()).sum(dim=-1) / log2(seq_len)   # normalized to [0,1]
ENT = 1 - otsu_coefficient(H.reshape(-1).numpy())  # clipped to [0,1]
```

**Correctness Assessment:**
- ✅ Shannon entropy normalized by log₂(seq_len) correctly bounds to [0,1]
- ✅ Otsu coefficient as bimodality measure is parameter-free and well-established (Otsu, 1979)
- ✅ Fully vectorized — no Python loops
- ⚠️ **Interpretation inversion is correct but subtle**: Low Otsu coefficient (unimodal entropy distribution) → less head specialization → ENT near 1 (risky). High coefficient (bimodal) → clear specialization → ENT near 0 (safe)

**Novelty Assessment:**
- The use of **Otsu coefficient on attention entropy distributions** as a bimodality signal is novel. Prior work (Lookback Lens, Chuang et al. 2024) uses attention ratios but not bimodality of entropy distributions across all heads
- LapEigvals (Binkowski et al., 2025) uses spectral features of attention maps (Laplacian eigenvalues) but treats attention as adjacency matrices — fundamentally different from ENT's information-theoretic approach
- **Unique contribution**: Using the Otsu coefficient — a thresholding algorithm from image processing — as a signal for head specialization is creative and parameter-free

**Potential Gap:**
- First generated token gets `attentions=None` and defaults ENT to 0.5. This is a known limitation in the code but could affect token-level detection accuracy for the first response token

---

### 3.2 MLP — JSD Transformation Magnitude (`signals/_jsd_base.py`)

**Algorithm:** Batched all-layer mean JSD(pre-MLP logits, post-MLP logits) on adaptive candidate set

**Implementation:**
```
k = max(2, effective_rank(softmax(logits)))     # adaptive candidate set
cand = topk(logits, k).indices ∪ {emitted_token}
# For each layer:
z_pre = W_cand @ final_norm(h_resid_attn)      # pre-MLP projection
z_post = W_cand @ final_norm(h_resid_mlp)      # post-MLP projection
p, q = softmax(z_pre), softmax(z_post)
m = 0.5*(p + q)
JSD = 0.5 * (xlogy(p, p/m) + xlogy(q, q/m)).sum(-1) / ln(2)
MLP = mean(JSD over all layers)
```

**Correctness Assessment:**
- ✅ JSD is bounded [0, 1] in bits — correct normalization by ln(2)
- ✅ `torch.xlogy` handles p*log(p/m) correctly with numerical stability
- ✅ Candidate set via effective_rank is parameter-free
- ✅ All-layer batched via stacked tensor operations
- ✅ Uses `final_norm` before `lm_head` projection — correctly reproduces logit computation

**Novelty Assessment:**
- Measuring **MLP transformation magnitude via JSD on projected logits** is novel. Prior work measures MLP contribution via activation norms or probing classifiers
- The **adaptive candidate set** via effective_rank of the softmax distribution is a clean parameter-free alternative to fixed top-k
- No prior work uses JSD between pre/post-MLP logit distributions across all layers as a hallucination signal

**Potential Gap:**
- Projects through `lm_head` at each layer, which assumes the lm_head is meaningful for intermediate representations. This is a common but debatable assumption (logit lens vs. tuned lens debate)

---

### 3.3 PSP — Prompt Subspace Projection (`signals/psp.py`)

**Algorithm:** Magnitude-gated projection onto prompt subspace basis

**Implementation:**
```
# Calibration:
center = mean(H_prompt)
_, S, Vh = svd(H_prompt - center)
V = Vh[:effective_rank(S)]        # prompt basis
tau = median(||h - center||)      # magnitude scale

# Per-token:
h_centered = h - center
s_prompt = 1 - ||V @ h_centered|| / ||h_centered||
gate = 1 - exp(-||h_centered||² / tau²)    # Rayleigh survival
PSP = 0.5 + (s_prompt - 0.5) * gate
return mean(PSP over all layers)
```

**Correctness Assessment:**
- ✅ SVD-based subspace extraction is standard and correct
- ✅ Effective rank determines dimensionality — parameter-free
- ✅ Rayleigh survival function for magnitude gating: `1 - exp(-x²/τ²)` is the CDF of the Rayleigh distribution. When ||h|| is small relative to τ, the gate suppresses PSP toward 0.5 (agnostic). Mathematically sound
- ✅ Projection norm ratio ||V@h||/||h|| correctly measures alignment with prompt subspace
- ⚠️ Gate formula: `1 - exp(-||h||²/τ²)` — this is the Rayleigh CDF (P(X ≤ x) for X ~ Rayleigh(τ/√2)). Calling it "Rayleigh survival" is slightly imprecise (it's actually the CDF, not the survival function), but the mathematical operation is correct for the intended gating behavior

**Novelty Assessment:**
- **HARP (Hu et al., 2025)** is the most directly comparable work — it also uses SVD-based subspace projection of hidden states. Key differences:
  - HARP decomposes the *unembedding layer* (lm_head) via SVD to find "reasoning" vs "semantic" subspaces
  - PSP decomposes the *prompt hidden states* via SVD to find the prompt-conditioned subspace
  - HARP is supervised (trains a probe on the projections); PSP is zero-shot
  - HARP projects onto the *model's* intrinsic subspace; PSP projects onto the *prompt's* subspace
- **HaloScope (Du et al., 2024)** also uses SVD on embeddings but for membership estimation, not per-token scoring
- PSP's **magnitude gating via Rayleigh CDF** is novel — it prevents spurious signals from low-magnitude hidden states

---

### 3.4 SPT — Spectral Phase Transition (`signals/spt.py`)

**Algorithm:** Tracy-Widom calibrated BBP phase-transition detection via sliding-window covariance spectrum

**Implementation:**
```
# Ring buffer of hidden states H (window_size × hidden_dim)
H = H - mean(H)
S = svdvals(H)
eigs = S² / W                         # eigenvalues of sample covariance

sigma² = median(eigs)                  # bulk variance estimate
gamma = d / W                          # aspect ratio
mu_TW = sigma² * (1 + √γ)²           # MP upper edge
sigma_TW = sigma² * (1 + √γ) * (1/√W + 1/√d)^{1/3}   # TW scaling

z_TW = (λ₁ - mu_TW) / sigma_TW       # standardized statistic
SPT = 1 - F_{TW,1}(z_TW)             # Tracy-Widom CDF
spectral_gap = λ₂ / (λ₁ + λ₂)       # directional coherence
```

**Correctness Assessment:**
- ✅ Marchenko-Pastur upper edge formula `σ²(1+√γ)²` is textbook (Marchenko & Pastur, 1967). Correctly applied to sample covariance eigenvalues (eigs = S²/W)
- ❌ **TW scaling rate — MISSING 1/√W FACTOR**: The implementation uses `σ²(1+√γ)(1/√W + 1/√d)^{1/3}`. Per Johnstone (2001) Theorem 1.1, for Wishart eigenvalues l₁ of W_p(I,n), the scaling is `(√n+√p)(1/√n+1/√p)^{1/3}`. AG-SAR uses sample covariance eigenvalues λ = l₁/W, so the correct scaling is `σ_{Wishart}/W`:
  - Johnstone scaling for Wishart: `σ²(√W+√d)(1/√W+1/√d)^{1/3}`
  - Divide by W for sample covariance: `σ²(√W+√d)(1/√W+1/√d)^{1/3}/W = σ²(1+√γ)/√W · (1/√W+1/√d)^{1/3}`
  - AG-SAR has: `σ²(1+√γ) · (1/√W+1/√d)^{1/3}` — **missing the 1/√W factor**
  - **Impact**: z_TW is systematically √W times too small. For W=25 (typical window), z is 5× too small → TW CDF is closer to 0 → SPT is pushed toward 1.0 (riskier). Ordinal ranking between tokens is preserved (monotonic distortion), and Otsu thresholding is adaptive, so the signal still functions as a relative detector — but absolute TW probabilities are miscalibrated
- ✅ Using `median(eigs)` as σ² estimate is robust to outlier eigenvalues — better than mean for spiked models
- ✅ Tracy-Widom CDF via Cornish-Fisher expansion with exact TW₁ moments matches Bornemann (2010) tabulated values
- ✅ Ring buffer with circular indexing avoids memory allocation per step

**Novelty Assessment:**
- **INSIDE (Chen et al., 2024)** uses eigenvalues of response covariance matrices (EigenScore), but:
  - EigenScore operates on *multiple sampled responses* (requires K generations); SPT operates on a *single generation's hidden state trajectory*
  - EigenScore uses log-determinant (sum of log eigenvalues); SPT uses Tracy-Widom calibrated leading eigenvalue
  - EigenScore is response-level; SPT is token-level
- **LapEigvals (Binkowski et al., 2025)** uses eigenvalues of attention map Laplacians; SPT uses eigenvalues of hidden state covariance — different matrices entirely
- Using **Tracy-Widom/BBP phase transition theory for hallucination detection** appears to be **genuinely novel** in the NLP literature. This is a well-established technique in random matrix theory (Baik, Ben Arous & Péché, 2005) and signal detection (Johnstone, 2001), but its application to token-level hallucination scoring via hidden state dynamics is unprecedented
- The **spectral gap ratio** λ₂/(λ₁+λ₂) as a directional coherence measure is a simple but effective complement. It captures whether the hidden state trajectory is dominated by a single direction (coherent, low risk) or diffuse (high risk)

**Potential Issues:**
- ⚠️ Window size = effective_rank(prompt SVD). For very long prompts with low rank, this could be very small (e.g., 3-4), making the covariance estimate highly unstable and the TW approximation less reliable
- ⚠️ The γ = d/W aspect ratio can be very large when W is small (e.g., d=4096, W=5 → γ=819). The TW distribution is asymptotically valid for large min(n,p); small windows push into finite-sample territory where the Cornish-Fisher approximation may be less accurate

---

### 3.5 Spectral Gap (`signals/spt.py`)

**Formula:** `λ₂ / (λ₁ + λ₂) ∈ [0, 0.5]`

**Correctness:** Mathematically trivial and correct. The ratio is bounded by construction.

**Novelty:** Simple but no prior work uses this specific ratio for hallucination detection. It provides complementary information to SPT — SPT measures whether λ₁ is anomalously large (signal detection), while spectral gap measures the relative gap between the top two eigenvalues (directional coherence).

---

### 3.6 Calibration System (`calibration.py`)

**Algorithm:** Self-calibration from prompt tail window

**Key Implementation Details:**
- **Tail window**: `sqrt(prompt_len)` — adaptive, parameter-free
- **PSP calibration**: Batched computation across `(n_tail × n_layers)` positions, per-position mean over layers → `(n_tail,)` array of sorted values for PIT
- **MLP calibration**: Forward tail hidden states through `lm_head`, compute per-position JSD → `(n_tail,)` sorted values for PIT
- **ENT/SPT/Spectral Gap**: Direct mode (no PIT), variance computed from prompt
- **Cross-signal precision**: Stack PSP and MLP tail values into `(n_tail, 2)`, apply Ledoit-Wolf shrinkage → 2×2 precision matrix, embed in 5×5 diagonal at indices [1:3, 1:3]

**Correctness Assessment:**
- ✅ Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) is the correct choice for small-sample covariance estimation — parameter-free optimal regularization
- ✅ PIT (Probability Integral Transform) with Haldane-Anscombe correction `(rank+0.5)/(n+1)` avoids boundary artifacts at 0 and 1
- ✅ Only PSP and MLP get cross-signal precision coupling via Ledoit-Wolf (they share prompt-tail calibration data)
- ⚠️ The 5×5 precision matrix structure: The code initializes ALL 5 diagonal entries with **inverse-variance weighting** (`1/variance`), NOT identity. The 2×2 Ledoit-Wolf precision block then replaces entries [1:3, 1:3] (MLP, PSP), adding cross-correlation. So all 5 signals get per-signal precision weighting; only PSP-MLP additionally capture cross-signal correlation
- ⚠️ **ENT variance heuristic**: ENT uses `mode="direct"` and has no tail-derived values. Its variance is set to `median([psp_variance, mlp_variance])` — a cross-signal heuristic, not computed from actual ENT observations. This means ENT's precision weight is inherited from the other signals
- ⚠️ For short prompts (e.g., 16 tokens), tail window = 4 positions. Estimating a 2×2 covariance from 4 samples is marginal even with Ledoit-Wolf shrinkage

**Design Strength:** The adaptive window `sqrt(prompt_len)` elegantly balances between having enough samples for stable estimation and staying local to the generation boundary.

---

### 3.7 Aggregation: Precision-Weighted Entropy-Gated Fusion (`aggregation/fusion.py`)

**Algorithm:**
```
E = (1 - H_binary(P))^κ          # decisiveness matrix (n_tokens, k)
W = max(E @ Ω, 0)                # precision-weighted fusion weights
risk(t) = Σ max(w_i,0) * p_i(t) / Σ max(w_i,0)
```

**Where:**
- `H_binary(p) = -(p log₂p + (1-p) log₂(1-p))` — binary entropy
- `κ = 1 + median(decisiveness of PSP, MLP tail samples)` — adaptive, in [1, 2]
- `Ω` — precision matrix (inverse covariance from Ledoit-Wolf)

**Correctness Assessment:**
- ✅ Binary entropy gating: when p=0.5 (maximum uncertainty), H=1, decisiveness=0, weight→0. Signals near 0.5 contribute nothing to fusion. Correct and principled
- ✅ Precision matrix coupling: `E @ Ω` produces weights that account for inter-signal correlation. If two signals are correlated, the precision matrix down-weights their joint contribution. This is the multivariate generalization of inverse-variance weighting
- ✅ `max(w, 0)` clipping prevents negative weights from the precision matrix from inverting signal polarity
- ✅ Response-level risk uses signal-first aggregation (mean per signal, then fuse) — consistent with meta-analysis methodology

**Connection to Meta-Analysis Literature:**
- **DerSimonian & Laird (1986)**: Standard random-effects meta-analysis uses `w_i = 1/σ_i²` (inverse variance weighting). AG-SAR generalizes this by using the full precision matrix Ω = Σ⁻¹, which accounts for *between-signal* correlations, not just per-signal variance
- **Hartung, Knapp & Sinha (2008)**: Their textbook covers multivariate meta-analysis with correlated estimators. AG-SAR's claim of "generalizing DerSimonian & Laird to correlated estimators" is accurate — the precision matrix from Ledoit-Wolf shrinkage plays the role of the inverse covariance in multivariate inverse-variance weighting
- The **entropy gating** is AG-SAR's addition — it dynamically zeros out signals that are near maximum uncertainty (p≈0.5). This has no direct analogue in the meta-analysis literature and is a novel contribution

**Novelty Assessment:**
- The combination of (1) precision-weighted fusion from statistical meta-analysis, (2) entropy-based gating, and (3) prompt-anchored calibration is novel. No prior hallucination detection work uses this aggregation framework
- Most multi-signal detectors use simple averaging, learned weights, or trained probes. AG-SAR's approach is training-free and statistically principled

---

### 3.8 Span Detection (`aggregation/spans.py`)

**Algorithm:** Otsu threshold on token risks → expected-gap merging

**Implementation:**
```
threshold = otsu_threshold(risks)
n_above = sum(risks >= threshold)
max_gap = len(risks) // n_above    # expected gap between high-risk tokens
# Group contiguous high-risk tokens, merge if gap <= max_gap
```

**Correctness:** Straightforward and correct. The expected-gap formula is a simple but effective heuristic — if N% of tokens are risky, the expected gap between them is 100/N%.

---

### 3.9 Hook System (`hooks/`)

**ModelAdapter:** Pattern-based auto-detection via `nn.Module.get_submodule()` across 5 architecture families. Clean and extensible.

**EphemeralHiddenBuffer:** Per-token bfloat16 storage, cleared after signal computation. Memory-efficient.

**LayerHooks:** 2-point capture (pre-hook on post_attn_norm, forward-hook on layer output). Correctly separates h_resid_attn (before MLP) and h_resid_mlp (after MLP).

**Correctness:** The hook placement is critical for MLP JSD computation. The pre-hook on `post_attention_layernorm` captures the hidden state *before* normalization is applied, which is then normalized via `final_norm` before projection through `lm_head`. This is consistent with the logit lens methodology.

---

### 3.10 Numerics (`numerics.py`)

**Tracy-Widom CDF:** Cornish-Fisher expansion with exact TW₁ moments.
- Moments match Bornemann (2010): μ = -1.2065335745820, σ = 1.2680340580149, skew = 0.29346452408
- Accuracy < 0.005 for |z| < 3, sufficient for the operating range

**Effective Rank:** `exp(H(p))` where `p_i = s_i / Σs_j`. Standard definition from Roy & Vetterli (2007).

**Otsu Threshold/Coefficient:** Vectorized implementation of Otsu (1979). Correct maximization of between-class variance.

**EPS:** `float32_eps² ≈ 1.42e-14`. Principled: derived from machine precision, not arbitrary.

---

## 4. Literature Comparison & Positioning

### 4.1 Most Comparable Works

| Method | Year | Approach | Training? | Token-level? | Multi-signal? | Context-free? |
|---|---|---|---|---|---|---|
| **AG-SAR** | 2025 | 5 internal signals + precision fusion | No | Yes | Yes (5) | Yes |
| INSIDE/EigenScore | 2024 | Eigenvalues of response covariance | No | No (response) | No (1) | Yes |
| HARP | 2025 | Reasoning subspace projection | Yes (probe) | No | No (1) | Yes |
| LapEigvals | 2025 | Laplacian eigenvalues of attention | Yes (probe) | Yes | No (1) | Yes |
| Semantic Entropy | 2024 | Entropy over semantic clusters | No | No (response) | No (1) | Yes |
| SelfCheckGPT | 2023 | Self-consistency via sampling | No | Yes | No (1) | Yes |
| Semantic Entropy Probes | 2024 | Probes on hidden states | Yes (probe) | Yes | No (1) | Yes |

### 4.2 Key Differentiators

1. **Multi-signal fusion**: AG-SAR is the only method that extracts and fuses *multiple independent signals* from different transformer components (attention, MLP, hidden states, spectral dynamics). All competitors use a single signal type

2. **Zero-shot + token-level**: Most zero-shot methods (EigenScore, Semantic Entropy) operate at response level. Methods that do token-level detection (LapEigvals, Semantic Entropy Probes) require trained probes. AG-SAR achieves both without training

3. **Tracy-Widom spectral calibration**: No prior NLP work uses TW-calibrated BBP phase transition for hallucination detection. This is a genuine methodological import from random matrix theory

4. **Statistical meta-analysis framework**: The aggregation mechanism draws from a well-established but previously unused (in NLP) mathematical framework. The precision-weighted entropy-gated fusion is a novel synthesis

---

## 5. Identified Issues & Recommendations

### 5.1 Correctness Issues

| Issue | Severity | Location | Detail |
|---|---|---|---|
| TW scaling rate | **High** | `spt.py:57-59` | Missing `1/√W` factor in TW scaling denominator. The standardized statistic z_TW is √W times too small, making TW probabilities miscalibrated. Ordinal ranking preserved but absolute probabilities are wrong. See Section 3.4 for full derivation |
| Rayleigh naming | Cosmetic | CLAUDE.md | Called "Rayleigh survival function" but is actually the Rayleigh CDF (1 - exp(-x²/τ²) = F_Rayleigh(x; τ/√2)). Code math is correct |
| First token ENT=0.5 | Low | `detector.py:178` | First generated token has no attentions from prefill. Could undercount risk on first token |
| Precision matrix structure | Low | `calibration.py:121` | All 5 diagonal entries use inverse-variance weighting (not identity). PSP-MLP additionally get cross-correlation via Ledoit-Wolf. The "cross-signal precision" claim is accurate for the PSP-MLP pair; the other 3 signals participate via inverse-variance but without cross-correlation |
| ENT variance heuristic | Low | `calibration.py:96` | ENT variance set to `median([psp_var, mlp_var])` — a cross-signal heuristic, not computed from actual ENT values. This is because ENT uses direct mode and has no calibration tail values |

### 5.2 Design Gaps

| Gap | Impact | Detail |
|---|---|---|
| Small SPT window | Medium | For long low-rank prompts, window_size could be 3-5. TW asymptotics require larger matrices for accuracy. Compounded by the TW scaling bug — small W amplifies the √W error factor |
| PIT boundary behavior | Low | If response values exceed all prompt-tail values, PIT saturates near 1.0 (or near 0.0). No explicit extrapolation handling, but Haldane-Anscombe correction mitigates |
| No attention for first token | Low | Prefill doesn't return attentions. ENT defaults to 0.5 for first token |
| Circular buffer overwrites | By design | SPT window is fixed-size. Old states are lost. This is intentional (local dynamics) but means SPT cannot capture long-range spectral shifts |

### 5.3 Strengths

1. **Mathematically principled**: Every component has clear theoretical grounding — Otsu (1979), effective rank (Roy & Vetterli 2007), Tracy-Widom (Tracy & Widom 1994, Johnstone 2001), Ledoit-Wolf (2004), PIT, DerSimonian-Laird (1986), Hartung-Knapp-Sinha (2008)
2. **Zero parameters**: No learned weights, no hyperparameter tuning, no calibration datasets
3. **Fully vectorized**: No Python loops in any signal computation path
4. **Clean architecture**: Modular signals, principled calibration, statistically-grounded fusion
5. **Novel signal combination**: Each signal captures a different aspect of generation (attention patterns, MLP dynamics, geometric drift, spectral stability, directional coherence)

---

## 6. Verdict

### Research Validity: **Strong**

AG-SAR's implementation faithfully realizes its stated mechanisms. The theoretical grounding is sound and the code correctly implements the mathematical formulations. The five signals are mechanistically distinct and capture complementary aspects of transformer generation dynamics.

### Novelty: **Significant**

1. **Tracy-Widom calibrated spectral detection for NLP** — genuinely novel import from RMT
2. **Otsu coefficient on attention entropy as bimodality signal** — novel
3. **All-layer JSD on pre/post-MLP logits with adaptive candidate set** — novel
4. **Prompt-conditioned subspace projection with magnitude gating** — novel (distinct from HARP's model-intrinsic subspace)
5. **Cross-signal precision-weighted entropy-gated fusion** — novel synthesis of meta-analysis methodology

### Weaknesses

1. **TW scaling bug (High)**: The SPT signal's Tracy-Widom scaling is missing a `1/√W` factor, making absolute TW probabilities miscalibrated. The signal still functions as a relative ranking (monotonic distortion preserved), but the "Tracy-Widom calibrated" claim is not fully realized
2. Cross-signal precision coupling is active only between PSP and MLP; the other 3 signals get inverse-variance weighting but no cross-correlation
3. Small-window regimes (short or low-rank prompts) stress the TW approximation — compounded by the scaling bug
4. No empirical comparison against HARP, LapEigvals, or INSIDE in the current evaluation pipeline
5. First-token ENT gap is minor but breaks the "every token scored" claim
6. ENT variance is derived from peer signals, not from actual ENT observations

### Recommendation

**Priority 1 — Fix TW scaling**: In `spt.py:57-59`, change the scaling to:
```python
sigma_tw = sigma2 * (1.0 + sqrt_gamma) / (W ** 0.5) * (
    1.0 / (W ** 0.5) + 1.0 / (self._d ** 0.5)
) ** (1.0 / 3.0)
```
This adds the missing `1/√W` factor per Johnstone (2001) Theorem 1.1 for sample covariance eigenvalues.

**Priority 2 — Precision matrix**: Consider either (a) extending cross-correlation to all 5 signals by bootstrapping ENT/SPT/spectral_gap from the prompt tail, or (b) documenting the structure as "PSP-MLP cross-signal precision with inverse-variance weighting for ENT/SPT/spectral_gap."

**Priority 3 — ENT variance**: Consider computing ENT from actual prompt-tail attention distributions rather than inheriting from peer signals.

---

## 7. References

- Baik, Ben Arous & Péché (2005). Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices. *Ann. Probab.*
- Bornemann (2010). On the numerical evaluation of distributions in random matrix theory. *Markov Processes Relat. Fields*
- Chen et al. (2024). INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. *ICLR 2024*
- Binkowski et al. (2025). Hallucination Detection in LLMs Using Spectral Features of Attention Maps. *EMNLP 2025*
- DerSimonian & Laird (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials*
- Hartung, Knapp & Sinha (2008). *Statistical Meta-Analysis with Applications.* Wiley
- Hu et al. (2025). HARP: Hallucination Detection via Reasoning Subspace Projection. *arXiv:2509.11536*
- Johnstone (2001). On the distribution of the largest eigenvalue in principal components analysis. *Ann. Statist.*
- Kuhn et al. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG. *ICLR 2023*
- Ledoit & Wolf (2004). A well-conditioned estimator for large-dimensional covariance matrices. *JMVA*
- Marchenko & Pastur (1967). Distribution of eigenvalues for some sets of random matrices. *Math. USSR Sbornik*
- Otsu (1979). A threshold selection method from gray-level histograms. *IEEE Trans. SMC*
- Roy & Vetterli (2007). The effective rank: A measure of effective dimensionality. *EUSIPCO*
- Tracy & Widom (1994). Level-spacing distributions and the Airy kernel. *Comm. Math. Phys.*

---

## Appendix A: Claim-by-Claim Verification Log

This appendix documents every specific claim in the original audit, verified line-by-line against source code.

### A.1 Claims Verified CORRECT

| # | Claim | Source File | Verification |
|---|---|---|---|
| 1 | ENT = 1 - otsu_coefficient(normalized entropies) | `ent.py:17-26` | Code: `1.0 - otsu_coefficient(H.reshape(-1).cpu().numpy())` ✅ |
| 2 | ENT fully vectorized, no Python loops | `ent.py:23-26` | Single tensor chain: clamp→log2→sum→reshape ✅ |
| 3 | ENT first token defaults to 0.5 | `detector.py:178` | `ent = 0.5; if attentions is not None:` ✅ |
| 4 | ENT entropy normalized by log₂(seq_len) | `ent.py:23-25` | `log2_n = math.log2(seq_len); H = -(a * a.log2()).sum(dim=-1) / log2_n` ✅ |
| 5 | MLP candidate set via effective_rank | `detector.py:172-175` | `k = max(2, effective_rank(probs)); cand = torch.topk(logits, k).indices` ✅ |
| 6 | MLP candidate set includes emitted token | `detector.py:175` | `torch.unique(torch.cat([cand, torch.tensor([emitted_token_id]...)]))` ✅ |
| 7 | MLP JSD bounded [0,1] in bits, normalized by ln(2) | `_jsd_base.py:50` | `jsd_bits = (0.5 * kl_pm + 0.5 * kl_qm) / math.log(2)` ✅ |
| 8 | MLP uses final_norm before lm_head projection | `_jsd_base.py:37-41` | `pre_norm = self.final_norm(attn_stack); ... F.linear(pre_norm, w_subset)` ✅ |
| 9 | MLP batched all layers via stacked tensors | `_jsd_base.py:33-34` | `torch.stack([s.h_resid_attn for s in layer_states.values()])` ✅ |
| 10 | MLP uses xlogy for numerical stability | `_jsd_base.py:48-49` | `torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m)` ✅ |
| 11 | PSP calibration: center = mean, SVD, effective_rank | `psp.py:24-28` | `self._prompt_center = prompt_hidden.mean(dim=0); _, S, Vh = torch.linalg.svd(centered...); self._prompt_basis = Vh[:effective_rank(S)]` ✅ |
| 12 | PSP magnitude tau = median norm | `psp.py:29-30` | `norms = torch.norm(centered, dim=-1); self._tau = float(norms.median().item())` ✅ |
| 13 | PSP = 1 - proj/mag, magnitude gated | `psp.py:40-43` | `psp_raw = 1.0 - s_prompt; gates = 1.0 - torch.exp(-mags.square() / (self._tau ** 2)); psp = 0.5 + (psp_raw - 0.5) * gates` ✅ |
| 14 | PSP all-layer mean | `psp.py:44` | `return float(psp.mean().item())` ✅ |
| 15 | Gate formula is Rayleigh CDF for X~Rayleigh(τ/√2) | `psp.py:42` | `1-exp(-x²/τ²) = F_Rayleigh(x; σ=τ/√2)` since standard CDF is `1-exp(-x²/(2σ²))` ✅ |
| 16 | SPT MP upper edge σ²(1+√γ)² | `spt.py:54` | `mu_tw = sigma2 * (1.0 + sqrt_gamma) ** 2` ✅ |
| 17 | SPT σ² estimated via median(eigenvalues) | `spt.py:49` | `sigma2 = float(eigs.median().item())` ✅ |
| 18 | SPT ring buffer circular indexing | `spt.py:33-34` | `self._pos = (self._pos + 1) % self._size; self._count = min(self._count + 1, self._size)` ✅ |
| 19 | SPT eigenvalues from S²/W | `spt.py:47` | `eigs = (S ** 2) / W` ✅ |
| 20 | Spectral gap = λ₂/(λ₁+λ₂) | `spt.py:70` | `spectral_gap = lambda_2 / (lambda_1 + lambda_2 + EPS)` ✅ |
| 21 | TW CDF Cornish-Fisher with exact moments | `numerics.py:16-34` | μ=-1.2065335745820, σ=1.2680340580149, skew=0.29346452408; `z_cf = z - (skew/6)*(z*z-1.0); norm.cdf(z_cf)` ✅ |
| 22 | Effective rank = exp(H(p)) | `numerics.py:37-46` | `p = S / S.sum(); H = -(p * p.log()).sum(); return round(H.exp().item())` ✅ |
| 23 | Otsu: maximizes between-class variance | `numerics.py:58-62` | `between_var = w0 * w1 * (mu0 - mu1) ** 2; best_idx = int(np.argmax(between_var))` ✅ |
| 24 | EPS = float32_eps² | `numerics.py:12` | `EPS = float(torch.finfo(torch.float32).eps ** 2)` ✅ |
| 25 | PIT Haldane-Anscombe correction | `fusion.py:44-45` | `ranks = np.searchsorted(sorted_vals, response_vals, side='right').astype(float); return (ranks + 0.5) / (n + 1)` ✅ |
| 26 | Adaptive kappa = 1 + median(decisiveness) | `fusion.py:48-55` | Loops PSP,MLP sorted_vals, computes binary entropy, median(1-H), returns `1.0 + float(np.median(decisiveness))` ✅ |
| 27 | Entropy-gated fusion: E=(1-H)^κ, W=max(E@Ω,0) | `fusion.py:105-106` | `E = (1.0 - self._binary_entropy(P)) ** kappa; W = np.maximum(E @ omega, 0.0)` ✅ |
| 28 | Response-level: signal-first then fuse | `fusion.py:120-134` | Per-signal mean → normalize → same fusion formula ✅ |
| 29 | Calibration uses Ledoit-Wolf on (n_tail, 2) matrix | `calibration.py:123-126` | `psp_mlp_matrix = np.column_stack([mlp_arr, psp_arr]); cov_2x2, _ = ledoit_wolf(psp_mlp_matrix); prec_2x2 = np.linalg.inv(cov_2x2)` ✅ |
| 30 | 2×2 precision embedded at [1:3, 1:3] | `calibration.py:128` | `precision[1:3, 1:3] = prec_2x2` ✅ |
| 31 | Tail window = sqrt(prompt_len) | `calibration.py:21-22` | `min(int(math.ceil(math.sqrt(prompt_len))), prompt_len)` ✅ |
| 32 | Hook: pre-hook on post_attn_norm captures h_resid_attn | `layer_hooks.py:27-28,33-34` | `register_forward_pre_hook(self._capture_resid_attn)` where `self._h_resid_attn = args[0]` ✅ |
| 33 | Hook: forward-hook on layer captures h_resid_mlp | `layer_hooks.py:30-31,36-38` | `register_forward_hook(self._capture_resid_mlp_and_store)` where `buffer.store(self.layer_idx, self._h_resid_attn, output[0])` ✅ |
| 34 | Buffer stores bfloat16, last position only | `buffer.py:27-29` | `h_resid_attn[:, -1, :].detach().bfloat16()` ✅ |
| 35 | ModelAdapter: 5 architecture patterns | `adapter.py:9-20` | LLaMA/Mistral/Qwen/Gemma, Phi, GPT-2/Neo, Falcon, GPT-NeoX ✅ |
| 36 | Span detection: Otsu threshold + expected-gap merging | `spans.py:35-43` | `threshold = otsu_threshold(risks); max_gap = len(risks) // n_above` ✅ |
| 37 | Span grouping via np.diff/np.split | `spans.py:55-57` | `gaps = np.diff(high_risk_indices); split_points = np.nonzero(gaps > self.max_gap)[0] + 1; groups = np.split(...)` ✅ |
| 38 | Three entry points: detect, detect_from_tokens, score | `detector.py:325-350` | All three methods present, calling `_generation_loop` ✅ |
| 39 | Score uses teacher-forced decode | `detector.py:342-350` | Tokenizes full text, extracts response_ids, passes to `_generation_loop(input_ids, response_ids=response_ids)` ✅ |
| 40 | SPT window size = effective_rank(prompt SVD) | `detector.py:139` | `spt_window = effective_rank(prompt_S)` ✅ |

### A.2 Claims Found INCORRECT or INACCURATE

| # | Original Claim | Actual Code | Correction |
|---|---|---|---|
| 1 | **TW scaling verified correct** (Section 3.4, 5.1) | `spt.py:57-59`: `sigma2 * (1+sqrt_gamma) * (1/√W+1/√d)^{1/3}` | **WRONG.** Missing `1/√W` factor. Correct: `sigma2 * (1+sqrt_gamma) / √W * (1/√W+1/√d)^{1/3}`. Derivation: Johnstone (2001) Wishart scaling divided by W for sample covariance → factor of `(√W+√d)/W = (1+√γ)/√W` not `(1+√γ)` |
| 2 | **ENT/SPT/spectral_gap get "unit precision (identity diagonal)"** (Section 3.6) | `calibration.py:121`: `precision = np.diag(1.0 / np.maximum(diag_vars, EPS))` | **WRONG.** All 5 diagonal entries are `1/variance` (inverse-variance weighting), not 1.0. Only the PSP-MLP 2×2 block adds cross-correlation |
| 3 | **ENT variance "computed from prompt"** (implicit in Section 3.6) | `calibration.py:96`: `float(np.median([stats[s]["variance"] for s in stats]))` | **INCOMPLETE.** ENT variance = median of PSP and MLP variances (cross-signal heuristic). Not computed from actual ENT observations since ENT has no calibration tail values |
| 4 | **JSD pseudocode shows `xlogy(p, p/m)`** (Section 3.2) | `_jsd_base.py:48`: `torch.xlogy(p_pre, p_pre) - torch.xlogy(p_pre, m)` | **MINOR.** Code decomposes as `xlogy(p,p) - xlogy(p,m)` (avoids explicit division), not `xlogy(p, p/m)`. Mathematically equivalent but code is more numerically stable |
| 5 | **PSP pseudocode shows `\|\|V @ h_centered\|\|`** (Section 3.3) | `psp.py:39`: `torch.norm(H_centered @ V.T, dim=-1)` | **MINOR.** Code transposes the operation for batched efficiency: `H @ V^T` instead of `V @ H`. Mathematically equivalent since `\|\|h @ V^T\|\| = \|\|V @ h\|\|` |
| 6 | **Otsu step 4 describes between-class variance** (Section 3.1) | `numerics.py:62`: `w0 * w1 * (mu0 - mu1) ** 2` | **MINOR.** The audit's ENT section 3.1 pseudocode step 4 appears to describe within-class variance formula (sum of squared deviations per class), not the actual between-class variance `w₀w₁(μ₀-μ₁)²`. The code is correct; the pseudocode description was misleading |

### A.3 Novelty Claims Verified

| Claim | Verification | Status |
|---|---|---|
| ENT: Otsu coefficient on attention entropy is novel | No prior work found using bimodality coefficient of head entropy distributions | ✅ Novel |
| MLP: All-layer pre/post-MLP JSD with adaptive candidate set is novel | Prior work uses activation norms or probes, not JSD on projected logits | ✅ Novel |
| PSP: Distinct from HARP (prompt subspace vs. model-intrinsic subspace) | HARP decomposes unembedding layer; PSP decomposes prompt hidden states. Different input, different basis | ✅ Distinct |
| SPT: Tracy-Widom/BBP for hallucination detection is novel in NLP | No prior NLP work found using TW-calibrated phase transition for hallucination | ✅ Novel (but implementation has scaling bug) |
| Fusion: Meta-analysis framework novel for hallucination detection | No prior detector uses precision-weighted entropy-gated fusion | ✅ Novel |
| INSIDE comparison: EigenScore requires K generations, SPT uses single | INSIDE paper confirms K sampled responses needed for covariance | ✅ Accurate |
| LapEigvals comparison: Different matrices (attention Laplacian vs. hidden covariance) | LapEigvals paper confirms Laplacian of attention maps, not hidden states | ✅ Accurate |

### A.4 Literature Comparison Verified

| Comparison | Audit Claim | Verification |
|---|---|---|
| INSIDE/EigenScore: response-level, not token-level | ✅ Confirmed — EigenScore outputs one score per response |
| INSIDE: requires K sampled generations | ✅ Confirmed — samples K responses and computes covariance |
| HARP: supervised (trains probe) | ✅ Confirmed — trains linear probe on projections |
| HARP: SVD on unembedding layer | ✅ Confirmed — decomposes W_unembed into semantic/reasoning subspaces |
| LapEigvals: supervised (trains probe) | ✅ Confirmed — trains MLP probe on Laplacian eigenvalues |
| Semantic Entropy: response-level | ✅ Confirmed — clusters responses, computes entropy over clusters |
| AG-SAR is only method with zero-shot + token-level + multi-signal | ✅ No counterexample found in literature search |
