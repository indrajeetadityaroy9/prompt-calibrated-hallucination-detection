# AG-SAR Signal Discrimination Synthesis

## Problem Diagnosis

Three signals lack discrimination during confident generation:

| Signal | Observed | Variance | Root Cause |
|--------|----------|----------|------------|
| CUS    | ≈ 0.34 uniform | ≈ 0 | Copying heads always attend similarly to context during fluent generation |
| POS    | ≈ 0.007 | ≈ 0 | MLP shifts are tiny when model is confident → Otsu selects nothing → fallback sqrt(n) also gives ~0 |
| DPS    | ≈ 0.55 | 0.015 | Best signal, but projection ratio barely moves token-to-token |

**Noisy-OR floor**: sigmoid(z=0) = 0.5 for uninformative signals. Three signals at ~0.5 → 1-(0.5)³ = 0.875 for everything. Even DPS with slight variation gets drowned by CUS and POS contributing 0.5 each.

**Core insight**: The problem is twofold — (A) signals are intrinsically flat during confident generation, and (B) the fusion operator treats "I don't know" (p=0.5) as "50% risk" rather than "abstain."

---

## Gap 1: Signal Discrimination — Better Signals

### 1.1 Replace CUS with Lookback Ratio (Chuang et al., EMNLP 2024)

**Current CUS** aggregates context attention mass across all identified copying heads into a single scalar. This destroys per-head information and produces a uniform ~0.34.

**Lookback Ratio** preserves per-head structure:

```
LR(l,h,t) = Σ_{s ∈ context} attn[l,h,t,s] / Σ_{s=1}^{t} attn[l,h,t,s]
```

This is the fraction of attention head (l,h) places on context vs. all positions at token t. Key findings from the paper:

- **Per-head features, not scalar**: Use the full vector `[LR(l,h,t)]` for all heads as a feature vector
- **Both polarities matter**: Some heads have LR↑ = grounded, others have LR↑ = hallucinating. A linear classifier over the LR vector achieves 91.2% AUROC
- **Middle layers most informative**: Layers in the middle third carry most discriminative signal

**AG-SAR adaptation (training-free)**: We can't train a classifier, but we can use the *variance* of the LR vector across heads as a signal. During grounded generation, copying heads attend to context (high LR) while non-copying heads don't (low LR) — creating bimodal LR distribution. During hallucination, LR distribution becomes unimodal (all heads similar). Measure this:

```
LR_signal(t) = 1 - bimodality(LR_vector(t))
```

Where bimodality = Otsu inter-class variance / total variance (Otsu coefficient). Range [0,1], higher = more unimodal = riskier.

### 1.2 Add DoLa Layer-Contrast Signal (Chuang et al., ICLR 2024)

**Key idea**: Factual knowledge emerges in late layers, producing large JSD between early and late layer logit distributions. When the model hallucinates, late layers may deviate from early-settled patterns.

**DoLa detection signal** (adapted from decoding to detection):

```
# For each token t:
# 1. Get logit distributions at each layer via LogitLens
q_j(t) = softmax(final_norm(h_j(t)) · W_U)    for layer j

# 2. Compute JSD between candidate early layers and final layer N
JSD_j(t) = JSD(q_N(t) || q_j(t))

# 3. Select premature layer with maximum divergence
M(t) = argmax_{j ∈ early_layers} JSD_j(t)

# 4. DoLa contrast score
DoLa_score(t) = log q_N(x_t | x_{<t}) - log q_M(x_t | x_{<t})
```

**Why this helps**: DoLa score is high when the model's factual processing (late layers) strongly differs from syntactic processing (early layers) — indicating the model is "adding knowledge." Low DoLa score means the token was settled early (function word, copied token). This provides discrimination *independent* of surface confidence.

**Integration with existing hooks**: We already capture `h_resid_attn` and `h_resid_mlp` per layer. The LogitLens projection is `final_norm(h) · W_U`, which the `CandidateJSDSignal` already computes. We can add DoLa as a new signal with minimal overhead:

```python
# In topk_jsd.py, add:
def compute_dola(self, layer_states, generated_token_id):
    """Layer-contrast score for factuality."""
    # Get final layer distribution
    h_final = layer_states[max(layer_states.keys())].h_resid_mlp
    q_final = softmax(self.final_norm(h_final) @ self.lm_head.weight.T)

    # Find maximally divergent early layer
    max_jsd, best_layer = 0, None
    for j in early_layers:
        h_j = layer_states[j].h_resid_mlp
        q_j = softmax(self.final_norm(h_j) @ self.lm_head.weight.T)
        jsd = JSD(q_final, q_j)
        if jsd > max_jsd:
            max_jsd, best_layer = jsd, j

    # Contrast: log-prob difference for the actual generated token
    return log(q_final[generated_token_id]) - log(q_best[generated_token_id])
```

High DoLa → model added factual content in late layers (more confident, but potentially fabricated).
Low DoLa → token was linguistically determined early (less risky).

### 1.3 Add Dirichlet Energy from GSP Framework (arXiv 2510.19117)

**Key idea**: Treat each attention head's attention matrix as a graph, compute smoothness of token representations on that graph.

```
# Symmetrize attention
W^(ℓ,h) = ½[A^(ℓ,h) + (A^(ℓ,h))ᵀ]
L^(ℓ,h) = D^(ℓ,h) - W^(ℓ,h)    # Graph Laplacian

# Dirichlet energy (smoothness measure)
E^(ℓ) = Tr(X^(ℓ)ᵀ L^(ℓ) X^(ℓ)) = Σ_{i<j} W̄_ij ||x_i - x_j||²

# Fiedler value (algebraic connectivity)
λ₂^(ℓ) = second-smallest eigenvalue of L^(ℓ)
```

**Hallucination signatures**:
- **Factual**: Energy mountain (rises then drops 50-60×), Fiedler value rises monotonically to 0.90+
- **Hallucination**: Entropy/HFER oscillations, Fiedler value unstable

**AG-SAR adaptation**: Computing full spectral decomposition per token is expensive. But we can cheaply approximate the **Fiedler value** of the final-layer attention Laplacian using power iteration (1-2 iterations suffice for the sign of the gap). Or simpler: compute just the **Smoothness Index**:

```
SMI(t) = E(t) / Tr(X(t)ᵀ X(t))    # Normalized Dirichlet energy
```

This can be computed from the attention matrix and hidden states we already capture. High SMI = representations disagree across the attention graph = potential hallucination.

### 1.4 Improve DPS with ReDeEP Decomposition (ICLR 2025, arXiv 2410.11414)

ReDeEP decomposes hallucination into two measurable components:

```
Hallucination score = Σ_{l∈F} α · PKS_l - Σ_{(l,h)∈A} β · ECS_{l,h}
```

Where:
- **PKS** (Parametric Knowledge Score) = JSD between pre-FFN and post-FFN distributions — *this is exactly our existing POS/JSD signal*
- **ECS** (External Context Score) = cosine similarity between attended-context embedding and final hidden state — *this is related to our CUS signal*

**Key insight from ReDeEP**: The signals should be *subtracted*, not combined via Noisy-OR. High PKS (FFN adding knowledge) combined with low ECS (not attending to context) = hallucination. High PKS with high ECS = legitimate knowledge grounding.

This suggests our DPS formula `s_rsn / (s_ctx + s_rsn)` is on the right track (it's a ratio), but we should also incorporate the *absolute magnitudes*, not just the ratio. When both projections are small (model is uncertain about everything), DPS ≈ 0.5 but the risk is different from when both projections are large and roughly equal.

**Proposed enhancement**:
```
DPS_enhanced(t) = s_rsn / (s_ctx + s_rsn + eps) × (1 - exp(-||h_centered||² / τ²))
```

The second factor gates DPS by representation magnitude — when the hidden state is close to the context centroid (small `||h_centered||`), the ratio is unreliable and should be dampened toward 0.5 (uninformative), which the entropy gate (Gap 2) will then suppress.

---

## Gap 2: Fusion — Replace Noisy-OR with Entropy-Gated Combination

### 2.1 The Core Problem with Noisy-OR

Noisy-OR: `R(t) = 1 - Π(1 - p_i(t))`

When signal i is uninformative, `p_i = sigmoid(0) = 0.5`, contributing `(1-0.5) = 0.5` to the product. Three uninformative signals: `1 - 0.5³ = 0.875`. The operator has no concept of "abstention."

### 2.2 Solution A: Entropy-Gated Fusion (from AGFN, arXiv 2510.01677 & AECF, arXiv 2505.15417)

**Key idea**: Weight each signal by how *informative* it is, measured by the inverse of its binary entropy.

```
H_i(t) = -p_i log₂(p_i) - (1-p_i) log₂(1-p_i)    # Binary entropy, max at p=0.5

# Gate: suppress signals near p=0.5
w_i(t) = (1 - H_i(t) / H_max)^κ    where H_max = 1 bit, κ ≥ 1

# Weighted linear combination (not Noisy-OR)
R(t) = Σ w_i(t) · p_i(t) / (Σ w_i(t) + ε)
```

When `p_i = 0.5`: `H_i = 1.0`, `w_i = 0` → signal is completely suppressed.
When `p_i = 0.01` or `p_i = 0.99`: `H_i ≈ 0.08`, `w_i ≈ 0.92^κ` → signal contributes strongly.

**Properties**:
- Uninformative signals contribute exactly 0 weight (not 50% risk)
- No learned parameters (κ=2 works well as default, or use κ=1 for simplicity)
- If ALL signals are uninformative, R(t) defaults to prior (0 or 0.5, configurable)
- Signals that are confident in EITHER direction (risky or safe) contribute

**This directly solves the Noisy-OR floor problem.**

### 2.3 Solution B: Dempster-Shafer Evidential Fusion (from DS-Evidential, arXiv 2309.05919)

**Alternative** if we want a principled probabilistic framework:

Each signal produces a mass function `m_i` over {Hallucination, Faithful, Uncertain}:
```
m_i(H) = p_i · (1 - u_i)    # belief in hallucination
m_i(F) = (1-p_i) · (1-u_i)  # belief in faithful
m_i(Ω) = u_i                 # vacuous mass (uncertainty)
```

Where `u_i = H_i(t)` (binary entropy = uncertainty). Dempster's combination rule:
```
m_12(A) = Σ_{B∩C=A} m_1(B)·m_2(C) / (1 - K)
K = Σ_{B∩C=∅} m_1(B)·m_2(C)    # conflict
```

**Key property**: Vacuous mass (total ignorance) is the neutral element — combining with an uncertain signal leaves the other signals' beliefs unchanged. This is exactly what we need: uninformative signals don't inflate risk.

**Practical concern**: DS combination is more complex and the conflict normalization can behave unexpectedly with >2 sources. The entropy-gated approach (2.2) is simpler and more predictable.

### 2.4 Solution C: Fisher's p-value Combination (from arXiv 2408.12296)

Convert each signal probability to a one-sided p-value (probability of seeing this extreme a value under null = "faithful"):

```
p_value_i = 1 - p_i    # For risk probabilities where high p = hallucination

# Fisher's combination
X² = -2 Σ log(p_value_i)
# Under null (all faithful): X² ~ χ²(2k) where k = number of signals
# p_combined = 1 - CDF_χ²(X², df=2k)
```

**Property**: When `p_i = 0.5`, `p_value_i = 0.5`, contributing `-2·log(0.5) = 1.386` to the statistic. Under null, the expected contribution per signal is 2.0 (mean of χ²(2)). So a signal at p=0.5 contributes *less* than expected under null — it's actually mildly exculpatory. Only signals with `p_i > 0.5` (p_value < 0.5) drive the statistic toward rejection.

**This is mathematically clean but less intuitive for practitioners.**

### 2.5 Recommended Approach: Entropy-Gated Fusion

Solution A (entropy-gated) is recommended because:
1. **Zero parameters** (κ=1 or κ=2, both principled)
2. **Transparent**: weight = 0 when uninformative, easy to debug
3. **Compatible with prompt-anchored z-scoring**: the sigmoid probabilities feed directly into the entropy gate
4. **Graceful degradation**: if only DPS is informative, it naturally becomes the sole contributor
5. **Preserves the "no learned weights" philosophy** of AG-SAR

---

## Gap 3: Contrastive Signals — Discrimination During Confident Generation

### 3.1 Context-Aware Decoding Contrast (Shi et al., NAACL 2024)

**Key idea**: Compare token probability WITH context vs WITHOUT context.

```
CAD(t) = log P(x_t | context, x_{<t}) - log P(x_t | x_{<t})
```

Positive CAD = context *helped* produce this token (grounded).
Near-zero CAD = context didn't matter (potentially hallucinated or generic).
Negative CAD = context *hurt* — model fighting against context (highly suspicious).

**Why this helps for confident generation**: Even when the model is confident (high P(x_t|...)), the *difference* between context-conditioned and unconditioned probability can vary dramatically. A hallucinated entity name might have P=0.95 with context and P=0.93 without — near-zero CAD despite high confidence. A grounded fact might have P=0.95 with context and P=0.02 without — large positive CAD.

**Implementation cost**: Requires a second forward pass without context, which doubles compute. However, we can amortize: run the context-free pass once for the full prompt template (without retrieved context), cache the logits, and compare per-token during generation.

**Cheaper approximation**: Use attention mass to context (our existing LR signal) as a proxy for CAD. If LR is high → token attends to context → likely grounded. This avoids the second forward pass.

### 3.2 Activation Steering Direction Projection (ContextFocus, arXiv 2601.04131)

**Key idea**: Find a "truthfulness direction" in activation space by contrasting hidden states between faithful and hallucinated examples, then project generation-time activations onto it.

**AG-SAR adaptation (training-free)**: We already have context hidden states from prefill. We can define a *context-grounding direction* as:

```
d_ctx = mean(h_context_tokens) - mean(h_prompt_tokens)    # Direction from prompt toward context
```

Then for each generation token:
```
grounding_score(t) = cos(h_gen(t) - mean(h_prompt), d_ctx)
```

Positive = generation moves toward context representation. Near-zero or negative = generation ignores or moves away from context. This is computable from states we already capture with zero additional overhead.

### 3.3 HARP Reasoning Subspace Separation (arXiv 2509.11536)

Our DPS already uses the bottom singular vectors of `lm_head.weight` as V_rsn. HARP validates this approach and provides additional insight:

- The reasoning subspace is ~5% of dimensions (consistent with our spectral gap selection)
- Projecting onto reasoning vs semantic subspace separates factual from confabulated content
- The key refinement: use **per-token trajectory** through the reasoning subspace across layers, not just the final projection ratio

```
trajectory_drift(t) = Σ_l ||proj_rsn(h_l(t)) - proj_rsn(h_{l-1}(t))|| / ||proj_rsn(h_l(t))||
```

High trajectory drift in reasoning subspace = model is "searching" for an answer across layers = uncertain/risky. Low drift = answer was stable across layers = confident and likely grounded.

---

## Concrete Implementation Plan

### Phase 1: Fix the Fusion (Highest Impact, Lowest Effort)

Replace `_noisy_or_fusion` in `prompt_anchored.py` with entropy-gated fusion:

```python
def _entropy_gated_fusion(self, probabilities, n_tokens):
    """
    Entropy-gated fusion: suppress uninformative signals.

    w_i(t) = (1 - H_i(t))^κ  where H_i = binary entropy
    R(t) = Σ w_i p_i / Σ w_i   (weighted mean, not Noisy-OR)
    """
    kappa = 2  # Sharpness: κ=1 linear, κ=2 quadratic suppression

    weighted_sum = np.zeros(n_tokens)
    weight_sum = np.zeros(n_tokens)

    for sig, p in probabilities.items():
        # Binary entropy: H = -p·log₂(p) - (1-p)·log₂(1-p)
        p_clipped = np.clip(p, EPS, 1 - EPS)
        H = -(p_clipped * np.log2(p_clipped) + (1 - p_clipped) * np.log2(1 - p_clipped))

        # Gate: 0 at H=1 (p=0.5), 1 at H=0 (p=0 or 1)
        w = (1.0 - H) ** kappa

        weighted_sum += w * p
        weight_sum += w

    # When all signals uninformative, default to 0 (no evidence of risk)
    return np.where(weight_sum > EPS, weighted_sum / weight_sum, 0.0)
```

**Expected impact**: Eliminates the 0.875 floor. When CUS=0.34 (H≈0.93, w≈0.005) and POS=0.5 (H=1.0, w=0), only DPS contributes meaningfully. If DPS=0.55 (H≈0.99, w≈0.0001) — even DPS gets suppressed because 0.55 is barely different from 0.5. This means the system correctly outputs *low risk* for tokens where no signal has evidence.

### Phase 2: Add Lookback Ratio Bimodality Signal (Medium Effort)

Replace/augment CUS in `copying_heads.py`:

```python
def compute_lookback_ratio_signal(self, attention_slices, prompt_len, gen_pos):
    """
    Lookback ratio bimodality as context utilization signal.

    LR(l,h) = Σ_{s<prompt_len} attn[l,h,gen_pos,s]  (already context_mass)
    Signal = Otsu inter-class variance of LR vector / total variance
    """
    lr_values = []
    for layer_idx, head_idx in self.copying_heads:
        if layer_idx in attention_slices:
            attn = attention_slices[layer_idx]
            context_mass = attn[head_idx, :prompt_len].sum().item()
            lr_values.append(context_mass)

    if len(lr_values) < 4:
        return 0.5  # Not enough heads for bimodality

    lr = np.array(lr_values)
    total_var = lr.var()
    if total_var < EPS:
        return 0.5  # All heads identical → uninformative

    # Otsu inter-class variance (reuse existing function)
    threshold = otsu_threshold(lr)
    class1 = lr[lr <= threshold]
    class2 = lr[lr > threshold]

    if len(class1) == 0 or len(class2) == 0:
        return 1.0  # Unimodal → no discrimination → risky

    # Inter-class variance / total variance = Otsu coefficient
    w1, w2 = len(class1)/len(lr), len(class2)/len(lr)
    mu1, mu2 = class1.mean(), class2.mean()
    inter_var = w1 * w2 * (mu1 - mu2)**2
    otsu_coeff = inter_var / (total_var + EPS)

    # High bimodality (otsu_coeff→1) = healthy separation = low risk
    # Low bimodality (otsu_coeff→0) = uniform attention = high risk
    return float(np.clip(1.0 - otsu_coeff, 0.0, 1.0))
```

### Phase 3: Add DoLa Layer-Contrast Signal (Medium Effort)

Add to `topk_jsd.py`:

```python
def compute_dola_score(self, layer_states, generated_token_id, candidate_set):
    """
    DoLa-style layer contrast: log P_final(token) - log P_premature(token).

    Premature layer = argmax JSD(q_final, q_j) over early layers.
    """
    layer_indices = sorted(layer_states.keys())
    n_layers = len(layer_indices)
    final_layer = layer_indices[-1]
    early_layers = layer_indices[:n_layers//2]  # First half

    # Final layer distribution (on candidate set for efficiency)
    h_final = layer_states[final_layer].h_resid_mlp
    q_final = self._logit_lens_probs(h_final, candidate_set)

    # Find premature layer with max JSD
    max_jsd, best_q = 0.0, None
    for j in early_layers:
        h_j = layer_states[j].h_resid_mlp
        q_j = self._logit_lens_probs(h_j, candidate_set)
        jsd = safe_jsd(q_final, q_j)
        if jsd > max_jsd:
            max_jsd = jsd
            best_q = q_j

    if best_q is None:
        return 0.0

    # Find generated token in candidate set
    token_idx = (candidate_set == generated_token_id).nonzero()
    if len(token_idx) == 0:
        return 0.0

    idx = token_idx[0].item()
    log_p_final = np.log(q_final[idx] + EPS)
    log_p_early = np.log(best_q[idx] + EPS)

    return float(log_p_final - log_p_early)
```

### Phase 4: Context-Grounding Direction (Low Effort, High Value)

Add to `context_grounding.py` or as new signal:

```python
def compute_grounding_direction_score(self, h_gen, context_center, prompt_center):
    """
    Projection of generation hidden state onto context-grounding direction.

    d_ctx = context_center - prompt_center  (direction toward context)
    score = cos(h_gen - prompt_center, d_ctx)

    Positive = generation aligns with context. Negative = moves away.
    """
    d_ctx = context_center - prompt_center
    d_ctx_norm = torch.norm(d_ctx)
    if d_ctx_norm < EPS:
        return 0.5

    h_relative = h_gen - prompt_center
    cos_sim = (h_relative @ d_ctx) / (torch.norm(h_relative) * d_ctx_norm + EPS)

    # Map from [-1, 1] to [0, 1] risk score: -1 (toward context) → 0, +1 (away) → 1
    return float((1.0 - cos_sim.item()) / 2.0)
```

This requires computing `prompt_center` (mean of non-context prompt tokens) during prefill, which we can do alongside context_center.

---

## Summary of Changes by Priority

| Priority | Change | File | Impact |
|----------|--------|------|--------|
| **P0** | Replace Noisy-OR with entropy-gated fusion | `prompt_anchored.py` | Eliminates 0.875 floor |
| **P1** | Replace CUS with lookback ratio bimodality | `copying_heads.py` | Signal goes from 0-variance to discriminative |
| **P1** | Add DoLa layer-contrast signal | `topk_jsd.py` | New signal independent of surface confidence |
| **P2** | Add context-grounding direction signal | `context_grounding.py` | Provides contrastive discrimination |
| **P2** | Gate DPS by representation magnitude | `context_grounding.py` | Suppresses DPS when unreliable |
| **P3** | Add Dirichlet energy / Fiedler value signal | New file or `numerics.py` | Spectral validation signal |

## Novelty Argument

The combined system is novel because:

1. **Entropy-gated fusion** has not been applied to training-free hallucination detection. AGFN/AECF use it with learned components; we derive it from first principles as the natural solution to the Noisy-OR floor.

2. **Lookback ratio bimodality** (using Otsu coefficient of the per-head LR vector) is a new training-free alternative to the trained linear classifier in Chuang et al.

3. **DoLa for detection** (not decoding): The original DoLa paper uses layer contrast to *improve* generation. We repurpose the contrast score as a *detection signal* — a fundamentally different application.

4. **Context-grounding direction** from prefill statistics combines the spirit of ContextFocus (activation steering) with our training-free, per-input calibration approach.

5. **DSG's unified framework**: No prior work combines all of: per-head attention analysis (Lookback Lens), layer-contrast signals (DoLa), subspace projection (HARP/DPS), and entropy-gated fusion in a single training-free detector.

## Key References

- Chuang et al. "Lookback Lens for Detecting and Mitigating Contextual Hallucinations" (EMNLP 2024, arXiv 2407.07071)
- Chuang et al. "DoLa: Decoding by Contrasting Layers Improves Factuality" (ICLR 2024, arXiv 2309.03883)
- Graph Signal Processing Framework for Hallucination Detection (arXiv 2510.19117)
- ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation (ICLR 2025, arXiv 2410.11414)
- AGFN: Dual Entropy-Gate Fusion (arXiv 2510.01677)
- AECF: Adaptive Entropy-Gated Contrastive Fusion (arXiv 2505.15417)
- HARP: Reasoning Subspace via SVD of Unembedding (arXiv 2509.11536)
- ContextFocus: Activation Steering Directions (arXiv 2601.04131)
- Dempster-Shafer Evidential Fusion with Contextual Discounting (arXiv 2309.05919)
- Fisher's p-value combination for Multiple Testing (arXiv 2408.12296)
