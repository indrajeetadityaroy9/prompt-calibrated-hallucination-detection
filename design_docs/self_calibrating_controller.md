# Self-Calibrating AG-SAR (SC-AGSAR) v10.0

## Problem Statement

The v9.0 task-adaptive approach uses hardcoded parameter presets:
```python
TASK_PRESETS = {
    "qa": {"calibration_temperature": 1.2, "dispersion_k": 5, ...},
    "summarization": {"calibration_temperature": 2.5, "dispersion_k": 10, ...},
}
```

**Fundamental limitations:**
1. Requires manual tuning per task/domain
2. Doesn't generalize to unseen tasks
3. Task detection from dataset name is fragile
4. Ignores per-sample difficulty variation

**Goal:** Derive calibration parameters mathematically from the model's internal dynamics, making AG-SAR a true zero-shot universal detector.

---

## Core Insight: Internal Signals Contain Calibration Information

AG-SAR already computes signals that encode task/sample characteristics:

| Signal | What it measures | Task correlation |
|--------|------------------|------------------|
| Attention Entropy | Focus vs diffuse attention | QA→low, Summarization→high |
| Authority Variance | Agreement across tokens | Easy→low, Hard→high |
| MLP Divergence | Model uncertainty | Confident→low, Uncertain→high |
| Top-k Dispersion | Semantic consistency | Grounded→low, Hallucinated→high |

**Key principle:** Instead of detecting task type → lookup preset, use these signals directly to modulate parameters in closed form.

---

## Mathematical Framework

### 1. Entropy-Adaptive Dispersion (replaces fixed `dispersion_k`)

**Intuition:** When attention is diffuse (high entropy), we need more tokens to capture semantic meaning. When focused (low entropy), fewer tokens suffice.

**Current (fixed):**
```python
dispersion_k = 5  # Fixed
```

**Self-calibrating:**
```python
# Attention entropy per response token
H(t) = -Σⱼ A[t,j] log A[t,j]
H_max = log(seq_len)  # Maximum possible entropy
H_normalized = H(t) / H_max  # ∈ [0, 1]

# Adaptive k: more tokens when attention is diffuse
k_effective(t) = k_min + (k_max - k_min) × H_normalized(t)
           = 3 + (15 - 3) × H_normalized(t)
           = 3 + 12 × H_normalized(t)
```

**Mathematical justification:** Information-theoretic—entropy measures the effective number of attended positions. When attention entropy is high, the information is spread across more tokens, requiring larger k to capture it.

---

### 2. Variance-Adaptive Aggregation (replaces fixed percentile)

**Intuition:** When authority scores are consistent (low variance), mean aggregation is appropriate. When scores vary wildly (high variance), conservative aggregation prevents overconfidence.

**Current (fixed):**
```python
aggregation = "percentile_10"  # or "mean"
```

**Self-calibrating:**
```python
# Compute authority variance across response tokens
σ²_auth = Var(authority[response_tokens])
σ_normalized = σ_auth / max(σ_auth_running_mean, ε)  # Relative to typical

# Interpolate between mean and conservative
α = sigmoid(γ × (σ_normalized - 1))  # γ controls sensitivity

score = α × percentile_10(authority) + (1 - α) × mean(authority)
```

**Mathematical justification:** When variance is typical (σ_normalized ≈ 1), we use a mix. When variance is high (σ_normalized >> 1), we shift toward conservative. This is a soft version of robust statistics.

---

### 3. Confidence-Modulated Temperature (replaces fixed `calibration_temperature`)

**Intuition:** The model's own softmax confidence indicates how much to trust its outputs. Overconfident predictions need softening (T > 1), underconfident need sharpening (T < 1).

**Current (fixed):**
```python
temperature = 1.8
```

**Self-calibrating:**
```python
# Model confidence from softmax (already computed)
c = mean(confidence[response_tokens])

# Expected calibration: well-calibrated models have accuracy ≈ confidence
# If confidence is high but we don't trust it → increase T (soften)
# Use entropy of logits as secondary signal
H_logits = entropy(softmax(logits))
H_max = log(vocab_size)
uncertainty_ratio = H_logits / H_max  # ∈ [0, 1]

# Temperature adjustment
T_effective = T_base × (1 + β × (c - uncertainty_ratio))

# If confidence >> uncertainty → overconfident → T > 1
# If confidence << uncertainty → underconfident → T < 1
```

**Mathematical justification:** This implements online calibration. The gap between confidence and entropy-based uncertainty indicates miscalibration direction.

---

### 4. Stability-Gated Authority (replaces fixed `parametric_weight`)

**Intuition:** The MLP divergence (already computed) measures how much the MLP "overrides" attention. High divergence means the model is uncertain and relying on parametric knowledge.

**Current (fixed):**
```python
parametric_weight = 0.5
```

**Self-calibrating:**
```python
# MLP divergence already computed in unified gating
δ(t) = 1 - cos_sim(h_attn[t], h_block[t])

# Gate interpolates between attention-based and confidence-based
# High divergence → trust attention less → higher parametric weight
w_parametric(t) = base_weight + (1 - base_weight) × δ(t)
                = 0.3 + 0.7 × δ(t)

# Final gated authority
A_gated(t) = (1 - w_parametric(t)) × A_flow(t) + w_parametric(t) × confidence(t)
```

**Mathematical justification:** MLP divergence indicates when the model's feedforward processing disagrees with attention-based information flow. In these cases, the attention-based uncertainty signal is less reliable.

---

## Unified Self-Calibrating Score Function

Combining all components:

```python
def compute_self_calibrating_score(
    authority_scores: Tensor,      # Per-token authority from flow
    attention_weights: Tensor,     # Raw attention for entropy
    confidence: Tensor,            # Model softmax confidence
    h_attn: Tensor,                # Pre-MLP hidden states
    h_block: Tensor,               # Post-MLP hidden states
    logits: Tensor,                # Output logits for temperature
) -> float:
    """
    Compute uncertainty score with all parameters derived from internal signals.

    No hardcoded task-specific parameters.
    """
    # === 1. Entropy-Adaptive Dispersion ===
    attn_entropy = compute_attention_entropy(attention_weights)
    k_effective = 3 + 12 * attn_entropy.mean()
    k_int = int(torch.clamp(k_effective, 3, 15).item())

    # Semantic dispersion with adaptive k
    dispersion = compute_semantic_dispersion(logits, k=k_int)

    # === 2. Variance-Adaptive Aggregation ===
    auth_var = authority_scores.var()
    auth_var_normalized = auth_var / (running_auth_var + 1e-8)
    alpha = torch.sigmoid(2.0 * (auth_var_normalized - 1.0))

    # Soft interpolation between mean and conservative
    auth_mean = authority_scores.mean()
    auth_p10 = torch.quantile(authority_scores, 0.1)
    authority_aggregated = alpha * auth_p10 + (1 - alpha) * auth_mean

    # === 3. Confidence-Modulated Temperature ===
    conf_mean = confidence.mean()
    logit_entropy = compute_logit_entropy(logits)
    T_effective = 1.0 + 0.5 * (conf_mean - logit_entropy)
    T_effective = torch.clamp(T_effective, 0.5, 3.0)

    # === 4. Stability-Gated Fusion ===
    divergence = 1 - F.cosine_similarity(h_attn, h_block, dim=-1).mean()
    w_parametric = 0.3 + 0.7 * divergence

    # Fuse authority with confidence
    raw_score = (1 - w_parametric) * (1 - authority_aggregated) + w_parametric * (1 - conf_mean)

    # === 5. Apply Adaptive Temperature ===
    # Temperature scaling via logit transform
    logit_score = torch.log(raw_score / (1 - raw_score + 1e-8))
    calibrated_logit = logit_score / T_effective
    final_score = torch.sigmoid(calibrated_logit)

    # === 6. Semantic Dispersion Modulation ===
    # High dispersion → boost uncertainty
    dispersion_factor = 1.0 + dispersion_sensitivity * dispersion
    final_score = final_score * dispersion_factor

    return torch.clamp(final_score, 0.0, 1.0)
```

---

## Online Statistics Tracking

To normalize signals relative to their typical values, we maintain running statistics:

```python
@dataclass
class OnlineStats:
    """Welford's algorithm for streaming mean/variance."""
    count: int = 0
    mean: float = 0.0
    M2: float = 0.0

    def update(self, x: float) -> Tuple[float, float]:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        var = self.M2 / self.count if self.count > 1 else 1.0
        return self.mean, var

    @property
    def std(self) -> float:
        return (self.M2 / self.count) ** 0.5 if self.count > 1 else 1.0


class SCCalibrator:
    """Self-calibrating statistics tracker."""

    def __init__(self, decay: float = 0.995):
        self.authority_var_stats = OnlineStats()
        self.entropy_stats = OnlineStats()
        self.divergence_stats = OnlineStats()
        self.decay = decay

    def normalize(self, value: float, stats: OnlineStats) -> float:
        """Z-score normalization with running statistics."""
        if stats.count < 10:  # Warmup period
            return 0.5  # Neutral value
        return (value - stats.mean) / (stats.std + 1e-8)
```

---

## Implementation Roadmap

### Phase 1: Core Self-Calibration (Immediate)

1. Add entropy computation to attention extraction
2. Replace fixed `dispersion_k` with entropy-adaptive formula
3. Replace fixed aggregation with variance-adaptive interpolation
4. Add running statistics tracking

**Files to modify:**
- `src/ag_sar/measures/authority.py` - Add variance-adaptive aggregation
- `src/ag_sar/measures/semantics.py` - Add entropy-adaptive dispersion
- `src/ag_sar/engine.py` - Add online stats, integrate SC score

### Phase 2: Advanced Calibration (Follow-up)

1. Confidence-modulated temperature
2. Stability-gated fusion refinement
3. Conformal calibration for threshold selection

### Phase 3: Learned Components (Future)

1. Optional learned calibration head (requires training data)
2. Head importance weighting via gradient

---

## Expected Benefits

| Aspect | v9.0 Task-Adaptive | v10.0 Self-Calibrating |
|--------|-------------------|------------------------|
| Task detection | From dataset name | Not needed |
| Parameter source | Hardcoded presets | Derived from signals |
| Generalization | Limited to known tasks | Universal |
| Per-sample adaptation | No | Yes |
| Warmup needed | No | Yes (~10 samples) |
| Theoretical grounding | Empirical | Information-theoretic |

---

## Validation Plan

1. **Ablation study**: Enable each self-calibrating component individually
2. **Compare with v9.0**: Same datasets, measure AUROC/ECE
3. **Zero-shot transfer**: Test on unseen task types (e.g., dialogue, code)
4. **Calibration curves**: Plot reliability diagrams pre/post SC

---

## References

1. Adaptive Temperature Scaling (EMNLP 2024)
2. Semantic Entropy (Nature 2024)
3. Conformal Prediction for LLMs (NeurIPS 2024)
4. HIES: Head Importance-Entropy Score (2025)
5. Online Entropy Matching (NeurIPS 2024)
