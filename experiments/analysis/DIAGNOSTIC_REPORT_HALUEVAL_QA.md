# AG-SAR Signal Isolation Diagnostic Report
## HaluEval QA Dataset Analysis

**Model:** `meta-llama/Llama-3.1-8B-Instruct`
**Samples:** 100 (45 Hallucinations, 55 Facts)
**Date:** 2025-01-08

---

## Executive Summary

The diagnostic reveals **fundamental issues** with the current AG-SAR approach on HaluEval QA:

1. **Authority is MASSIVELY INVERTED** (AUROC = 0.1289) - Hallucinations have LOWER authority
2. **Varentropy is the ONLY strong signal** (AUROC = 0.8566) - Works as expected
3. **Severe length bias** (Hall: 18 tokens vs Fact: 3.7 tokens) - 4.9x difference
4. **The multiplicative equation destroys the varentropy signal** by mixing it with inverted authority

---

## Question 1: Signal Isolation Check

### Individual Component AUROCs

| Component | AUROC | Direction | Interpretation |
|-----------|-------|-----------|----------------|
| **Authority (A)** | **0.1289** | **INVERTED** | Hall=0.746 < Fact=0.787 |
| **Varentropy (V)** | **0.8566** | ✓ Correct | Hall=1.153 > Fact=0.477 |
| **Dispersion (D)** | **0.7160** | ✓ Correct | Hall=0.124 > Fact=0.077 |
| **LogProb (P)** | 0.5620 | Weak | Nearly random |
| **Entropy (H)** | 0.7798 | ✓ Correct | Hall=0.727 > Fact=0.381 |

### Statistical Summary

```
Component     Hall Mean ± Std    Fact Mean ± Std    Delta
─────────────────────────────────────────────────────────
Authority     0.746 ± 0.023      0.787 ± 0.028      -0.041
Varentropy    1.153 ± 0.726      0.477 ± 0.313      +0.676
Dispersion    0.124 ± 0.072      0.077 ± 0.067      +0.047
LogProb      -2.014 ± 1.026     -2.853 ± 3.528      +0.838
Entropy       0.727 ± 0.436      0.381 ± 0.251      +0.345
```

### Key Finding

**Authority is ANTI-predictive** (AUROC < 0.5 means flipping the sign would improve it):
- When the model hallucinates, it looks **LESS** at the prompt/context
- This makes sense: fabrication comes from parametric memory, not grounded context
- The v3.4 equation `Score = A × (1-D) × f(V)` **multiplies by a broken signal**

**Varentropy alone achieves AUROC = 0.8566** - this is the primary useful signal.

---

## Question 2: Token-Level Dynamics

### Hallucination Example (Sample 0)

**Question:** Which wildlife sanctuary is located in the town in which the Hollis/Brookline High School services?
**Answer (HALLUCINATION):** The Hollis/Brookline High School is located in the Talbot-Taylor Wildlife Sanctuary.

```
Token           Varentropy   Authority   Notes
─────────────────────────────────────────────────
"The"           1.23         0.80
"Hollis"        2.69         0.83        High V (uncertain start)
"/Brookline"    ~0.00        0.72        Confident (copying)
"High School"   ~0.00        0.75        Confident (copying)
"is located"    1.20         0.78
"in the"        1.16         0.74
"Talbot"        0.39         0.80        Wrong entity begins
"-Taylor"       4.63         0.69        **SPIKE** (fabrication)
"Wildlife"      5.69         0.67        **MAX SPIKE** (wrong!)
"Sanctuary"     0.22         0.69
"."             3.18         0.71        End uncertainty
─────────────────────────────────────────────────
Aggregates: V_mean=1.11, V_max=5.69, A_mean=0.74
```

### Fact Example (Sample 1)

**Question:** What American songwriter, born in 1932, wrote a pop hit single released by Nancy Sinatra?
**Answer (FACT):** Billy Edward "Edd" Wheeler

```
Token           Varentropy   Authority   Notes
─────────────────────────────────────────────────
"Billy"         0.44         0.83        Slightly uncertain (name)
"Edward"        0.28         0.77        Confident
""Edd""         0.07         0.72        Very confident
"Wheeler"       0.05         0.79        Very confident
─────────────────────────────────────────────────
Aggregates: V_mean=0.14, V_max=0.44, A_mean=0.76
```

### Key Finding

- **Hallucinations show BURST patterns**: V spikes to 4-6 at fabricated entities
- **Facts show STABLE low V**: V_max rarely exceeds 1.0
- **Signal is LOCALIZED**: The "Wildlife" token (wrong entity) has V=5.69
- **Mean aggregation washes out the burst**: V_mean=1.11 vs V_max=5.69

**Implication:** Consider `max` or `quantile_90` aggregation to capture bursts.

---

## Question 3: Instruction Tuning Factor

### Template Verification

```
Expected Llama 3.1 Instruct format:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

### Template Match Analysis

| Token | Status | Present |
|-------|--------|---------|
| `<\|begin_of_text\|>` | ✓ | Yes |
| `<\|start_header_id\|>user<\|end_header_id\|>` | ✓ | Yes |
| `<\|start_header_id\|>assistant<\|end_header_id\|>` | ✓ | Yes |
| `<\|eot_id\|>` | ✓ | Yes |

**Template is CORRECT** for Llama 3.1 Instruct.

### Potential Issues

1. **HaluEval was generated with GPT-3.5/4, NOT Llama**
   - Answers use GPT's writing style, not Llama's
   - We're measuring Llama's uncertainty about **GPT's phrasing**, not factual uncertainty

2. **Knowledge context format differs from training**
   - Llama was trained on "Document: ... Question: ..." formats
   - HaluEval uses "Knowledge: ... Question: ..."

3. **Cross-model evaluation artifact**
   - Llama may be "confidently hallucinating" because it would phrase differently
   - A correct fact in GPT's words may look "uncertain" to Llama

### Recommendation

Test on a **Llama-generated** hallucination dataset or use **model-native generation** for fairer evaluation.

---

## Question 4: Length Bias Analysis

### Response Length Distribution

```
Category        Mean ± Std    Range      Count
─────────────────────────────────────────────
Hallucinations  18.0 ± 1.7    [4, 50]    n=45
Facts           3.7 ± 1.7     [1, 8]     n=55
─────────────────────────────────────────────
Difference:     +14.4 tokens (4.9x longer)
```

### Severity: **CRITICAL**

This is a **4.9x length difference** between classes:
- Facts are typically **1-2 word answers**: "triathlete", "October 22, 1992"
- Hallucinations are **verbose explanations**: "The Hollis/Brookline High School is located in..."

### Impact on Aggregation

With `mean` aggregation:
- Short facts (3-4 tokens) are dominated by **first-token noise**
- Long hallucinations (18+ tokens) get **smoothed out**, hiding V spikes

### Recommended Solutions

1. **First-K Tokens**: `V[0:5].mean()` - equalize comparison window
2. **Max Aggregation**: `V.max()` - capture worst-case uncertainty
3. **Length Normalization**: `V.sum() / sqrt(len)` - account for length
4. **Last Token**: `V[-1]` - end-of-generation uncertainty

---

## Question 5: Ground Truth Integrity

### Potential Label Issues Detected

#### 1. Suspicious "Facts" (Low V, but labeled correct)

| Sample | V_mean | Question | Answer |
|--------|--------|----------|--------|
| 31 | 0.133 | Which English philosopher did Henry Seymour believe authored plays under the pseudonym of William Shakespeare? | Sir Francis Bacon |
| 1 | 0.139 | What American songwriter, born in 1932, wrote a pop hit single released by Nancy Sinatra? | Billy Edward "Edd" Wheeler |
| 42 | 0.140 | When was the American rapper from Atlanta, Georgia born who's song is "Sneakin'"? | October 22, 1992 |

**Analysis:** These appear to be **correctly labeled facts** - the model is appropriately confident on correct answers.

#### 2. Suspicious "Hallucinations" (High V, model is uncertain)

| Sample | V_mean | Question | Answer | Issue |
|--------|--------|----------|--------|-------|
| 47 | 3.70 | Who won the singles title at a 1973 tennis tournament...? | Serena Williams won... | Anachronism (Serena wasn't born) - Model is **appropriately uncertain** |
| 23 | 2.40 | Barbara Linares is a model for what parent company...? | Barbara Linares is the model of an independent... | Wrong company - Model is **appropriately uncertain** |
| 18 | 2.26 | Who wrote Doomquest...? | Stan Lee and Jack Kirby | Wrong authors - Model is **appropriately uncertain** |

**Analysis:** These are **correctly labeled hallucinations** and the model's high uncertainty is **appropriate**. AG-SAR is working correctly here.

#### 3. Dangerous: Confident Hallucinations (Low V, but wrong)

| Sample | V_mean | Question | Answer |
|--------|--------|----------|--------|
| 3 | 0.316 | Reagan and Give Us Our Skeletons were both which kind of films? | ...both documentaries |
| 12 | 0.464 | In which year was the author of Clara S born? | Elfriede Jelinek was born in 1945 |
| 19 | 0.594 | Light Chasers... led by which singer/songwriter? | Craig Minowa |

**Analysis:** These are **confident hallucinations** - the model is **wrong but confident**. These are the hardest cases to detect and represent the true challenge.

### Label Quality Assessment

- **No obvious mislabeling detected** in the inspected samples
- The "confident hallucinations" are genuine failures of uncertainty quantification
- Some high-V hallucinations show the model IS uncertain when fabricating

---

## Root Cause Analysis

### Why the Current Equation Fails

The v3.4 equation:
```
Score = A × (1-D) × sigmoid(V-τ)
```

**Problem 1: Authority is INVERTED**
- Expected: High A on facts, Low A on hallucinations
- Reality: Hall A=0.746, Fact A=0.787 (inverted!)
- Multiplying by A **destroys** the good V signal

**Problem 2: Multiplicative Interaction**
- Even if we flip A to (1-A), multiplying correlated signals can:
  - Double-count when they agree
  - Cancel when one is weak

**Problem 3: Length Bias**
- Mean aggregation over different lengths is biased
- Short facts dominated by noise, long hallucinations get smoothed

### Mathematical Proof of Failure

If we compute the final score:
- Hall: A=0.75 × (1-D)=0.88 × sigmoid(V=1.15) = 0.75 × 0.88 × 0.76 = **0.50**
- Fact: A=0.79 × (1-D)=0.92 × sigmoid(V=0.48) = 0.79 × 0.92 × 0.62 = **0.45**

The hallucination gets a **HIGHER** score than the fact! This is inverted.

---

## Recommended Solutions

### Solution 1: Pure Varentropy (Simplest, AUROC ≈ 0.86)

```python
def compute_uncertainty_v4(varentropy, tau=0.5, k=2.0):
    """Varentropy-only score. Ignores broken Authority."""
    return torch.sigmoid((varentropy.mean() - tau) * k)
```

**Pros:** Uses the strongest signal directly
**Cons:** Ignores context grounding information

### Solution 2: Authority Inversion

```python
def compute_uncertainty_v4b(authority, varentropy):
    """Flip authority direction since it's inverted."""
    context_detachment = 1.0 - authority  # High when NOT looking at context
    return context_detachment * varentropy
```

**Pros:** Uses both signals correctly
**Cons:** May over-penalize legitimate parametric retrieval

### Solution 3: Max-Token Aggregation

```python
def aggregate_uncertainty(varentropy, method="max"):
    """Use max to capture burst patterns."""
    if method == "max":
        return varentropy.max()
    elif method == "quantile_90":
        return torch.quantile(varentropy, 0.9)
```

**Pros:** Captures localized uncertainty spikes
**Cons:** Sensitive to outliers

### Solution 4: Conditional Architecture

```python
def compute_uncertainty_v4c(authority, varentropy, dispersion):
    """Switch logic based on context engagement."""
    is_grounded = authority > 0.78  # Above fact mean

    if is_grounded:
        # Trust context-based generation
        return dispersion  # Low dispersion = good
    else:
        # Model is "making stuff up" - use varentropy
        return varentropy
```

**Pros:** Handles both modes
**Cons:** Threshold sensitivity

### Solution 5: First-K Token Focus (Handles Length Bias)

```python
def compute_uncertainty_v4d(varentropy, k=5):
    """Only look at first K tokens to handle length bias."""
    return varentropy[:k].mean()
```

**Pros:** Equalizes comparison across lengths
**Cons:** May miss late-sequence fabrication

---

## Recommended Next Steps

1. **Immediate:** Test Solution 1 (Pure Varentropy) to establish baseline
2. **Short-term:** Implement Solution 3 (Max Aggregation) to capture bursts
3. **Medium-term:** Develop Solution 4 (Conditional Architecture) for robustness
4. **Validation:** Test on RAGTruth and FAVA to ensure generalization

---

## Appendix: Raw Data

### Per-Sample Scores (First 20)

| # | Label | Authority | Varentropy | Dispersion | LogProb |
|---|-------|-----------|------------|------------|---------|
| 1 | HALL | 0.740 | 1.110 | 0.100 | -1.55 |
| 2 | FACT | 0.759 | 0.139 | 0.029 | -0.24 |
| 3 | HALL | 0.747 | 1.272 | 0.125 | -2.20 |
| 4 | HALL | 0.768 | 0.316 | 0.027 | -0.57 |
| 5 | FACT | 0.810 | 0.542 | 0.136 | -10.56 |
| 6 | FACT | 0.807 | 0.312 | 0.068 | -1.89 |
| 7 | FACT | 0.767 | 0.253 | 0.001 | -0.04 |
| 8 | HALL | 0.755 | 1.516 | 0.185 | -1.07 |
| 9 | FACT | 0.785 | 0.492 | 0.071 | -5.65 |
| 10 | FACT | 0.804 | 0.162 | 0.015 | -0.53 |
