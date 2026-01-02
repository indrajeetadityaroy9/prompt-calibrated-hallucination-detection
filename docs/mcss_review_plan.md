# MC-SS Implementation Review Plan

## Overview

This document provides a structured review plan for the Manifold-Consistent Spectral Surprisal (MC-SS) implementation in AG-SAR. The review is organized into 6 stages, each examining specific components for correctness, robustness, and compatibility.

**Total Scope**: ~2,322 lines across 6 files

---

## Stage 1: Configuration Layer Review

**File**: `ag_sar/config.py` (Lines 82-145)

### 1.1 Parameter Definition Audit

| Parameter | Type | Default | Range | Verify |
|-----------|------|---------|-------|--------|
| `uncertainty_metric` | str | "gse" | {"gse", "gss", "mcss"} | [ ] Valid enum values |
| `mcss_beta` | float | 5.0 | (0, ∞) | [ ] Positive constraint |
| `mcss_hebbian_tau` | float | 0.1 | [0, 1] | [ ] Normalized range |
| `mcss_penalty_weight` | float | 1.0 | [0, ∞) | [ ] Non-negative |

### 1.2 Validation Logic (Lines 131-145)

**Check Items**:
- [ ] `uncertainty_metric` validation rejects invalid values
- [ ] `mcss_beta <= 0` raises ValueError
- [ ] `mcss_hebbian_tau` outside [0, 1] raises ValueError
- [ ] `mcss_penalty_weight < 0` raises ValueError
- [ ] Error messages are descriptive and actionable

**Test Command**:
```python
# Test invalid configs
AGSARConfig(uncertainty_metric="invalid")  # Should raise
AGSARConfig(mcss_beta=-1.0)  # Should raise
AGSARConfig(mcss_hebbian_tau=1.5)  # Should raise
AGSARConfig(mcss_penalty_weight=-0.1)  # Should raise
```

### 1.3 Serialization Compatibility (Lines 147-192)

**Check Items**:
- [ ] `to_dict()` includes all 4 MC-SS parameters
- [ ] `from_dict()` correctly deserializes MC-SS parameters
- [ ] Legacy configs without MC-SS params load with defaults
- [ ] Round-trip: `from_dict(config.to_dict()) == config`

**Potential Issues**:
- [ ] Missing `"gss"` in uncertainty_metric enum (referenced in config but implementation unclear)
- [ ] `from_dict()` doesn't explicitly handle missing MC-SS keys (relies on defaults)

### 1.4 Interface Contract

**Downstream Dependencies**:
- `ag_sar.py`: Reads `uncertainty_metric`, `mcss_*` params in `compute_uncertainty()`
- `exp13_mcss_ablation.py`: Creates configs programmatically

**Check Items**:
- [ ] All downstream code accesses config via `self.config.<param>`
- [ ] No hardcoded defaults bypassing config

---

## Stage 2: Core Computation Functions

### 2.1 Bounded Surprisal (`uncertainty.py` Lines 219-257)

**Function**: `compute_bounded_surprisal(logits, input_ids, beta, attention_mask)`

**Mathematical Specification**:
```
S_bounded(x_t) = tanh(-log P(x_t | x_{<t}) / β)
```

**Check Items**:
- [ ] Output range is strictly [0, 1) (tanh never reaches 1)
- [ ] `beta` division happens BEFORE tanh (not after)
- [ ] Shift alignment: Uses `compute_token_surprisal()` which handles logit shifting
- [ ] Masked positions return exactly 0.0 (not small epsilon)

**Edge Cases**:
- [ ] `beta → 0`: Should saturate to ~1.0 for all non-zero surprisal
- [ ] `beta → ∞`: Should give near-linear surprisal (S/β → 0)
- [ ] `P(x_t) = 1.0`: S = 0, bounded = 0
- [ ] `P(x_t) → 0`: S → ∞, bounded → 1.0

**Numerical Stability**:
- [ ] No log(0) risk (softmax guarantees P > 0)
- [ ] No overflow in -log(P) (clamped by softmax numerical limits)

### 2.2 Hebbian Weights (`centrality.py` Lines 159-245)

**Function**: `compute_hebbian_weights(K_stack, prompt_end_idx, tau, attention_mask)`

**Mathematical Specification**:
```
1. K_consensus = mean(K_stack, dim=heads)  # (B, S, D)
2. μ = mean(K_consensus[:, :prompt_end], dim=seq)  # (B, 1, D)
3. sim_t = cos_sim(K_t, μ)  # (B, S)
4. h_t = ReLU(sim_t - τ)
5. h_norm = h / max(h)  # (B, S) in [0, 1]
```

**Check Items**:
- [ ] Consensus embedding averages across dimension 1 (heads)
- [ ] Centroid uses ONLY prompt tokens (`:prompt_end_idx`)
- [ ] Cosine similarity uses F.normalize with p=2, dim=-1
- [ ] ReLU threshold subtracts tau before clamping
- [ ] Max-normalization uses `max(dim=-1, keepdim=True)` + epsilon
- [ ] Attention mask applied AFTER max-norm (masks padded tokens)

**Edge Cases**:
- [ ] `prompt_end_idx = 0`: Empty prompt → centroid is zero vector → undefined behavior
- [ ] `prompt_end_idx >= seq_len`: All tokens are prompt → response has no Hebbian weights
- [ ] `tau = 1.0`: All similarities < 1.0 → all weights = 0 after ReLU
- [ ] All tokens identical: All similarities = 1.0, all weights = 1.0

**Potential Issues**:
- [ ] **CRITICAL**: If `prompt_end_idx = 0`, `K_prompt` is empty tensor → mean returns NaN
- [ ] Missing validation that `prompt_end_idx > 0`

### 2.3 Hebbian Power Iteration (`centrality.py` Lines 248-322)

**Function**: `matrix_free_power_iteration_hebbian(Q_stack, K_stack, hebbian_weights, ...)`

**Mathematical Specification**:
```
v^(0) = uniform initialization
v^(k+1) = normalize(A · (v^(k) ⊙ h_hebbian))
```

**Check Items**:
- [ ] Hebbian modulation applied BEFORE matrix multiply (not after)
- [ ] L1 normalization after each iteration (for stability)
- [ ] Uses Triton kernel for A·v computation (O(N) memory)
- [ ] Convergence check uses tolerance parameter

**Interface Match**:
- [ ] Input shapes match: Q_stack (B, H, S, D), K_stack (B, H, S, D), hebbian_weights (B, S)
- [ ] Output shape: (B, S) L1-normalized centrality

### 2.4 MC-SS Main Function (`uncertainty.py` Lines 260-326)

**Function**: `compute_manifold_consistent_spectral_surprisal(bounded_surprisal, centrality, attention_mask, penalty_weight)`

**Mathematical Specification**:
```
1. v_masked = centrality * mask
2. v_norm = v_masked / max(v_masked)  # MAX-norm, not L1
3. penalty = 1 - v_norm  # Inverted: low centrality → high penalty
4. score_t = S_bounded + λ * penalty  # ADDITIVE
5. MC-SS = mean(score_t * mask) / sum(mask)
```

**Check Items**:
- [ ] Uses MAX-normalization (not L1) - Line 306-307
- [ ] Additive formulation: `bounded + penalty_weight * penalty` - Line 317
- [ ] Mask applied before max computation (prevents max in padding)
- [ ] Division by mask sum uses clamp(min=1) to avoid div-by-zero
- [ ] Epsilon in max-norm denominator: `+ 1e-10`

**Edge Cases**:
- [ ] All centrality = 0: Max-norm gives 0/epsilon → all zeros → penalty = 1
- [ ] Uniform centrality: All tokens get penalty = 0 (all are "max")
- [ ] Single unmasked token: That token IS the max → penalty = 0

**Critical Verification**:
- [ ] **Confident Lie Detection**: When S ≈ 0 but centrality ≈ 0:
  - Additive: 0 + λ*1 = λ (HIGH)
  - Multiplicative: 0 * (1+λ) = 0 (FAILS)
- [ ] Verify multiplicative is NOT used anywhere in final implementation

---

## Stage 3: Pipeline Integration

### 3.1 AGSAR.compute_uncertainty() (`ag_sar.py` Lines 176-366)

**Integration Flow**:
```
1. Check uncertainty_metric config (Line 264)
2. If "mcss": compute_hebbian_weights(K_stack, response_start) (Lines 269-275)
3. Pass hebbian_weights to compute_sink_aware_centrality() (Lines 283-292)
4. If "mcss": compute_bounded_surprisal() + compute_mcss() (Lines 312-327)
5. Return uncertainty score (Line 366)
```

**Check Items**:
- [ ] `response_start` correctly passed as `prompt_end_idx` to Hebbian function
- [ ] `attention_mask` correctly passed through all functions
- [ ] `return_details=True` includes MC-SS specific fields (bounded_surprisal, hebbian_weights)
- [ ] Legacy `gse` key maintained for backwards compatibility

### 3.2 Interface Contracts

**compute_hebbian_weights() → compute_sink_aware_centrality()**:
- [ ] `hebbian_weights` shape (B, S) matches expectation
- [ ] `use_hebbian=True` triggers Hebbian path in centrality computation
- [ ] Fallback when `use_hebbian=False`: standard power iteration used

**compute_bounded_surprisal() → compute_mcss()**:
- [ ] `bounded_surprisal` shape (B, S) matches
- [ ] `centrality` (actually `relevance`) shape (B, S) matches
- [ ] Both use same `response_mask` for consistency

### 3.3 Configuration Propagation

**Check Items**:
- [ ] `self.config.mcss_hebbian_tau` → `compute_hebbian_weights(tau=...)`
- [ ] `self.config.mcss_beta` → `compute_bounded_surprisal(beta=...)`
- [ ] `self.config.mcss_penalty_weight` → `compute_mcss(penalty_weight=...)`

### 3.4 Error Handling

**Check Items**:
- [ ] Empty response handling returns 0.0 (Lines 211-232)
- [ ] Response start >= seq_len returns 0.0 (Lines 234-248)
- [ ] No unhandled exceptions in MC-SS path

---

## Stage 4: compute_sink_aware_centrality() Integration

### 4.1 Parameter Additions (`centrality.py` Lines 325-443)

**New Parameters**:
```python
hebbian_weights: Optional[torch.Tensor] = None,
use_hebbian: bool = False,
```

**Check Items**:
- [ ] Parameters are optional with sensible defaults
- [ ] `use_hebbian=True` without `hebbian_weights` → should handle gracefully
- [ ] `hebbian_weights` provided but `use_hebbian=False` → weights ignored

### 4.2 Routing Logic (Lines 384-396)

```python
if use_hebbian and hebbian_weights is not None:
    centrality = matrix_free_power_iteration_hebbian(...)
else:
    centrality = matrix_free_power_iteration(...)
```

**Check Items**:
- [ ] Both paths return same shape (B, S)
- [ ] Both paths use same normalization (L1)
- [ ] Residual weight applied consistently in both paths

### 4.3 Return Value Compatibility

**Check Items**:
- [ ] Return tuple structure unchanged: `(relevance, centrality, per_head_contrib)`
- [ ] `relevance` computation (centrality × value_norms) still applied after Hebbian path
- [ ] `per_head_contrib` returned correctly when `return_raw=True`

---

## Stage 5: Test Coverage Analysis

### 5.1 Test Matrix (`tests/test_mcss.py`)

| Function | Shape | Range | Edge Cases | Integration |
|----------|-------|-------|------------|-------------|
| `compute_bounded_surprisal` | ✓ | ✓ | ✓ | - |
| `compute_hebbian_weights` | ✓ | ✓ | Partial | - |
| `compute_mcss` | ✓ | ✓ | ✓ | - |
| `AGSAR.compute_uncertainty (mcss)` | - | - | - | Partial |

### 5.2 Missing Test Coverage

**Critical Gaps**:
- [ ] No test for `prompt_end_idx = 0` edge case in Hebbian weights
- [ ] No test for very long sequences (memory/numerical stability)
- [ ] No end-to-end test with actual GPT-2 model and MC-SS config
- [ ] No test for `uncertainty_metric = "gss"` (undefined path)

**Integration Gaps**:
- [ ] No test verifying `ag_sar.compute_uncertainty(config=AGSARConfig(uncertainty_metric="mcss"))`
- [ ] No test for `return_details=True` with MC-SS (checking new fields)

### 5.3 Test Robustness

**Check Items**:
- [ ] Tests use deterministic random seeds
- [ ] Tests don't rely on GPU (or skip appropriately)
- [ ] Tests validate numerical precision (atol/rtol appropriate)

---

## Stage 6: Experiment Validation

### 6.1 exp13_mcss_ablation.py Structure

**Check Items**:
- [ ] All hyperparameter sweep values covered: τ, β, λ
- [ ] Ablation configs correctly disable components
- [ ] `compute_mcss_score()` correctly handles post-hoc ablations

### 6.2 Ablation Implementation Correctness

**Multiplicative Ablation (Lines 267-273)**:
```python
if mcss_config.use_additive:
    score_token = bounded + (penalty_weight * penalty)
else:
    score_token = bounded * (1.0 + penalty_weight * penalty)
```

**Check Items**:
- [ ] Multiplicative formula is mathematically correct alternative
- [ ] `1.0 + penalty_weight * penalty` ensures non-zero base (but may not match paper)
- [ ] **Potential Issue**: Should multiplicative be `bounded * penalty` not `bounded * (1 + λ*penalty)`?

### 6.3 Evaluation Metrics

**Check Items**:
- [ ] AUROC computed correctly (higher uncertainty = hallucination)
- [ ] Ground truth labels from ROUGE threshold are consistent
- [ ] Results JSON includes all hyperparameters for reproducibility

---

## Stage 7: Integration Gap Analysis

### 7.1 Identified Gaps

| ID | Location | Issue | Severity | Resolution |
|----|----------|-------|----------|------------|
| G1 | config.py | `"gss"` metric undefined | Medium | Remove or implement |
| G2 | centrality.py:159 | `prompt_end_idx=0` causes NaN | High | Add validation |
| G3 | ag_sar.py | No test for MC-SS end-to-end | Medium | Add integration test |
| G4 | test_mcss.py:245 | `pass` in test body | Low | Documented intentional |
| G5 | exp13 | Multiplicative formula differs from paper | Medium | Verify against spec |

### 7.2 Interface Mismatches

| Interface | Expected | Actual | Issue |
|-----------|----------|--------|-------|
| `compute_hebbian_weights.K_stack` | (B, H, S, D) | (B, L*H, S, D) | Dimension naming inconsistency |
| `compute_mcss.centrality` | Hebbian-filtered | Could be `relevance` | Naming confusion |

### 7.3 Implementation Deviations from Plan

| Plan Spec | Implementation | Deviation |
|-----------|----------------|-----------|
| "Max-normalized centrality for scoring" | MAX-norm applied in `compute_mcss()` | ✓ Correct |
| "Consensus embedding first" | `K_stack.mean(dim=1)` in Hebbian | ✓ Correct |
| "Additive: S + λ(1-v)" | `bounded + penalty_weight * penalty` | ✓ Correct |
| "Prompt-only centroid" | `K[:, :prompt_end_idx, :]` | ✓ Correct |

---

## Review Execution Checklist

### Pre-Review Setup
- [ ] Checkout clean branch with MC-SS implementation
- [ ] Ensure all dependencies installed: `pip install -e ".[dev]"`
- [ ] Run existing tests to establish baseline: `pytest tests/ -v`

### Stage Execution Order
1. [ ] **Stage 1**: Config layer (30 min)
2. [ ] **Stage 2**: Core functions (90 min)
3. [ ] **Stage 3**: Pipeline integration (45 min)
4. [ ] **Stage 4**: Centrality integration (30 min)
5. [ ] **Stage 5**: Test coverage (45 min)
6. [ ] **Stage 6**: Experiment validation (30 min)
7. [ ] **Stage 7**: Gap analysis (45 min)

### Post-Review Actions
- [ ] Document all identified issues in GitHub issues
- [ ] Create fix PRs for High severity gaps
- [ ] Update tests to cover identified gaps
- [ ] Re-run full test suite after fixes

---

## Appendix: Quick Reference

### File Locations
```
ag_sar/config.py          # Lines 82-145: MC-SS config
ag_sar/uncertainty.py     # Lines 219-326: bounded surprisal + MC-SS
ag_sar/centrality.py      # Lines 159-322: Hebbian weights + power iteration
ag_sar/ag_sar.py          # Lines 264-327: Pipeline integration
tests/test_mcss.py        # Full file: Test suite
eval/experiments/exp13_mcss_ablation.py  # Full file: Ablation study
```

### Key Formulas
```
Bounded Surprisal:  S_b = tanh(-log P / β)
Hebbian Weights:    h = ReLU(cos_sim(K, μ_prompt) - τ) / max(h)
MC-SS (Additive):   score = mean(S_b + λ(1 - h_norm))
```

### Config Defaults
```python
uncertainty_metric = "gse"
mcss_beta = 5.0
mcss_hebbian_tau = 0.1
mcss_penalty_weight = 1.0
```
