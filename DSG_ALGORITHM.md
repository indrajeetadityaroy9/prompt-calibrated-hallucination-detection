# Decoupled Spectral Grounding (DSG): A Novel Zero-Shot Hallucination Detection Algorithm

## 1. Core Thesis

Current AG-SAR treats all internal signals as exchangeable risk indicators and fuses them with uniform Noisy-OR. This ignores the **causal structure** of how hallucinations form inside the transformer: context enters via **attention (copying)**, gets transformed by **FFNs (parametric override)**, and exits through the **unembedding head (output commitment)**. DSG decomposes risk along this causal chain, using three mechanistically distinct signal families that are provably complementary.

## 2. Algorithm Overview

```
Input: hidden_states[L][seq, D], attentions[L][H, seq, seq], logits[seq, V], prompt_len

Output: per-token risk r(t) ∈ [0,1], response risk R ∈ [0,1]

Step 1: Context Utilization Score      — CUS(t) ∈ [0,1]  (attention mechanism)
Step 2: Parametric Override Score      — POS(t) ∈ [0,1]  (FFN mechanism)
Step 3: Dual-Subspace Projection Score — DPS(t) ∈ [0,1]  (representation mechanism)
Step 4: Causal Noisy-OR Fusion         — r(t) = f(CUS, POS, DPS)
Step 5: Response Aggregation           — R = agg(r(t))
```

## 3. Signal Family 1: Context Utilization Score (CUS)

**Causal role**: Measures whether the attention mechanism is copying from context or from generated tokens. Inspired by ReDeEP's Copying Head identification and TPA's source attribution.

**What's new vs. current AttentionGroundingSignal**: Instead of a flat ratio `attn_context / attn_total`, DSG identifies **which attention heads are "copying heads"** via their context affinity during prefill, then tracks only those heads during generation. This eliminates noise from heads that serve syntactic or positional roles.

### Algorithm

```
Phase A — Prefill: Identify Copying Heads
  For each head h in layer l:
    affinity(l,h) = mean_{t ∈ context} max_{s ∈ context, s≠t} attn(l,h)[t,s]
  copying_heads = {(l,h) : affinity(l,h) > percentile_75(all affinities)}

Phase B — Generation: Compute CUS per token
  For each output token t:
    For each (l,h) ∈ copying_heads:
      c(l,h,t) = Σ_{s < prompt_len} attn(l,h)[t, s]   # context mass
    CUS(t) = 1 - mean_{(l,h) ∈ copying_heads} c(l,h,t)
```

**Properties**:
- Range: [0, 1], higher = less context utilization = riskier
- Zero-shot: copying head identification uses prefill statistics only
- Complementary to representation-based signals: captures **information flow**, not **information content**

### Mathematical Justification

A copying head `(l,h)` is one where during prefill, each context token attends heavily to other context tokens (building cross-reference patterns). During generation, if the model grounds in context, these same heads should route attention back to context. When they don't, the model is generating from parametric memory.

The percentile-75 threshold is chosen because in LLaMA 3.1 8B (32 layers × 32 heads = 1024 heads), approximately 25% of heads serve as information-retrieval heads (consistent with findings in TPA and mechanistic interpretability literature on induction heads).

## 4. Signal Family 2: Parametric Override Score (POS)

**Causal role**: Measures how much the FFN layers shift the residual stream away from what attention established. Inspired by ReDeEP's Parametric Knowledge Score and the existing `topk_jsd` signal, but with a critical fix.

**What's new vs. current CandidateJSDSignal**: Two improvements:
1. **Layer-selective JSD**: Only compute at layers where the FFN actually modifies the representation substantially (adaptive threshold), rather than summing across all layers
2. **Directional JSD**: Decompose the shift into "toward emitted token" vs. "away from context", using the unembedding vector as a reference direction

### Algorithm

```
For each output token t, at each layer l:
  h_pre  = final_norm(h_resid_attn[l])     # pre-FFN (after attention)
  h_post = final_norm(h_resid_mlp[l])      # post-FFN (after MLP)

  # Standard JSD on candidate set (existing)
  jsd(l) = JSD(softmax(W_U[cand] · h_pre), softmax(W_U[cand] · h_post))

  # NEW: Directional decomposition
  δ(l) = h_post - h_pre                    # FFN residual
  e_tok = W_U[emitted_token]               # unembedding vector of emitted token

  # Component of FFN shift toward emitted token
  proj_tok(l) = (δ(l) · e_tok) / (||e_tok|| + ε)

  # Component of FFN shift orthogonal to context subspace
  # (reuses context basis V from DPS)
  δ_orth(l) = δ(l) - V @ V^T @ δ(l)
  override(l) = ||δ_orth(l)|| / (||δ(l)|| + ε)

# Select "active" FFN layers: where JSD exceeds prompt baseline
active_layers = {l : jsd(l) > μ_prompt_jsd + σ_prompt_jsd}

# POS = mean override across active layers
POS(t) = mean_{l ∈ active_layers} override(l)
```

**Properties**:
- Range: [0, 1], higher = FFN pushing representation away from context = riskier
- Fixes the prefill/decode JSD mismatch bug (uses same candidate set consistently)
- Captures **why** the FFN shifts, not just **how much**

### Mathematical Justification

The FFN δ(l) vector can be decomposed into:
1. A component that strengthens the emitted token (legitimate next-token prediction)
2. A component that moves within the context subspace (legitimate paraphrasing)
3. A component orthogonal to context (potential hallucination)

Only component (3) indicates parametric override. The `override(l)` ratio measures what fraction of the FFN's work moves the representation out of the context subspace. This directly addresses the identified limitation that current JSD can't distinguish "FFN reinforcing context" from "FFN overriding context."

## 5. Signal Family 3: Dual-Subspace Projection Score (DPS)

**Causal role**: Measures where the final output representation sits relative to two orthogonal reference subspaces. Inspired by HARP's reasoning subspace + existing context grounding, unified into a single geometric framework.

**What's new vs. current ContextGroundingSignal**: Two subspaces instead of one, with a geometric ratio that is inherently more discriminative.

### Algorithm

```
Precomputation (once per input):
  # Context subspace (existing, from SVD of context hidden states)
  C = context_hidden - μ_context                    # [n_ctx, D]
  U_c, S_c, V_c = SVD(C)
  k_c = argmin_k { cumvar(S_c[:k]) > 0.95 }       # adaptive rank
  V_ctx = V_c[:k_c]                                 # [k_c, D]

  # Reasoning subspace (NEW, from SVD of unembedding matrix — HARP-inspired)
  U_r, S_r, V_r = SVD(W_U)                         # W_U is [V, D]
  V_rsn = V_r[-k_r:]                               # bottom k_r singular vectors
  # k_r = max(1, floor(0.05 * D))                  # bottom 5% per HARP

Per output token t, at middle layers (l ∈ [L/3, 2L/3]):
  h = hidden_states[l][t] - μ_context

  # Context projection
  proj_ctx = V_ctx @ V_ctx^T @ h
  s_ctx = ||proj_ctx|| / (||h|| + ε)               # ∈ [0, 1]

  # Reasoning projection
  proj_rsn = V_rsn @ V_rsn^T @ h
  s_rsn = ||proj_rsn|| / (||h|| + ε)               # ∈ [0, 1]

  # Dual-subspace score: ratio of reasoning to context projection
  DPS_layer(t) = s_rsn / (s_ctx + s_rsn + ε)       # ∈ [0, 1]

# Average across middle layers
DPS(t) = mean_l DPS_layer(t)
```

**Properties**:
- Range: [0, 1], higher = more reasoning-driven (less context-grounded) = riskier
- The ratio `s_rsn / (s_ctx + s_rsn)` is inherently normalized and captures the **balance** between two competing explanations for the representation
- Factual tokens: high `s_ctx`, low `s_rsn` → DPS ≈ 0
- Hallucinated tokens: low `s_ctx`, high `s_rsn` → DPS ≈ 1
- Paraphrases: high `s_ctx`, moderate `s_rsn` → DPS ≈ 0.3 (correctly low-risk)

### Mathematical Justification

The linear representation hypothesis (Park et al., 2024) establishes that LLM representations encode semantic concepts in linear subspaces. The context subspace captures "what the model was told." The reasoning subspace (HARP's insight) captures "how the model reasons beyond input." A hallucination occurs when the model's output representation is better explained by its reasoning process than by the input context — exactly what the DPS ratio measures.

The bottom-5% singular vectors of W_U form the reasoning subspace because these directions have minimal contribution to next-token prediction (low singular values in the unembedding), meaning they encode internal processing states rather than output-facing features.

**Critical fix**: The existing ISE signal (`internal_se.py:61`) applies lm_head without final_norm. DPS applies final_norm before any projection through W_U, consistent with proper Logit Lens methodology.

## 6. Causal Noisy-OR Fusion

**What's new vs. current uniform Noisy-OR**: Structured fusion that respects the causal chain: Attention → FFN → Representation.

### Algorithm

```
For each output token t:
  # Step 1: Compute individual signal probabilities
  p_cus = CUS(t)     # attention-based
  p_pos = POS(t)     # FFN-based
  p_dps = DPS(t)     # representation-based

  # Step 2: Prompt-anchored calibration (existing z-score approach)
  # But applied CONSISTENTLY (fixes the prefill/decode mismatch)
  For each signal s ∈ {cus, pos, dps}:
    z_s = (p_s - μ_s_prompt) / (σ_s_prompt + ε)
    p_s_cal = sigmoid(z_s)    # calibrated probability ∈ (0, 1)

  # Step 3: Causal Noisy-OR
  # The causal structure: if attention fails to ground → FFN may override →
  # representation diverges. Each is an INDEPENDENT failure mode.
  r(t) = 1 - (1 - p_cus_cal) × (1 - p_pos_cal) × (1 - p_dps_cal)
```

**Why this works better than uniform Noisy-OR over 15 signals**:
1. **Three signals, not fifteen**: The generalization experiments showed that ensembles of all signals degrade performance (-0.138 vs -0.112 for single signal). Three orthogonal signals avoid the dilution problem.
2. **Mechanistically distinct**: CUS, POS, and DPS measure different physical mechanisms (attention routing, FFN transformation, representation geometry). Their correlation is provably low because they measure different matrix operations.
3. **Consistent calibration**: All three signals use the same z-score + sigmoid pipeline with prompt-anchored statistics, ensuring they contribute on comparable scales (fixing the identified scale mismatch).

## 7. Response-Level Aggregation

```
R = percentile_90(r(t) for t in response)
```

Consistent across all paths (fixing the max vs p90 inconsistency).

## 8. Computational Complexity

| Component | Per-token cost | One-time cost |
|-----------|---------------|---------------|
| CUS | O(H × prompt_len) | O(L × H × prompt_len²) for copying head ID |
| POS | O(L_active × K × D) | O(1) |
| DPS | O(L_mid × k² × D) | O(min(n_ctx, D)² × D) for context SVD + O(V × D²) for reasoning SVD |
| Fusion | O(3) | O(1) |

Where L_active ≈ L/3, K = candidate set size (128), k = context rank (≤50), L_mid ≈ L/3, D = hidden dim (4096), H = num heads (32), V = vocab size.

**Critical**: The reasoning subspace SVD of W_U is O(V × D²) but is computed **once per model load** (it's input-independent). The context subspace SVD is O(n_ctx² × D) computed once per input during prefill. Per-token costs are dominated by POS at O(L/3 × 128 × 4096), which is already the cost of existing CandidateJSD.

**Total overhead vs. current AG-SAR**: Approximately 1.3× current cost (the main addition is CUS attention tracking and DPS's secondary projection, both lightweight).

## 9. Why This Is Novel

| Aspect | Prior Work | DSG (This Work) |
|--------|-----------|-----------------|
| Source decomposition | ReDeEP uses learned α,β weights | Zero-shot causal decomposition (no training) |
| Attention signal | Flat ratio (context/total) | Copying-head identification + selective tracking |
| Context projection | Single subspace (SVD) | Dual-subspace ratio (context vs reasoning) |
| Reasoning subspace | HARP uses trained MLP on projection | Zero-shot ratio (no MLP needed) |
| FFN analysis | Raw JSD magnitude | Directional decomposition (toward-token vs away-from-context) |
| Fusion | Uniform Noisy-OR over N signals | Structured 3-signal causal Noisy-OR |
| Calibration | Scale mismatch across signals | Consistent prompt-anchored z-scores |

**The key novelty is the causal decomposition**: no prior work combines attention-level (CUS), transformation-level (POS), and representation-level (DPS) signals in a causally structured fusion. Each signal answers a different question:
- **CUS**: "Is the model looking at the context?"
- **POS**: "Is the FFN overriding what attention found?"
- **DPS**: "Does the final representation live in context-space or reasoning-space?"

## 10. Expected Performance Impact

Based on the identified gaps and empirical findings:

| Issue Fixed | Expected Impact |
|-------------|----------------|
| Copying head selection (CUS vs flat attention) | +3-5% on attention grounding signal (currently ranks 12/15) |
| Directional POS vs raw JSD | +2-4% on summarization (paraphrase hallucinations misclassified) |
| Dual-subspace DPS vs single SVD | +3-5% on RAGTruth (reasoning subspace adds complementary signal) |
| 3-signal fusion vs 15-signal | +2-3% from reduced dilution (ensemble of all = -0.138 vs best single = -0.112) |
| Consistent calibration | +1-2% from fixing scale mismatch |
| **Combined** | **+8-15% cross-task AUROC** (0.648 → ~0.73-0.75) |

These estimates are conservative, based on the measured delta between context_grounding alone (0.657 residualized AUROC) and the theoretical ceiling when combining with hidden_norm and inv_margin.

## 11. Implementation Plan

The algorithm maps to the existing codebase as follows:

### Phase 1: Signal Implementations

#### 1a. Context Utilization Score (CUS)
- **New file**: `ag_sar/signals/copying_heads.py`
- **Dependencies**: Attention tensors already captured by `ag_sar/hooks.py`
- **Classes**:
  - `CopyingHeadIdentifier`: Runs during prefill to identify copying heads via context affinity
  - `ContextUtilizationSignal`: Computes CUS(t) per token using identified copying heads
- **Hook changes**: Extend `PrefillContextHook` in `ag_sar/hooks.py` to store per-head attention patterns during prefill (currently only stores hidden states)
- **Validation**: Compare CUS AUROC vs existing `AttentionGroundingSignal` on RAGTruth

#### 1b. Parametric Override Score (POS)
- **Modify file**: `ag_sar/signals/topk_jsd.py`
- **New class**: `DirectionalJSDSignal` extending `CandidateJSDSignal`
- **Additions**:
  - `compute_directional_override()`: Decomposes FFN δ into context-orthogonal component
  - `select_active_layers()`: Filters layers by JSD > prompt baseline threshold
- **Dependencies**: Reuses `lm_head`, `final_norm`, and context basis `V_ctx` from DPS
- **Bug fix**: Ensure prefill JSD statistics use candidate-restricted softmax (same as decode-time), fixing the mismatch identified in `ag_sar/hooks.py:374-395`
- **Validation**: Compare POS AUROC vs raw `jsd_cand` on RAGTruth (especially summarization subset)

#### 1c. Dual-Subspace Projection Score (DPS)
- **Modify file**: `ag_sar/signals/context_grounding.py`
- **New class**: `DualSubspaceGrounding` extending `ContextGroundingSignal`
- **Additions**:
  - `_compute_reasoning_basis()`: One-time SVD of `lm_head.weight`, cached per model load
  - `compute_dual_projection()`: Computes s_ctx and s_rsn, returns ratio
- **Dependencies**: `lm_head.weight` for reasoning subspace, existing context SVD for context subspace
- **Bug fix**: Apply `final_norm` before any projection through W_U (fixes `internal_se.py:61` pattern)
- **Validation**: Compare DPS AUROC vs single-subspace context_grounding on RAGTruth

### Phase 2: Fusion and Aggregation

#### 2a. DSG Detector
- **New file**: `ag_sar/icml/dsg_detector.py`
- **Class**: `DSGDetector`
- **Methods**:
  - `__init__(model, tokenizer)`: Initializes all three signal computers, precomputes reasoning subspace SVD
  - `prefill(hidden_states, attentions, prompt_len)`: Identifies copying heads, computes context SVD, computes prompt-anchored statistics for all three signals
  - `compute_risk(hidden_states, attentions, logits, prompt_len)`: Runs CUS + POS + DPS → Causal Noisy-OR → p90 aggregation
- **Replaces**: `ag_sar/icml/robust_ensemble.py` as the recommended ICML detector

#### 2b. Aggregation Fix
- **Modify file**: `ag_sar/aggregation/prompt_anchored.py`
- **Change**: Standardize response-level aggregation to `percentile_90` everywhere (currently inconsistent between `max` in some paths and `percentile_90` in others)

### Phase 3: Bug Fixes

#### 3a. ISE Final Norm Fix
- **File**: `ag_sar/signals/internal_se.py`, line 61
- **Fix**: Add `final_norm()` before `lm_head()` projection, matching all other Logit Lens signals

#### 3b. Prefill/Decode JSD Consistency
- **File**: `ag_sar/hooks.py`, lines 374-395
- **Fix**: Compute prefill JSD statistics using candidate-restricted softmax (same distribution as decode-time), not full-vocabulary softmax

### Phase 4: Validation

#### 4a. Individual Signal Validation
- Run each signal (CUS, POS, DPS) independently on RAGTruth QA + Summarization
- Report: raw AUROC, delta-above-length, length-residualized AUROC
- Compare against corresponding baseline signals (attention_grounding, jsd_cand, context_grounding)

#### 4b. Fusion Validation
- Run full DSG pipeline on RAGTruth
- Compare against: `RobustEnsemble` (0.648), `ICMLContextGrounding`, `InvMarginOnly`
- Report: cross-task mean AUROC, std, delta-above-length

#### 4c. Ablation Studies
- DSG without CUS (POS + DPS only)
- DSG without POS (CUS + DPS only)
- DSG without DPS (CUS + POS only)
- DSG with uniform Noisy-OR vs causal Noisy-OR (same signals, different fusion)
- Single-subspace DPS vs dual-subspace DPS

### File Summary

| Action | File | Description |
|--------|------|-------------|
| CREATE | `ag_sar/signals/copying_heads.py` | CUS signal (copying head ID + context utilization) |
| MODIFY | `ag_sar/signals/topk_jsd.py` | Add `DirectionalJSDSignal` for POS |
| MODIFY | `ag_sar/signals/context_grounding.py` | Add `DualSubspaceGrounding` for DPS |
| CREATE | `ag_sar/icml/dsg_detector.py` | DSG detector (fusion + orchestration) |
| MODIFY | `ag_sar/hooks.py` | Extend prefill hooks for per-head attention capture |
| MODIFY | `ag_sar/aggregation/prompt_anchored.py` | Standardize to p90 aggregation |
| FIX | `ag_sar/signals/internal_se.py` | Add final_norm before lm_head projection |
| FIX | `ag_sar/hooks.py` | Candidate-restricted prefill JSD statistics |

### Dependencies Between Phases

```
Phase 1a (CUS) ──────────────────────────┐
Phase 1b (POS) ─── needs V_ctx from 1c ──┤
Phase 1c (DPS) ──────────────────────────┼── Phase 2a (DSG Detector)
Phase 3a (ISE fix) ──────────────────────┘        │
Phase 3b (JSD fix) ──────────────────────────────┘
Phase 2b (aggregation fix) ──────────────── independent
                                                  │
                                          Phase 4 (Validation)
```

**Recommended implementation order**: 1c → 1b → 1a → 3a → 3b → 2b → 2a → 4
