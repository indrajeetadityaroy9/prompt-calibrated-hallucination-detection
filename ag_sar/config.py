"""
AG-SAR Configuration - Minimal Dynamic Architecture.

Core mechanism parameters only. All thresholds are auto-calibrated from prompt statistics.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class AGSARConfig:
    """
    Minimal configuration for AG-SAR hallucination detection.

    Most parameters are auto-derived from prompt calibration.
    Only essential tuning knobs are exposed.

    v2 Core Equation:
        Uncertainty(t) = 1 - Authority(t)
        Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)
        Trust(t) = 1 - Dispersion(t) × (1 + λ × Varentropy(t))

    v3.4 Core Equation (Cognitive Load):
        Score(t) = A(t) × (1 - D_t) × E_t^λ              # Power law epistemic weight
        E_t = V_t / τ                                     # Cognitive load ratio (τ=5.0, calibration-free)
        A(t) = Σ_ctx α_tj + Σ_gen α_tj × A(j)            # Recursive authority
        (1 - D_t) = Semantic consistency
        E^λ = Epistemic weight (λ=4.0 default, clamped at 1.0)

    v4 Core Equation (Epistemic Dominance with Semantic Shielding):
        P_faith(t) = A(t) × (1 - P_v × P_d)             # Shielded penalty
        P_v = tanh(V_t / scale)                          # Varentropy risk
        P_d = sigmoid((D_t - τ) × stiffness)            # Dispersion shield (dynamic τ)
        Score_seq = min(P_faith(t))                      # Weakest link aggregation
        - Dynamic τ = μ_D(prompt) + σ_D(prompt), clamped [0.10, 0.40]

    v5 Core Equation (Dual-Path Aggregation):
        Score_final = Mean(A_t) × Min(T_t)              # Heterogeneous aggregation
        A_t = Authority Flow (provenance from context)
        T_t = 1 - (P_v × P_d) (trust from semantic shielding)
        - Mean(A_t): Global Grounding - "On average, did we rely on context?"
        - Min(T_t): Local Trust - "Did we fabricate anything?"
        - Correctly respects Topology of Error: Authority=global, Trust=local

    v6 Core Equation (Gaussian Complexity Matching):
        Score_final = Mean(A_t) × Min(T_t)              # Dual-path from v5
        T_t = (1 - D_t) × G_complexity(R_t, σ)          # Trust with complexity penalty
        G_complexity = exp(-(R_t - 1.0)² / 2σ²)         # Gaussian centered at R=1
        R_t = V_t / max(μ_prompt, ε)                    # Complexity ratio
        - Solves "Goldilocks Information Problem":
          * HaluEval: Hall V >> V_prompt (confusion) → R >> 1 → G → 0
          * RAGTruth: Hall V << V_prompt (oversimplification) → R << 1 → G → 0
          * Truth: Fact V ≈ V_prompt → R ≈ 1 → G ≈ 1

    v8 Core Equation (Residual Stream Contrast):
        Score(t) = (1 - FFN_Interference(t)) × (1 - D_t)
        FFN_Interference(t) = JSD(P_attn || P_final)     # Jensen-Shannon Divergence
        P_attn = softmax(h_attn @ lm_head.T)             # What context says
        P_final = softmax(h_block @ lm_head.T)           # What model outputs
        - Detects "Confident Lies" (CHOKE phenomenon):
          * FFN silently overrides context signal even with good attention
          * Measures OUTCOMES (logit divergence) not INPUTS (attention weights)

    v9 Core Equation (Holographic Dual-Stream):
        Score_seq = min(mean(Stability), mean(Grounding))  # Sequence-level MIN
        Stability(t) = (1 - V_norm) × (1 - D_t)            # Epistemic stability (v4)
        Grounding(t) = 1 - JSD(P_attn || P_final)          # Mechanistic grounding (v8)
        - Key insight: Aggregate BEFORE combining, not combine BEFORE aggregating
        - Each signal is aggregated with mean (optimal for its characteristics)
        - Then MIN takes the sequence-level worst signal
        - Unified detection of BOTH hallucination types:
          * HaluEval (Confusion): mean(Stability) is low → min catches it
          * RAGTruth (Deception): mean(Grounding) is low → min catches it
          * Fact: Both means high → min is high

    v10 Core Equation (Orthogonal Signal Fusion):
        Score_seq = S_seq × G_seq                              # Multiplicative fusion
        S_seq = SlidingWindowPercentile_10(Stability_t)        # Burst detection (HaluEval)
        G_seq = Mean(Grounding_t)                              # Systemic detection (RAGTruth)
        Stability(t) = (1 - V_norm) × (1 - D_t)                # Epistemic stability
        Grounding(t) = 1 - JSD(P_attn || P_final)              # Mechanistic grounding
        - Key insight: Decouple aggregation topologies - each signal uses optimal reduction
        - Stability → SlidingWindowPercentile_10 (catches phrase-level bursts, tolerates noise)
        - Grounding → Mean (captures systemic FFN override)
        - Solves v9's interference problem: signals don't wash each other out
        - Detection: HaluEval via low S_seq, RAGTruth via low G_seq

    v11 Core Equation (Information Physics - Energy-Based):
        E_t = JSD(P_attn || P_final) + tanh(V_t)               # Hallucination Energy
        Uncertainty = SoftMax_β(E_t)                           # LogSumExp aggregation
        - JSD ∈ [0, 1]: "Work Against Context" - external conflict (FFN override)
        - tanh(V) ∈ [0, 1]: "Internal Disorder" - epistemic confusion
        - E_t ∈ [0, 2]: Total energy (higher = more likely hallucination)
        - Key insight: Replace heuristics with first-principles thermodynamics
        - SoftMax aggregation smoothly interpolates mean↔max:
          * Burst (HaluEval): exp() amplifies spike → acts like Max
          * Systemic (RAGTruth): sum accumulates → acts like Mean
        - No thresholds, no calibration, no task-specific heuristics

    v12 Core Equation (Dual-Stream Risk):
        Risk_seq = max(R_internal, R_external)                # MAX fusion
        R_internal = Percentile_90(tanh(V/τ) × Shield(D))     # Confusion detector (HaluEval)
        R_external = Mean(JSD)                                 # Deception detector (RAGTruth)
        - Key insight: Hallucination detection is multi-objective
        - Internal risk: Percentile_90 catches burst confusion, ignores noise
        - External risk: Mean captures systemic FFN override
        - MAX: If either detector flags, flag the sequence (OR logic)
        - Detection: HaluEval via high R_internal, RAGTruth via high R_external

    v13 Core Equation (Adaptive Regime Switching - Hybrid):
        Score_seq = (1 - w) × S_v5 + w × G_v8            # Adaptive fusion
        w = sigmoid((Length - τ_len) × α)                # Regime weight
        S_v5 = Mean(Authority) × Min(Trust)              # v5's full formula
        G_v8 = Mean(1 - JSD(P_attn || P_final))          # v8's grounding
        - Key insight: Hallucination mechanism changes with generation length
        - Short (QA): Uses v5 (Authority+Trust) → catches confusion (w≈0)
        - Long (RAG): Uses v8 (JSD Grounding) → catches FFN override (w≈1)
        - τ_len=30 tokens: Transition center
        - α=0.2: Smooth sigmoid transition over [20, 40] range
        - Limitation: Length heuristic fails for "Long Confusion" (HaluEval Summ)

    v14 Core Equation (Conservative Min-Fusion - Safety Veto):
        Score_seq = min(S_seq, G_seq)                    # Logical AND fusion
        S_seq = Mean(Authority) × Min(Trust)             # v5's stability (confusion)
        G_seq = Mean(1 - JSD(P_attn || P_final))         # v8's grounding (deception)
        - Key insight: Faithful = Stable AND Grounded (both required)
        - No length heuristics - let signals speak for themselves
        - HaluEval (Confusion): Low S, High G → min=Low → CAUGHT
        - RAGTruth (Deception): High S, Low G → min=Low → CAUGHT
        - Facts: High S, High G → min=High → PASS
        - Cross-dataset SOTA: Robust to task type variation

    v15 Core Equation (Coherence-Interaction Model):
        Score = (1 - JSD_seq) × (1 - max_t(V×Sigmoid(D)))
        - Mechanistic Integrity: Mean(1 - JSD) catches FFN override (deception)
        - Epistemic Coherence: 1 - Percentile_90(V×D) catches confusion with shielding
        - Key innovation: V×Sigmoid(D) only penalizes when BOTH V and D are high
        - Semantic Shielding: High V with low D is protected (valid reasoning)
        - Solves v5's inversion: (1-V)×(1-D) wrongly penalized reasoning
        - FAILED: Dispersion is task-dependent, not label-dependent

    v16 Core Equation (Grounded-Risk Architecture):
        Risk_seq = max(R_deception, R_confusion)
        R_deception = Mean(JSD)                       # FFN override (deception)
        R_confusion = Percentile_90(V × (1 - A))      # Ungrounded uncertainty
        - Key innovation: Authority Shield replaces failed Dispersion Shield
        - Valid Summary: High V, High A → Low (1-A) → Low Risk (saved)
        - Confused Hall: High V, Low A → High (1-A) → High Risk (caught)
        - Robust to long-form generation: Authority measures actual provenance
        - FAILED: MAX fusion selects noisy JSD on HaluEval

    v17 Core Equation (Thermodynamic Gating - Maxwell's Demon):
        Risk_seq = gate × R_confusion + (1-gate) × R_deception
        gate = sigmoid((V_mean - τ) × α)              # Thermodynamic regime selector
        R_deception = Mean(JSD)                       # Trust in Deception Regime
        R_confusion = Percentile_90(V × (1 - A))      # Trust in Confusion Regime
        - Key innovation: Use ENTROPY to select which detector to trust
        - High V (Confusion Regime): Trust R_confusion (HaluEval)
        - Low V (Deception Regime): Trust R_deception (RAGTruth)
        - Replaces failed Length Heuristic (v13) with State-of-Matter Heuristic
        - FAILED: V varies by LABEL within datasets, not by TASK across datasets

    v19 Core Equation (Hinge-Risk Architecture - Zero-Shot SOTA):
        R_deception = Mean(JSD)                        # Systemic FFN override
        R_confusion = Max(V_norm × D)                  # Burst epistemic collapse
        Risk_seq = Max(R_deception, R_confusion)       # Either detector flags

        Key Innovation: Topology-Aware Aggregation
        - Deception (RAGTruth): Systemic pattern → Mean catches distributed override
        - Confusion (HaluEval): Burst pattern → Max catches single-token collapse
        - V×D conjunction: Only penalizes when BOTH varentropy AND dispersion high
          * High V alone: Protected (valid reasoning)
          * High D alone: Protected (rare words)
          * High V AND High D: Penalized (confused generation)

        Cross-dataset SOTA (Avg AUROC = 0.68):
          * HaluEval QA: 0.84    (V×D detects confusion bursts)
          * HaluEval Summ: 0.61  (V×D detects sustained confusion)
          * RAGTruth QA: 0.72    (JSD detects FFN deception)
          * RAGTruth Summ: 0.56  (JSD detects systemic override)

    v21 Core Equation (Prompt-Gated Fusion Architecture):
        Risk_seq = Max(R_confusion, (1 - w_prompt) × R_deception)
        w_prompt = sigmoid((V_prompt - τ) × α)
        R_confusion = Max(V_norm × D)       # Safe Core (works on all tasks)
        R_deception = Mean(JSD)              # Gated based on prompt complexity

        Key Innovation: Prompt Complexity Gating
        - JSD detects "Internal Conflict" not "Hallucination"
        - In standard RAG: Hallucination → High Conflict → High JSD (works)
        - In counterfactual RAG: Hallucination → Low Conflict → Low JSD (INVERTED!)
        - Solution: Gate JSD based on prompt varentropy (complexity)
          * Complex prompts (counterfactual): High V_prompt → suppress JSD
          * Simple prompts (standard RAG): Low V_prompt → use JSD

        Evidence (AUROC on Faithfulness Tasks):
          | Dataset                  | V×D Only | JSD Alone | v19  |
          |--------------------------|----------|-----------|------|
          | FaithEval Counterfactual | 0.71     | 0.43      | 0.53 |
          | FaithEval Unanswerable   | 0.88     | 0.14      | 0.14 |
          | RAGBench                 | 0.75     | 0.20      | 0.33 |

        V×D is the "Safe Core" - works correctly on ALL task types.
        v21 preserves JSD for standard RAG while protecting against inversion.
    """

    # === CORE MECHANISM ===
    semantic_layers: int = 4
    """Number of final transformer layers to analyze."""

    varentropy_lambda: float = 1.0
    """Varentropy weighting for dispersion. Higher = penalize oscillating confidence more."""

    # === CALIBRATION ===
    sigma_multiplier: float = -1.0
    """Z-score threshold for adaptive CPG detection. Negative = below mean."""

    calibration_window: int = 64
    """Tokens to analyze during calibration. Captures instruction uncertainty."""

    # === CLASSIFICATION ===
    hallucination_threshold: float = 0.5
    """Decision boundary. Uncertainty > threshold = hallucination."""

    # === HARDWARE ===
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    """Compute precision."""

    # === VERSION DISPATCH ===
    version: int = 19
    """Algorithm version: 2-19. v19 (Hinge-Risk) recommended for cross-dataset SOTA."""

    # === V3 PARAMETERS ===
    lambda_struct: float = 2.0
    """v3: Sigmoid sensitivity for varentropy centering. Higher = sharper threshold."""

    tau: float = 5.0
    """v3: Cognitive load threshold. V < τ = rote retrieval, V ≥ τ = active reasoning."""

    entropy_floor: float = 0.01
    """v3: Loop breaker - force score=0.0 when H < floor (catches stuttering loops)."""

    # === V4 PARAMETERS (Semantic Shielding) ===
    varentropy_scale: float = 3.0
    """v4: Scaling factor for varentropy penalty. tanh(V/scale) normalizes V to [0,1]."""

    dispersion_scale: float = 0.15
    """v4: Scaling factor for dispersion shield. Higher = more lenient shield, only breaks on high D."""

    # === V6 PARAMETERS (Gaussian Complexity Matching) ===
    complexity_sigma: float = 0.5
    """v6: Gaussian tolerance width for complexity matching.
    Controls how much complexity deviation from expected ratio is allowed.
    - σ=0.3: Strict - R_t must be very close to center
    - σ=0.5: Moderate (default) - R_t within ±0.5 of center scores well
    - σ=1.0: Lenient - Wide tolerance for complexity variation
    """

    complexity_epsilon: float = 0.1
    """v6: Floor for varentropy_mu to prevent division artifacts when prompt
    has very low varentropy. Ensures R_t computation is numerically stable."""

    complexity_center: float = 0.4
    """v6: Expected complexity ratio R_t = V_response / V_prompt.
    - For short-answer QA (RAGTruth): center ≈ 0.3-0.5 (response simpler than prompt)
    - For reasoning tasks (HaluEval): center ≈ 0.8-1.2 (response matches prompt)
    - Default 0.4 works well for most QA benchmarks where answers are shorter than prompts.
    """

    # === V7 PARAMETERS (Geometric Manifold Adherence) ===
    lid_window_size: int = 8
    """v7: Sliding window size for Local Intrinsic Dimension (LID) computation.
    - Smaller (4-6): More local, faster, but noisier
    - Medium (8): Default balance of context vs speed
    - Larger (12-16): More context, captures longer-range patterns, slower
    """

    # === V8/V10/V11/V12 PARAMETERS (Residual Stream Contrast) ===
    jsd_top_k: int = 50
    """v8/v10/v11/v12: Top-k approximation for JSD computation.
    Uses top-k logits for O(k) instead of O(V) per token.
    - 50 (default): Fast, captures main prediction mass (~2600x faster than full vocab)
    - 100: More thorough, better for summarization tasks
    """

    # === V11 PARAMETERS (Information Physics) ===
    softmax_beta: float = 1.0
    """v11: Temperature parameter for SoftMax (LogSumExp) aggregation.
    Controls sensitivity to peak energy vs average energy.
    - β=1.0 (default): Mean-like (optimal for cross-dataset SOTA)
    - β=5.0: Robust anomaly detection (amplifies spikes, hurts RAGTruth)
    - β=10.0: Approaches Max (dominated by highest energy)
    Empirically β=1.0 achieves best cross-dataset performance.
    """

    # === V13 PARAMETERS (Adaptive Regime Switching - Length-based) ===
    regime_threshold: float = 30.0
    """v13: Sequence length threshold for regime switching (τ_len).
    - Length < threshold → favor Stability signal (QA mode)
    - Length > threshold → favor Grounding signal (RAG mode)
    - Default 30 works well for QA (~10 tokens) vs RAG (~100 tokens) separation.
    NOTE: v17 uses varentropy_threshold instead (thermodynamic gating).
    """

    regime_slope: float = 0.2
    """v13: Sigmoid slope for regime transition (α).
    Controls how sharply the detector switches between Stability and Grounding.
    - slope=0.2 (default): Smooth transition over [20, 40] tokens
    - slope=0.5: Sharper transition over [24, 36] tokens
    - slope=0.1: Very gradual transition over [10, 50] tokens
    w = sigmoid((Length - τ_len) × α)
    NOTE: v17 uses varentropy_slope instead (thermodynamic gating).
    """

    # === V17/V19 PARAMETERS (Varentropy Hinge) ===
    varentropy_threshold: float = 2.0
    """v17/v19: Varentropy threshold for reasoning protection.
    v17 (Thermodynamic Gating - DEPRECATED):
        - V_mean < threshold → Deception Regime (trust JSD)
        - V_mean > threshold → Confusion Regime (trust V×(1-A))
    v19 (Hinge-Risk Architecture - RECOMMENDED):
        - V < threshold → Confusion Risk = 0 (Reasoning Protected)
        - V > threshold → Confusion Risk = tanh((V - τ) / scale)
        - Creates "Reasoning Protection Zone" in [0, 2.0] range
    Evidence-based:
        - RAGTruth V ∈ [1.0, 1.5] - mostly protected
        - HaluEval QA halls V ~ 4.7 - strongly penalized
        - HaluEval Summ halls V ~ 2.5 - penalized (τ=2.0 catches this)
    Threshold 2.0 balances: protects RAGTruth reasoning, catches HaluEval confusion.
    """

    varentropy_slope: float = 2.0
    """v17: Sigmoid slope for thermodynamic gating.
    Controls how sharply the gate transitions between regimes.
    - 2.0 (default): Moderate transition
    - 5.0: Sharp regime switching
    - 1.0: Gradual blending
    gate = sigmoid((V_mean - threshold) × slope)
    """

    # === V21 PARAMETERS (Prompt-Gated Fusion) ===
    prompt_gate_threshold: float = 3.0
    """v21: Prompt varentropy threshold for JSD gating.
    Controls when JSD signal is suppressed due to complex/counterfactual context.
    - V_prompt < threshold → JSD active (standard RAG)
    - V_prompt > threshold → JSD suppressed (counterfactual protection)
    Evidence-based:
        - Standard RAG prompts: V_prompt ~ 1.5-2.5
        - Counterfactual prompts: V_prompt ~ 3.5-4.5
    Threshold 3.0 separates normal from complex contexts.
    w_prompt = sigmoid((V_prompt - threshold) × slope)
    """

    prompt_gate_slope: float = 1.0
    """v21: Sigmoid slope for prompt complexity gating.
    Controls how sharply the gate transitions.
    - 1.0 (default): Gradual transition (±1.0 varentropy = ±0.73 gate change)
    - 2.0: Moderate transition (±0.5 varentropy = ±0.76 gate change)
    - 0.5: Very gradual transition (±2.0 varentropy = ±0.76 gate change)
    """

    # === LEGACY PARAMETERS (kept for API compatibility, ignored in v3.2) ===
    gate_threshold: float = 0.3
    """DEPRECATED in v3.2: Was τ_G for MLP override threshold."""

    emergence_threshold: float = 0.5
    """DEPRECATED in v3.2: Was τ_E for valid reasoning detection."""

    dispersion_threshold: float = 0.15
    """v15: Threshold for dispersion gate. Semantic shielding activates above this value.
    Evidence-based: HaluEval Facts D=0.14 vs Halls D=0.23. Setting 0.15 protects facts."""

    # === V15 PARAMETERS (Coherence-Interaction) ===
    dispersion_stiffness: float = 20.0
    """v15: Sigmoid sharpness for dispersion gate.
    Controls how sharply the shield transitions around dispersion_threshold.
    - 20.0 (default): ±0.05 transition window
    - 10.0: ±0.10 transition window (more gradual)
    """

    # === AGGREGATION ===
    aggregation_method: Literal["mean", "importance_weighted", "percentile_10", "min", "geometric_mean", "sliding_window_min", "softmax"] = "mean"
    """
    Token-to-sequence aggregation strategy:
    - mean: Standard statistical mean (v5 default - v5 pre-aggregates internally)
    - importance_weighted: Varentropy-weighted mean (upweights "thinking" tokens)
    - percentile_10: Weakest-link via 10th percentile (safety-critical)
    - min: Weakest-link via minimum (v4 default - captures hallucination bursts)
    - geometric_mean: Principled probabilistic aggregation
      Treats sequence as chain of independent faithfulness probabilities.
      Geometric mean = exp(mean(log(scores))) = "Average Likelihood of Truth"
    - sliding_window_min: Phrase-level weakest-link (v9 recommended)
      Computes mean within 5-token windows, then takes min across windows.
      Smooths single-token noise while catching phrase-level hallucinations.
    - softmax: LogSumExp aggregation (v11 default)
      Differentiable interpolation between mean and max: (1/β) × log(mean(exp(β×x)))
      - β→0: Mean (dilutes spikes), β→∞: Max (ignores average)
      - β=5.0 (default): Robust anomaly detection

    Note: For v5/v10/v11, aggregation is performed internally with version-specific strategy.
    This setting is effectively a no-op for those versions.
    """

    def __post_init__(self):
        if self.semantic_layers < 1:
            raise ValueError(f"semantic_layers must be >= 1, got {self.semantic_layers}")
        if not 0.0 <= self.varentropy_lambda <= 5.0:
            raise ValueError(f"varentropy_lambda must be in [0, 5], got {self.varentropy_lambda}")
        if self.calibration_window < 1:
            raise ValueError(f"calibration_window must be >= 1, got {self.calibration_window}")
        if not 0.0 <= self.hallucination_threshold <= 1.0:
            raise ValueError(f"hallucination_threshold must be in [0, 1], got {self.hallucination_threshold}")
        # Version validation
        if self.version not in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21):
            raise ValueError(f"version must be 2-17, 19, or 21, got {self.version}")
        # v3.4 validation (power law and cognitive threshold)
        if not 0.0 <= self.lambda_struct <= 10.0:
            raise ValueError(f"lambda_struct must be in [0, 10], got {self.lambda_struct}")
        if not 1.0 <= self.tau <= 10.0:
            raise ValueError(f"tau must be in [1, 10], got {self.tau}")
        if not 0.0 <= self.entropy_floor <= 0.5:
            raise ValueError(f"entropy_floor must be in [0, 0.5], got {self.entropy_floor}")
        if self.aggregation_method not in ("mean", "importance_weighted", "percentile_10", "min", "geometric_mean", "sliding_window_min", "softmax"):
            raise ValueError(f"aggregation_method must be mean/importance_weighted/percentile_10/min/geometric_mean/sliding_window_min/softmax, got {self.aggregation_method}")
        # v4 validation (Semantic Shielding)
        if not 0.1 <= self.varentropy_scale <= 10.0:
            raise ValueError(f"varentropy_scale must be in [0.1, 10], got {self.varentropy_scale}")
        if not 0.01 <= self.dispersion_scale <= 1.0:
            raise ValueError(f"dispersion_scale must be in [0.01, 1.0], got {self.dispersion_scale}")
        # v6 validation (Gaussian Complexity Matching)
        if not 0.1 <= self.complexity_sigma <= 2.0:
            raise ValueError(f"complexity_sigma must be in [0.1, 2.0], got {self.complexity_sigma}")
        if not 0.01 <= self.complexity_epsilon <= 1.0:
            raise ValueError(f"complexity_epsilon must be in [0.01, 1.0], got {self.complexity_epsilon}")
        if not 0.1 <= self.complexity_center <= 2.0:
            raise ValueError(f"complexity_center must be in [0.1, 2.0], got {self.complexity_center}")
        # v7 validation (Geometric Manifold Adherence)
        if not 2 <= self.lid_window_size <= 32:
            raise ValueError(f"lid_window_size must be in [2, 32], got {self.lid_window_size}")
        # v8 validation (Residual Stream Contrast)
        if not 10 <= self.jsd_top_k <= 500:
            raise ValueError(f"jsd_top_k must be in [10, 500], got {self.jsd_top_k}")
        # v11 validation (Information Physics)
        if not 0.1 <= self.softmax_beta <= 20.0:
            raise ValueError(f"softmax_beta must be in [0.1, 20.0], got {self.softmax_beta}")
        # v13 validation (Adaptive Regime Switching)
        if not 5.0 <= self.regime_threshold <= 100.0:
            raise ValueError(f"regime_threshold must be in [5.0, 100.0], got {self.regime_threshold}")
        if not 0.05 <= self.regime_slope <= 1.0:
            raise ValueError(f"regime_slope must be in [0.05, 1.0], got {self.regime_slope}")
        # v21 validation (Prompt-Gated Fusion)
        if not 0.5 <= self.prompt_gate_threshold <= 10.0:
            raise ValueError(f"prompt_gate_threshold must be in [0.5, 10.0], got {self.prompt_gate_threshold}")
        if not 0.1 <= self.prompt_gate_slope <= 5.0:
            raise ValueError(f"prompt_gate_slope must be in [0.1, 5.0], got {self.prompt_gate_slope}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
