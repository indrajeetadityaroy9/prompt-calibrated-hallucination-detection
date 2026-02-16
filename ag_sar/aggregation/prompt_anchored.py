"""
Prompt-Anchored Signal Aggregation for Hallucination Detection.

Signal-specific normalization:
    - Direct signals (CUS): Value IS the probability (no z-score).
      CUS ∈ [0,1] with semantic meaning (0=grounded, 1=ungrounded).
    - Z-scored signals (POS, DPS, DoLa, CGD): Prompt-relative z-scoring + sigmoid.
      Z_i(t) = (S_i(t) - μ_prompt) / σ_prompt → P_i(t) = sigmoid(Z_i(t))

Fusion: Entropy-gated weighted mean.
    w_i(t) = (1 - H_i(t))^κ where H_i = binary entropy of p_i.
    R(t) = Σ w_i·p_i / Σ w_i. Signals at p=0.5 (uninformative) get weight 0.

Token-level: Entropy-gated fusion for per-token risk (span detection).
Response-level: Signal-first aggregation — mean of per-signal response
    probabilities. Captures diffuse mean shifts that entropy gating kills.

No magic numbers, no learned weights, no hardcoded priors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Union

from ..numerics import EPS
from ..config import SIGNAL_REGISTRY, NormMode


@dataclass
class AggregationResult:
    """Result of prompt-anchored aggregation."""

    # Response-level risk score [0, 1]
    risk: float

    # Per-token risk scores
    token_risks: np.ndarray

    # Z-scores per signal (for analysis)
    z_scores: Dict[str, np.ndarray]

    # Probabilities per signal (sigmoid of z-scores)
    probabilities: Dict[str, np.ndarray]

    # Anchor statistics used (for transparency)
    anchor_stats: Dict[str, Dict[str, float]]

    # Per-signal response-level probabilities (for diagnostics)
    signal_response_probs: Dict[str, float] = field(default_factory=dict)


class PromptAnchoredAggregator:
    """
    Training-free signal aggregation with signal-specific normalization.

    Pipeline:
    ═════════
    1. Per-signal normalization (signal-specific):
       - Direct signals (CUS): value = probability (no z-score)
       - Z-scored signals (POS, DPS, DoLa, CGD): z = (s - μ_prompt) / σ_prompt → sigmoid
    2. Token-level: entropy-gated weighted mean for per-token risk (span detection)
       w_i = (1 - H_i)^κ, R(t) = Σ w_i·p_i / Σ w_i
    3. Response-level: mean of per-signal response probabilities (signal-first)
       Captures diffuse mean shifts that entropy gating kills at per-token level.

    No magic numbers. No learned weights. No calibration data.
    """

    @staticmethod
    def _is_direct(sig: str, stats: Dict) -> bool:
        """Check if signal should use direct probability mapping."""
        if stats.get("mode") == "direct":
            return True
        meta = SIGNAL_REGISTRY.get(sig)
        return meta is not None and meta.norm_mode == NormMode.DIRECT

    def __init__(
        self,
        active_signals: Set[str] = None,
    ):
        self.active_signals = active_signals or {"cus", "pos", "dps", "dola", "cgd"}

    def compute_risk(
        self,
        prompt_stats: Dict[str, Dict[str, float]],
        response_signals: Dict[str, np.ndarray],
    ) -> AggregationResult:
        """
        Compute response risk using prompt-anchored normalization.

        Args:
            prompt_stats: Statistics for each signal from prompt.
                         Format: {signal_name: {"mu": float, "sigma": float}}
            response_signals: Signal values for each response token.
                             Format: {signal_name: np.ndarray of shape [n_tokens]}

        Returns:
            AggregationResult containing risk score and diagnostics.
        """
        # Determine which signals we can process
        available_signals = set(prompt_stats.keys()) & set(response_signals.keys()) & self.active_signals

        if not available_signals:
            return AggregationResult(
                risk=0.0,
                token_risks=np.array([]),
                z_scores={},
                probabilities={},
                anchor_stats={},
            )

        # Get number of tokens
        first_signal = next(iter(available_signals))
        n_tokens = len(response_signals[first_signal])

        if n_tokens == 0:
            return AggregationResult(
                risk=0.0,
                token_risks=np.array([]),
                z_scores={},
                probabilities={},
                anchor_stats={},
            )

        # Stage 1+2: Signal-specific normalization → probabilities
        z_scores = {}
        probabilities = {}
        anchor_stats_used = {}

        for sig in available_signals:
            response_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]

            if self._is_direct(sig, stats):
                # Direct mode: value IS the probability (CUS ∈ [0,1])
                probabilities[sig] = np.clip(response_vals, 0.0, 1.0)
                z_scores[sig] = response_vals  # Raw values for diagnostics
                anchor_stats_used[sig] = {"mode": "direct"}
            else:
                # Prompt-anchored z-score → sigmoid
                mu_prompt = stats.get("mu", 0.0)
                sigma_prompt = stats.get("sigma", 1.0)
                sigma_prompt = max(sigma_prompt, EPS)

                z = (response_vals - mu_prompt) / sigma_prompt
                z_clipped = np.clip(z, -20, 20)

                z_scores[sig] = z
                probabilities[sig] = 1.0 / (1.0 + np.exp(-z_clipped))
                anchor_stats_used[sig] = {
                    "mu_prompt": mu_prompt,
                    "sigma_prompt": sigma_prompt,
                    "mode": "zscore",
                }

        # Stage 3: Entropy-gated fusion (per-token, for span detection)
        token_risks = self._entropy_gated_fusion(probabilities, n_tokens)

        # Stage 4: Response-level risk via signal-first aggregation
        # (replaces adaptive quantile which captures outlier tokens in ALL responses)
        risk, signal_probs = self._response_level_risk(
            response_signals, prompt_stats, available_signals,
        )

        return AggregationResult(
            risk=risk,
            token_risks=token_risks,
            z_scores=z_scores,
            probabilities=probabilities,
            anchor_stats=anchor_stats_used,
            signal_response_probs=signal_probs,
        )

    def _entropy_gated_fusion(
        self,
        probabilities: Dict[str, np.ndarray],
        n_tokens: int,
    ) -> np.ndarray:
        """
        Entropy-gated fusion: suppress uninformative signals.

        w_i(t) = (1 - H_i(t))^κ  where H_i = binary entropy of p_i
        R(t) = Σ w_i·p_i / Σ w_i  (weighted mean)

        When p_i = 0.5: H_i = 1.0, w_i = 0 → signal completely suppressed.
        When p_i ≈ 0 or ≈ 1: H_i ≈ 0, w_i ≈ 1 → signal contributes fully.

        If ALL signals are uninformative, R(t) defaults to 0 (no evidence of risk).

        Reference: AGFN (arXiv:2510.01677), AECF (arXiv:2505.15417).
        Adapted for training-free detection with κ=2 (quadratic suppression).
        """
        kappa = 2  # Quadratic suppression: principled, no tuning needed

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
        # Use safe division to avoid RuntimeWarning
        safe_denom = np.where(weight_sum > EPS, weight_sum, 1.0)
        return np.where(weight_sum > EPS, weighted_sum / safe_denom, 0.0)

    def _response_level_risk(
        self,
        response_signals: Dict[str, np.ndarray],
        prompt_stats: Dict[str, Dict[str, float]],
        available_signals: Set[str],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Response-level risk via signal-first aggregation.

        For each signal:
          1. Compute mean of raw values across all response tokens
          2. Normalize: direct signals clip to [0,1],
             z-scored signals use (mean - mu_prompt) / sigma_prompt -> sigmoid

        Fuse via simple (unweighted) mean of per-signal response probabilities.

        Rationale: Discriminative signals (POS, DPS) exhibit a diffuse mean
        shift across tokens, not per-token spikes. Entropy gating destroys
        this because per-token probabilities hover near 0.5 and kappa=2
        suppresses them to near-zero weight. Simple mean preserves rank
        ordering: signals at p=0.5 contribute neutrally, while discriminative
        signals shift the mean. This is zero-parameter.
        """
        signal_probs = {}

        for sig in available_signals:
            raw_vals = np.asarray(response_signals[sig])
            if len(raw_vals) == 0:
                continue
            stats = prompt_stats.get(sig, {})
            mean_val = float(np.mean(raw_vals))

            if self._is_direct(sig, stats):
                signal_probs[sig] = np.clip(mean_val, 0.0, 1.0)
            else:
                mu = stats.get("mu", 0.0)
                sigma = max(stats.get("sigma", 1.0), EPS)
                z = np.clip((mean_val - mu) / sigma, -20, 20)
                signal_probs[sig] = float(1.0 / (1.0 + np.exp(-z)))

        if not signal_probs:
            return 0.0, {}

        risk = float(np.mean(list(signal_probs.values())))
        return risk, signal_probs
