"""
Prompt-Anchored Signal Aggregation for Hallucination Detection.

Signal-specific normalization:
    - Direct signals (CUS): Value IS the probability (no z-score).
      CUS ∈ [0,1] with semantic meaning (0=grounded, 1=ungrounded).
    - Z-scored signals (POS, DPS): Prompt-relative z-scoring + sigmoid.
      Z_i(t) = (S_i(t) - μ_prompt) / σ_prompt → P_i(t) = sigmoid(Z_i(t))

Fusion: Noisy-OR: P(Risk) = 1 - ∏(1 - P_i(t))
Aggregation: Adaptive quantile q = 1 - 1/(n+1) (order statistics).

No magic numbers, no learned weights, no hardcoded priors.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Union

from ..numerics import EPS


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


class PromptAnchoredAggregator:
    """
    Training-free signal aggregation with signal-specific normalization.

    Pipeline:
    ═════════
    1. Per-signal normalization (signal-specific):
       - Direct signals (CUS): value = probability (no z-score)
       - Z-scored signals (POS, DPS): z = (s - μ_prompt) / σ_prompt → sigmoid
    2. Fuse via Noisy-OR: R(t) = 1 - Π(1 - p_i(t))
    3. Aggregate via adaptive quantile: q = 1 - 1/(n+1)

    No magic numbers. No learned weights. No calibration data.
    """

    # Signals that use direct probability mapping (no z-score).
    # CUS is in [0,1] with semantic meaning — z-scoring destroys this
    # and causes saturation when prefill/generation distributions differ.
    DIRECT_SIGNALS = {"cus"}

    def __init__(
        self,
        active_signals: Set[str] = None,
    ):
        self.active_signals = active_signals or {"cus", "pos", "dps"}

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

            if sig in self.DIRECT_SIGNALS or stats.get("mode") == "direct":
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

        # Stage 3: Noisy-OR fusion (uniform, no weights)
        token_risks = self._noisy_or_fusion(probabilities, n_tokens)

        # Stage 4: Adaptive quantile aggregation
        # q(n) = 1 - 1/(n+1): order-statistics approach that naturally
        # adapts to response length. Short responses use lower quantile
        # (avoid single-token noise), long responses focus on the tail.
        # Reference: Blom (1958); David & Nagaraja (2003)
        if len(token_risks) == 0:
            risk = 0.0
        else:
            n = len(token_risks)
            q = max(0.5, min(1.0 - 1.0 / (n + 1), 0.999))
            risk = float(np.percentile(token_risks, 100 * q))

        return AggregationResult(
            risk=risk,
            token_risks=token_risks,
            z_scores=z_scores,
            probabilities=probabilities,
            anchor_stats=anchor_stats_used,
        )

    def _noisy_or_fusion(
        self,
        probabilities: Dict[str, np.ndarray],
        n_tokens: int,
    ) -> np.ndarray:
        """
        Fuse signals via uniform Noisy-OR.

        R(t) = 1 - Π_s (1 - p_s(t))

        Interpretation: probability that at least one signal indicates risk.
        No weights - all signals contribute equally (independence assumption).
        """
        # Start with "no risk" (all signals say OK)
        complement_product = np.ones(n_tokens)

        for sig, p in probabilities.items():
            complement_product *= (1.0 - p)

        # Noisy-OR: P(at least one cause)
        return 1.0 - complement_product
