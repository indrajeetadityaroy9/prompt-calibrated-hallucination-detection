"""
Prompt-Anchored Signal Aggregation for Hallucination Detection.

Implements the Standard Model:
    1. Signal Extraction → raw metrics (H, JSD, etc.)
    2. Prompt-Relative Z-Scoring: Z_i(t) = (S_i(t) - μ_prompt) / (σ_prompt + ε)
    3. Probabilistic Mapping: P_i(t) = sigmoid(Z_i(t))
    4. Independence Assumption (Noisy-OR): P(Risk) = 1 - ∏(1 - P_i(t))

CRITICAL: Normalizes RESPONSE against PROMPT statistics (not self-normalization).
This prevents the "Relativity Trap" where hallucinated responses normalize their
own high entropy to appear normal.

No magic numbers, no learned weights - pure information-theoretic approach.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union



# Numerical stability constant (the ONLY allowed constant)
EPS = 1e-10


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
    Training-free signal aggregation with prompt-anchored normalization.

    The Standard Model Pipeline:
    ════════════════════════════
    1. Extract prompt statistics (μ_prompt, σ_prompt) during prefill
    2. Compute anchored z-scores: z = (s - μ_prompt) / σ_prompt
    3. Map to probabilities via standard sigmoid: p = 1/(1 + e^(-z))
    4. Fuse signals via Noisy-OR: R = 1 - Π(1 - p_i)
    5. Aggregate to response level (max of token risks)

    No magic numbers. No learned weights. No calibration data.
    A z-score of 3 is always significant regardless of model/dataset.
    """

    def __init__(
        self,
        active_signals: Optional[Set[str]] = None,
    ):
        """
        Initialize the aggregator.

        Args:
            active_signals: Set of signals to include. If None, uses all available.
        """
        self.active_signals = active_signals or {
            "jsd", "entropy", "inv_margin", "lci", "var_logp"
        }

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

        # Stage 1: Compute prompt-anchored z-scores
        z_scores = {}
        anchor_stats_used = {}

        for sig in available_signals:
            response_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]

            mu_prompt = stats.get("mu", 0.0)
            sigma_prompt = stats.get("sigma", 1.0)

            # Ensure numerical stability (sigma >= EPS)
            sigma_prompt = max(sigma_prompt, EPS)

            # Anchored z-score: how much does response deviate from prompt baseline?
            z = (response_vals - mu_prompt) / sigma_prompt

            z_scores[sig] = z
            anchor_stats_used[sig] = {
                "mu_prompt": mu_prompt,
                "sigma_prompt": sigma_prompt,
            }

        # Stage 2: Sigmoid probability mapping (standard sigmoid, no parameters)
        probabilities = {}
        for sig in available_signals:
            z = z_scores[sig]
            # Clip for numerical stability
            z_clipped = np.clip(z, -20, 20)
            probabilities[sig] = 1.0 / (1.0 + np.exp(-z_clipped))

        # Stage 3: Noisy-OR fusion (uniform, no weights)
        token_risks = self._noisy_or_fusion(probabilities, n_tokens)

        # Stage 4: Response-level aggregation (max - no magic percentile)
        risk = float(np.max(token_risks)) if len(token_risks) > 0 else 0.0

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


def compute_prompt_statistics(
    signal_values: Dict[str, np.ndarray],
    tail_fraction: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics from prompt signal values.

    Uses tail sampling: only the last `tail_fraction` of tokens are used.
    This focuses on the tokens most relevant to the response.

    Args:
        signal_values: Dict mapping signal names to arrays of values
        tail_fraction: Fraction of tokens to use (from end)

    Returns:
        Dict mapping signal names to {"mu": float, "sigma": float}
    """
    stats = {}

    for sig, values in signal_values.items():
        values = np.asarray(values)
        n = len(values)

        if n == 0:
            continue

        # Tail sampling
        start_idx = max(0, int(n * (1 - tail_fraction)))
        tail_values = values[start_idx:]

        if len(tail_values) > 0:
            stats[sig] = {
                "mu": float(np.mean(tail_values)),
                "sigma": float(np.std(tail_values)) if len(tail_values) > 1 else EPS,
                "n_tokens": len(tail_values),
            }

    return stats
