"""
Prompt-anchored signal aggregation.

Direct (CUS) or PIT (POS/DPS) normalization.
Fusion: w_i = (1/var_i) * (1-H_i)^kappa — DerSimonian & Laird (1986) + adaptive entropy gating.
kappa = 1 + median(prompt decisiveness) in [1, 2].
Token-level: entropy-gated weighted mean. Response-level: signal-first aggregation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set

from ..numerics import EPS


@dataclass
class AggregationResult:
    """Result of prompt-anchored aggregation."""
    risk: float
    token_risks: np.ndarray
    z_scores: Dict[str, np.ndarray]
    probabilities: Dict[str, np.ndarray]
    signal_response_probs: Dict[str, float] = field(default_factory=dict)


class PromptAnchoredAggregator:
    """Training-free: normalize -> entropy-gated token fusion -> precision-weighted response risk."""

    @staticmethod
    def _binary_entropy(p):
        """H(p) = -(p log2 p + (1-p) log2(1-p)). Works on scalars and arrays."""
        p = np.clip(p, EPS, 1 - EPS)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    @staticmethod
    def _pit_normalize(sorted_vals: np.ndarray, response_vals: np.ndarray) -> np.ndarray:
        """PIT: p = (rank + 0.5)/(n + 1) via empirical CDF. Haldane-Anscombe correction."""
        n = len(sorted_vals)
        ranks = np.searchsorted(sorted_vals, response_vals, side='right').astype(float)
        return (ranks + 0.5) / (n + 1)

    @staticmethod
    def _adaptive_kappa(prompt_stats: Dict[str, Dict]) -> float:
        """kappa = 1 + median(per-signal decisiveness)."""
        decisiveness = []
        for sig, stats in prompt_stats.items():
            if "sorted_vals" not in stats:
                continue
            sv = stats["sorted_vals"]
            if np.any(sv < 0.0) or np.any(sv > 1.0):
                continue
            H = PromptAnchoredAggregator._binary_entropy(sv)
            decisiveness.append(float(np.median(1.0 - H)))
        if not decisiveness:
            return 2.0
        return 1.0 + float(np.median(decisiveness))

    _SIGNALS = {"cus", "pos", "dps", "spt"}

    def compute_risk(
        self,
        prompt_stats: Dict[str, Dict[str, float]],
        response_signals: Dict[str, np.ndarray],
        disabled_signals: Set[str] = None,
    ) -> AggregationResult:
        """Signal normalization -> entropy-gated token fusion -> precision-weighted response risk."""
        available_signals = set(prompt_stats.keys()) & set(response_signals.keys()) & self._SIGNALS
        if disabled_signals:
            available_signals -= disabled_signals

        first_signal = next(iter(available_signals))
        n_tokens = len(response_signals[first_signal])

        z_scores = {}
        probabilities = {}

        for sig in available_signals:
            response_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]

            if stats.get("mode") == "direct":
                probabilities[sig] = response_vals
                z_scores[sig] = response_vals
            else:
                sorted_vals = stats["sorted_vals"]
                p = self._pit_normalize(sorted_vals, response_vals)
                z_scores[sig] = response_vals
                probabilities[sig] = p

        kappa = self._adaptive_kappa(prompt_stats)

        token_risks = self._entropy_gated_fusion(
            probabilities, prompt_stats, n_tokens, kappa,
        )

        risk, signal_probs = self._response_level_risk(
            response_signals, prompt_stats, available_signals, kappa,
        )

        return AggregationResult(
            risk=risk,
            token_risks=token_risks,
            z_scores=z_scores,
            probabilities=probabilities,
            signal_response_probs=signal_probs,
        )

    def _entropy_gated_fusion(
        self,
        probabilities: Dict[str, np.ndarray],
        prompt_stats: Dict[str, Dict[str, float]],
        n_tokens: int,
        kappa: float,
    ) -> np.ndarray:
        """w_i(t) = (1/var_i) * (1-H_i)^kappa."""
        precisions = {}
        for sig in probabilities:
            variance = prompt_stats[sig]["variance"]
            precisions[sig] = 1.0 / max(variance, EPS)

        max_prec = max(precisions.values())
        for sig in precisions:
            precisions[sig] /= max_prec

        weighted_sum = np.zeros(n_tokens)
        weight_sum = np.zeros(n_tokens)

        for sig, p in probabilities.items():
            H = self._binary_entropy(p)
            w = precisions[sig] * (1.0 - H) ** kappa
            weighted_sum += w * p
            weight_sum += w

        return weighted_sum / weight_sum

    def _response_level_risk(
        self,
        response_signals: Dict[str, np.ndarray],
        prompt_stats: Dict[str, Dict[str, float]],
        available_signals: Set[str],
        kappa: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Signal-first: per-signal mean -> PIT normalize -> precision * entropy weighted fusion."""
        signal_probs = {}
        raw_precisions = {}

        for sig in available_signals:
            raw_vals = np.asarray(response_signals[sig])
            stats = prompt_stats.get(sig, {})
            mean_val = float(np.mean(raw_vals))

            if stats.get("mode") == "direct":
                p = mean_val
            else:
                sorted_vals = stats["sorted_vals"]
                p_arr = self._pit_normalize(sorted_vals, np.array([mean_val]))
                p = float(p_arr[0])

            signal_probs[sig] = p
            variance = prompt_stats[sig]["variance"]
            raw_precisions[sig] = 1.0 / max(variance, EPS)

        max_prec = max(raw_precisions.values())

        signal_weights = {}
        for sig in signal_probs:
            precision = raw_precisions[sig] / max_prec
            H = self._binary_entropy(signal_probs[sig])
            entropy_weight = (1.0 - H) ** kappa
            signal_weights[sig] = float(precision * entropy_weight)

        total_weight = sum(signal_weights.values())
        risk = sum(signal_probs[s] * signal_weights[s] for s in signal_probs) / total_weight
        return float(risk), signal_probs
