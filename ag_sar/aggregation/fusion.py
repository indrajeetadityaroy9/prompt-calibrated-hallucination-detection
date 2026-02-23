"""
Prompt-anchored signal aggregation.

Direct (CUS) or PIT (POS/DPS/DoLa/CGD) normalization.
Fusion: w_i = (1/var_i) × (1-H_i)^κ — DerSimonian & Laird (1986) + adaptive entropy gating.
κ = 1 + median(prompt decisiveness) ∈ [1, 2], derived from prompt-tail signal entropy.
Token-level: conflict-aware weighted mean. Response-level: signal-first aggregation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set

from ..numerics import EPS


@dataclass
class AggregationResult:
    """Result of prompt-anchored aggregation."""

    # Response-level risk score [0, 1]
    risk: float

    # Per-token risk scores
    token_risks: np.ndarray

    # Raw signal values per signal (for analysis)
    z_scores: Dict[str, np.ndarray]

    # Probabilities per signal (PIT-normalized or direct)
    probabilities: Dict[str, np.ndarray]

    # Anchor statistics used (for transparency)
    anchor_stats: Dict[str, Dict[str, float]]

    # Per-signal response-level probabilities (for diagnostics)
    signal_response_probs: Dict[str, float] = field(default_factory=dict)

    # Per-token conflict coefficient [0, 1] (diagnostic)
    conflict_per_token: np.ndarray = field(default_factory=lambda: np.array([]))


class PromptAnchoredAggregator:
    """Training-free: normalize → entropy-gated token fusion → precision-weighted response risk."""

    @staticmethod
    def _pit_normalize(sorted_vals: np.ndarray, response_vals: np.ndarray) -> np.ndarray:
        """PIT: p = (rank + 0.5)/(n + 1) via empirical CDF. Haldane-Anscombe correction."""
        n = len(sorted_vals)
        ranks = np.searchsorted(sorted_vals, response_vals, side='right').astype(float)
        return (ranks + 0.5) / (n + 1)

    @staticmethod
    def _adaptive_kappa(prompt_stats: Dict[str, Dict]) -> float:
        """Derive entropy gating exponent from prompt-tail signal decisiveness.

        κ = 1 + median(per-signal decisiveness), where decisiveness_i = median(1 - H(ref_vals_i)).
        κ ∈ [1, 2]: floor = linear gating (noisy prompt), ceiling = quadratic (decisive prompt).
        """
        decisiveness = []
        for sig, stats in prompt_stats.items():
            if "sorted_vals" not in stats:
                continue
            sv = np.clip(stats["sorted_vals"], EPS, 1 - EPS)
            H = -(sv * np.log2(sv) + (1 - sv) * np.log2(1 - sv))
            decisiveness.append(float(np.median(1.0 - H)))
        if not decisiveness:
            return 2.0  # All signals are direct-mode (CUS only) → max gating
        return 1.0 + float(np.median(decisiveness))

    _SIGNALS = {"cus", "pos", "dps", "dola", "cgd", "std"}

    def compute_risk(
        self,
        prompt_stats: Dict[str, Dict[str, float]],
        response_signals: Dict[str, np.ndarray],
        disabled_signals: Set[str] = None,
    ) -> AggregationResult:
        """Signal normalization → entropy-gated token fusion → precision-weighted response risk."""
        # Determine which signals we can process
        available_signals = set(prompt_stats.keys()) & set(response_signals.keys()) & self._SIGNALS
        if disabled_signals:
            available_signals -= disabled_signals

        # Get number of tokens
        first_signal = next(iter(available_signals))
        n_tokens = len(response_signals[first_signal])

        # Stage 1+2: Signal-specific normalization → probabilities
        z_scores = {}
        probabilities = {}
        anchor_stats_used = {}

        for sig in available_signals:
            response_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]

            if stats.get("mode") == "direct":
                probabilities[sig] = np.clip(response_vals, 0.0, 1.0)
                z_scores[sig] = response_vals
                anchor_stats_used[sig] = {"mode": "direct"}
            else:
                sorted_vals = stats["sorted_vals"]
                p = self._pit_normalize(sorted_vals, response_vals)

                z_scores[sig] = response_vals
                probabilities[sig] = p
                anchor_stats_used[sig] = {
                    "mode": "pit",
                    "n_reference": len(sorted_vals),
                }

        # Compute kappa once for both fusion stages
        kappa = self._adaptive_kappa(prompt_stats)

        # Stage 3: Conflict-aware precision-weighted fusion (per-token)
        token_risks, conflict = self._conflict_aware_fusion(
            probabilities, prompt_stats, n_tokens, kappa,
        )

        # Stage 4: Response-level risk via precision-weighted signal-first aggregation
        risk, signal_probs = self._response_level_risk(
            response_signals, prompt_stats, available_signals, kappa,
        )

        return AggregationResult(
            risk=risk,
            token_risks=token_risks,
            z_scores=z_scores,
            probabilities=probabilities,
            anchor_stats=anchor_stats_used,
            signal_response_probs=signal_probs,
            conflict_per_token=conflict,
        )

    def _conflict_aware_fusion(
        self,
        probabilities: Dict[str, np.ndarray],
        prompt_stats: Dict[str, Dict[str, float]],
        n_tokens: int,
        kappa: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """w_i(t) = (1/var_i) × (1-H_i)^κ. DerSimonian & Laird (1986) + adaptive entropy gating."""

        # Compute per-signal precision (1/variance). Fallback 0.25 = Var(Bernoulli(0.5)).
        precisions = {}
        for sig in probabilities:
            variance = prompt_stats.get(sig, {}).get("variance", 0.25)
            precisions[sig] = 1.0 / max(variance, EPS)

        # Normalize precisions so max = 1 (prevents numerical dominance)
        max_prec = max(precisions.values()) if precisions else 1.0
        max_prec = max(max_prec, EPS)
        for sig in precisions:
            precisions[sig] /= max_prec

        # Entropy-gated fusion with precision weighting
        weighted_sum = np.zeros(n_tokens)
        weight_sum = np.zeros(n_tokens)
        all_probs = []

        for sig, p in probabilities.items():
            p_clipped = np.clip(p, EPS, 1 - EPS)
            H = -(p_clipped * np.log2(p_clipped) + (1 - p_clipped) * np.log2(1 - p_clipped))
            w = precisions[sig] * (1.0 - H) ** kappa
            weighted_sum += w * p
            weight_sum += w
            all_probs.append(p)

        safe_denom = np.where(weight_sum > EPS, weight_sum, 1.0)
        token_risks = np.where(weight_sum > EPS, weighted_sum / safe_denom, 0.0)

        # Conflict coefficient: variance of probabilities across signals per token.
        # Normalized by observed max conflict (data-driven, not theoretical max).
        if len(all_probs) >= 2:
            prob_stack = np.stack(all_probs, axis=0)
            conflict = np.var(prob_stack, axis=0)
            max_conflict = max(float(conflict.max()), EPS)
            conflict_normalized = conflict / max_conflict
        else:
            conflict_normalized = np.zeros(n_tokens)

        return token_risks, conflict_normalized

    def _response_level_risk(
        self,
        response_signals: Dict[str, np.ndarray],
        prompt_stats: Dict[str, Dict[str, float]],
        available_signals: Set[str],
        kappa: float,
    ) -> Tuple[float, Dict[str, float]]:
        """Signal-first: per-signal mean → PIT normalize → precision × entropy weighted fusion."""
        signal_probs = {}
        signal_weights = {}

        for sig in available_signals:
            raw_vals = np.asarray(response_signals[sig])
            if len(raw_vals) == 0:
                continue
            stats = prompt_stats.get(sig, {})
            mean_val = float(np.mean(raw_vals))

            if stats.get("mode") == "direct":
                p = float(np.clip(mean_val, 0.0, 1.0))
            else:
                sorted_vals = stats["sorted_vals"]
                p_arr = self._pit_normalize(sorted_vals, np.array([mean_val]))
                p = float(p_arr[0])

            variance = stats.get("variance", 0.25)
            precision = 1.0 / max(variance, EPS)

            signal_probs[sig] = p

            # Weight = precision × (1 - H(p))^κ
            p_clip = np.clip(p, EPS, 1 - EPS)
            H = -(p_clip * np.log2(p_clip) + (1 - p_clip) * np.log2(1 - p_clip))
            entropy_weight = (1.0 - H) ** kappa
            signal_weights[sig] = float(precision * entropy_weight)

        total_weight = sum(signal_weights.values())
        if total_weight < EPS:
            return float(np.mean(list(signal_probs.values()))), signal_probs

        risk = sum(signal_probs[s] * signal_weights[s] for s in signal_probs) / total_weight
        return float(risk), signal_probs
