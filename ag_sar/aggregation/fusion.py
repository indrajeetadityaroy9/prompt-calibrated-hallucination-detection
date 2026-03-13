"""
Prompt-anchored signal aggregation with cross-signal precision fusion.

Direct (CUS/SPT/spectral_gap) or PIT (POS/DPS) normalization.
Fusion: w_i = Σ_j Ω_ij * (1-H_j)^kappa — generalized DerSimonian & Laird (1986)
with full cross-signal precision matrix Ω = Σ⁻¹ (Hartung, Knapp & Sinha, 2008).
kappa = 1 + median(prompt decisiveness) in [1, 2].
Token-level: entropy-gated precision-coupled weighted mean.
Response-level: signal-first aggregation with cross-signal precision.
"""

from typing import Any

import numpy as np
from dataclasses import dataclass, field

from ..numerics import EPS


@dataclass
class AggregationResult:
    """Result of prompt-anchored aggregation."""
    risk: float
    token_risks: np.ndarray
    raw_signals: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    signal_response_probs: dict[str, float] = field(default_factory=dict)


# Canonical signal ordering and index mapping for precision matrix
_SIGNAL_ORDER = ["cus", "pos", "dps", "spt", "spectral_gap"]
_SIGNAL_INDEX = {sig: i for i, sig in enumerate(_SIGNAL_ORDER)}
_SIGNAL_SET = set(_SIGNAL_ORDER)


class PromptAnchoredAggregator:
    """Training-free: normalize -> cross-signal precision fusion -> response risk."""

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
    def _adaptive_kappa(prompt_stats: dict[str, dict]) -> float:
        """kappa = 1 + median(per-signal decisiveness)."""
        decisiveness = []
        for sig, stats in prompt_stats.items():
            if sig.startswith("_"):
                continue
            if "sorted_vals" not in stats:
                continue
            sv = stats["sorted_vals"]
            H = PromptAnchoredAggregator._binary_entropy(sv)
            decisiveness.append(float(np.median(1.0 - H)))
        return 1.0 + float(np.median(decisiveness))

    def compute_risk(
        self,
        prompt_stats: dict[str, Any],
        response_signals: dict[str, np.ndarray],
        disabled_signals: set[str] | None = None,
    ) -> AggregationResult:
        """Signal normalization -> cross-signal precision fusion -> response risk."""
        available_signals = set(prompt_stats.keys()) & set(response_signals.keys()) & _SIGNAL_SET
        if disabled_signals:
            available_signals -= disabled_signals

        # Use deterministic ordering from _SIGNAL_ORDER
        ordered_available = [s for s in _SIGNAL_ORDER if s in available_signals]
        n_tokens = len(response_signals[ordered_available[0]])

        raw_signals = {}
        probabilities = {}

        for sig in available_signals:
            response_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]

            if stats.get("mode") == "direct":
                probabilities[sig] = response_vals
                raw_signals[sig] = response_vals
            else:
                sorted_vals = stats["sorted_vals"]
                p = self._pit_normalize(sorted_vals, response_vals)
                raw_signals[sig] = response_vals
                probabilities[sig] = p

        kappa = self._adaptive_kappa(prompt_stats)

        precision_matrix = prompt_stats["_cross_signal_precision"]

        token_risks = self._entropy_gated_fusion(
            probabilities, prompt_stats, n_tokens, kappa, precision_matrix,
        )

        risk, signal_probs = self._response_level_risk(
            response_signals, prompt_stats, available_signals, kappa, precision_matrix,
        )

        return AggregationResult(
            risk=risk,
            token_risks=token_risks,
            raw_signals=raw_signals,
            probabilities=probabilities,
            signal_response_probs=signal_probs,
        )

    @staticmethod
    def _compute_cross_signal_weights(
        entropy_mods: dict[str, np.ndarray],
        precision_matrix: np.ndarray,
        available_signals: set[str],
    ) -> dict[str, np.ndarray]:
        """Compute precision-coupled entropy-modulated weights.

        w_i = Σ_j Ω_ij × e_j  where e_j = (1 - H_j)^κ

        When Ω is diagonal, this reduces to w_i = (1/var_i) × e_i (current AG-SAR).
        Off-diagonal terms couple weights: if signals i and j are correlated (Ω_ij < 0),
        high informativeness of j reduces i's effective weight (avoids double-counting).
        """
        ordered_sigs = [s for s in _SIGNAL_ORDER if s in available_signals]

        # Compute w_i = Σ_j Ω_ij × e_j for available signals
        weights = {}
        for sig_i in ordered_sigs:
            idx_i = _SIGNAL_INDEX[sig_i]
            w = np.zeros_like(entropy_mods[sig_i])
            for sig_j in ordered_sigs:
                idx_j = _SIGNAL_INDEX[sig_j]
                w = w + precision_matrix[idx_i, idx_j] * entropy_mods[sig_j]
            # Clamp to non-negative (negative weights from strong negative correlation
            # indicate redundancy — set to zero rather than subtracting)
            weights[sig_i] = np.maximum(w, 0.0)

        return weights

    def _entropy_gated_fusion(
        self,
        probabilities: dict[str, np.ndarray],
        prompt_stats: dict[str, dict[str, float]],
        n_tokens: int,
        kappa: float,
        precision_matrix: np.ndarray,
    ) -> np.ndarray:
        """Cross-signal precision-coupled entropy-gated fusion.

        w_i(t) = Σ_j Ω_ij × (1-H_j(t))^κ
        risk(t) = Σ_i max(w_i, 0) × p_i(t) / Σ_i max(w_i, 0)
        """
        entropy_mods = {}
        for sig, p in probabilities.items():
            H = self._binary_entropy(p)
            entropy_mods[sig] = (1.0 - H) ** kappa

        weights = self._compute_cross_signal_weights(
            entropy_mods, precision_matrix, set(probabilities.keys()),
        )

        weighted_sum = np.zeros(n_tokens)
        weight_sum = np.zeros(n_tokens)

        for sig, p in probabilities.items():
            w = weights[sig]
            weighted_sum += w * p
            weight_sum += w

        return np.where(weight_sum > 0, weighted_sum / weight_sum, 0.5)

    def _response_level_risk(
        self,
        response_signals: dict[str, np.ndarray],
        prompt_stats: dict[str, dict[str, float]],
        available_signals: set[str],
        kappa: float,
        precision_matrix: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Signal-first: per-signal mean -> normalize -> cross-signal precision fusion."""
        signal_probs = {}

        for sig in available_signals:
            raw_vals = np.asarray(response_signals[sig])
            stats = prompt_stats[sig]
            mean_val = float(np.mean(raw_vals))

            if stats.get("mode") == "direct":
                p = mean_val
            else:
                sorted_vals = stats["sorted_vals"]
                p_arr = self._pit_normalize(sorted_vals, np.array([mean_val]))
                p = float(p_arr[0])

            signal_probs[sig] = p

        entropy_mods = {}
        for sig, p in signal_probs.items():
            H = float(self._binary_entropy(p))
            entropy_mods[sig] = (1.0 - H) ** kappa

        entropy_mods_arr = {sig: np.array([e]) for sig, e in entropy_mods.items()}
        weights_arr = self._compute_cross_signal_weights(
            entropy_mods_arr, precision_matrix, available_signals,
        )
        signal_weights = {sig: float(w[0]) for sig, w in weights_arr.items()}

        total_weight = sum(signal_weights.values())
        risk = sum(signal_probs[s] * signal_weights[s] for s in signal_probs) / (total_weight + EPS)
        return float(risk), signal_probs
