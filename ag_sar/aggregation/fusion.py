import numpy as np
from dataclasses import dataclass

from ..numerics import EPS


@dataclass
class AggregationResult:
    risk: float
    token_risks: np.ndarray


_SIGNAL_ORDER = ["ent", "mlp", "psp", "spt", "spectral_gap"]
_SIGNAL_INDEX = {sig: i for i, sig in enumerate(_SIGNAL_ORDER)}
_SIGNAL_SET = set(_SIGNAL_ORDER)


class PromptAnchoredAggregator:

    @staticmethod
    def _binary_entropy(p):
        p = np.clip(p, EPS, 1 - EPS)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    @staticmethod
    def _pit_normalize(sorted_vals: np.ndarray, response_vals: np.ndarray) -> np.ndarray:
        n = len(sorted_vals)
        ranks = np.searchsorted(sorted_vals, response_vals, side='right').astype(float)
        return (ranks + 0.5) / (n + 1)

    @staticmethod
    def _adaptive_kappa(prompt_stats: dict) -> float:
        decisiveness = []
        for sig in ("psp", "mlp"):
            sv = prompt_stats[sig]["sorted_vals"]
            H = PromptAnchoredAggregator._binary_entropy(sv)
            decisiveness.append(float(np.median(1.0 - H)))
        return 1.0 + float(np.median(decisiveness))

    def compute_risk(
        self,
        prompt_stats: dict,
        response_signals: dict[str, np.ndarray],
        disabled_signals: set[str] | None = None,
    ) -> AggregationResult:
        available_signals = _SIGNAL_SET - (disabled_signals or set())
        ordered_sigs = [s for s in _SIGNAL_ORDER if s in available_signals]
        indices = np.array([_SIGNAL_INDEX[s] for s in ordered_sigs])

        n_tokens = len(response_signals[ordered_sigs[0]])

        P = np.column_stack([np.asarray(response_signals[s]) for s in ordered_sigs])
        for i, sig in enumerate(ordered_sigs):
            stats = prompt_stats[sig]
            if stats.get("mode") != "direct":
                P[:, i] = self._pit_normalize(stats["sorted_vals"], P[:, i])

        kappa = self._adaptive_kappa(prompt_stats)
        precision_matrix = prompt_stats["_cross_signal_precision"]
        omega = precision_matrix[np.ix_(indices, indices)]

        token_risks = self._entropy_gated_fusion(P, kappa, omega)

        risk = self._response_level_risk(
            response_signals, prompt_stats, ordered_sigs, kappa, omega,
        )

        return AggregationResult(risk=risk, token_risks=token_risks)

    def _entropy_gated_fusion(
        self,
        P: np.ndarray,
        kappa: float,
        omega: np.ndarray,
    ) -> np.ndarray:
        E = (1.0 - self._binary_entropy(P)) ** kappa
        W = np.maximum(E @ omega, 0.0)

        weighted_sum = np.sum(W * P, axis=1)
        weight_sum = np.sum(W, axis=1)
        return np.where(weight_sum > 0, weighted_sum / weight_sum, 0.5)

    def _response_level_risk(
        self,
        response_signals: dict[str, np.ndarray],
        prompt_stats: dict,
        ordered_sigs: list[str],
        kappa: float,
        omega: np.ndarray,
    ) -> float:
        probs = np.empty(len(ordered_sigs))
        for i, sig in enumerate(ordered_sigs):
            mean_val = float(np.mean(response_signals[sig]))
            stats = prompt_stats[sig]
            if stats.get("mode") == "direct":
                probs[i] = mean_val
            else:
                probs[i] = float(self._pit_normalize(stats["sorted_vals"], np.array([mean_val]))[0])

        E = (1.0 - self._binary_entropy(probs)) ** kappa
        W = np.maximum(E @ omega, 0.0)

        total_weight = W.sum()
        return float(np.sum(probs * W) / (total_weight + EPS))
