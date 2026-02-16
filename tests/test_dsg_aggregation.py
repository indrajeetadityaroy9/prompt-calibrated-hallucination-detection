"""
Tests for DSG aggregation integration.

Validates that the prompt-anchored aggregator works correctly
with DSG signals (CUS, POS, DPS).
"""

import numpy as np
import pytest

from ag_sar.config import DSGConfig, DSGTokenSignals
from ag_sar.aggregation import PromptAnchoredAggregator


class TestDSGAggregationIntegration:
    """Test the aggregator with DSG signals."""

    def test_collect_dsg_signal_arrays(self):
        """Test that DSG signal arrays are correctly collected from token results."""
        token_results = [
            DSGTokenSignals(cus=0.1, pos=0.2, dps=0.3),
            DSGTokenSignals(cus=0.4, pos=0.5, dps=0.6),
            DSGTokenSignals(cus=0.7, pos=0.8, dps=0.9),
        ]

        signal_arrays = {
            "cus": np.array([s.cus for s in token_results]),
            "pos": np.array([s.pos for s in token_results]),
            "dps": np.array([s.dps for s in token_results]),
        }

        np.testing.assert_array_almost_equal(signal_arrays["cus"], [0.1, 0.4, 0.7])
        np.testing.assert_array_almost_equal(signal_arrays["pos"], [0.2, 0.5, 0.8])
        np.testing.assert_array_almost_equal(signal_arrays["dps"], [0.3, 0.6, 0.9])

    def test_aggregator_with_dsg_signals(self):
        """Test prompt-anchored aggregator with DSG signals."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})

        prompt_stats = {
            "cus": {"mu": 0.3, "sigma": 0.1},
            "pos": {"mu": 0.2, "sigma": 0.1},
            "dps": {"mu": 0.4, "sigma": 0.15},
        }

        # Response with one suspicious token (token 2)
        response_signals = {
            "cus": np.array([0.3, 0.3, 0.8, 0.3, 0.3]),
            "pos": np.array([0.2, 0.2, 0.7, 0.2, 0.2]),
            "dps": np.array([0.4, 0.4, 0.9, 0.4, 0.4]),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Token 2 should have higher risk than others
        assert result.token_risks[2] > result.token_risks[0]
        assert result.token_risks[2] > result.token_risks[1]
        assert result.token_risks[2] > result.token_risks[3]

        # Response risk should be elevated
        assert result.risk > 0.5

    def test_dsg_config_defaults(self):
        """Test DSGConfig has correct defaults."""
        config = DSGConfig()
        assert config.layer_subset == "all"

    def test_hallucination_vs_confident_response(self):
        """
        Key test: Aggregator should distinguish between
        hallucinated and confident responses using DSG signals.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})

        prompt_stats = {
            "cus": {"mu": 0.3, "sigma": 0.1},
            "pos": {"mu": 0.2, "sigma": 0.1},
            "dps": {"mu": 0.4, "sigma": 0.15},
        }

        # Hallucinated response (high CUS, POS, DPS)
        hallucinated = {
            "cus": np.array([0.8, 0.85, 0.9, 0.85, 0.8]),
            "pos": np.array([0.7, 0.75, 0.8, 0.75, 0.7]),
            "dps": np.array([0.85, 0.9, 0.95, 0.9, 0.85]),
        }

        # Confident response (low CUS, POS, DPS)
        confident = {
            "cus": np.array([0.1, 0.12, 0.1, 0.11, 0.1]),
            "pos": np.array([0.05, 0.06, 0.04, 0.05, 0.05]),
            "dps": np.array([0.2, 0.22, 0.19, 0.2, 0.2]),
        }

        hall_result = aggregator.compute_risk(prompt_stats, hallucinated)
        conf_result = aggregator.compute_risk(prompt_stats, confident)

        # Hallucinated should have much higher risk
        assert hall_result.risk > conf_result.risk
        assert hall_result.risk > 0.7
        assert conf_result.risk < 0.5


class TestResponseLevelMeanShift:
    """
    Test that response-level aggregation captures diffuse mean shifts.

    This is the key regression test: real hallucination signal is a slight
    elevation of POS/DPS across ALL tokens, not per-token spikes. The old
    adaptive quantile approach missed this (AUROC=0.50). The new signal-first
    aggregation should capture it.
    """

    def test_diffuse_pos_shift_detected(self):
        """POS with diffuse mean shift should produce different response risks."""
        aggregator = PromptAnchoredAggregator(
            active_signals={"cus", "pos", "dps", "dola", "cgd"}
        )
        prompt_stats = {
            "cus": {"mode": "direct"},
            "pos": {"mu": 0.3, "sigma": 0.1},
            "dps": {"mu": 0.5, "sigma": 0.2},
            "dola": {"mu": 5.0, "sigma": 2.0},
            "cgd": {"mu": 0.5, "sigma": 0.2},
        }

        n = 30
        # Hallucinated: POS and DPS slightly elevated across ALL tokens
        hall_signals = {
            "cus": np.full(n, 0.5),
            "pos": np.full(n, 0.36),   # slightly above mu=0.3
            "dps": np.full(n, 0.55),   # slightly above mu=0.5
            "dola": np.full(n, 5.0),
            "cgd": np.full(n, 0.5),
        }
        # Non-hallucinated: POS and DPS slightly below
        good_signals = {
            "cus": np.full(n, 0.5),
            "pos": np.full(n, 0.24),   # slightly below mu=0.3
            "dps": np.full(n, 0.45),   # slightly below mu=0.5
            "dola": np.full(n, 5.0),
            "cgd": np.full(n, 0.5),
        }

        hall_result = aggregator.compute_risk(prompt_stats, hall_signals)
        good_result = aggregator.compute_risk(prompt_stats, good_signals)

        assert hall_result.risk > good_result.risk, (
            f"Hallucinated risk {hall_result.risk:.4f} should exceed "
            f"non-hallucinated {good_result.risk:.4f}"
        )

    def test_uninformative_signals_neutral(self):
        """Signals at their prompt mean should produce neutral response risk."""
        aggregator = PromptAnchoredAggregator(
            active_signals={"pos", "dps"}
        )
        prompt_stats = {
            "pos": {"mu": 0.3, "sigma": 0.1},
            "dps": {"mu": 0.5, "sigma": 0.2},
        }

        # All signals at their prompt means
        neutral = {
            "pos": np.full(10, 0.3),
            "dps": np.full(10, 0.5),
        }
        result = aggregator.compute_risk(prompt_stats, neutral)

        # Risk should be near 0.5 (neutral)
        assert 0.45 <= result.risk <= 0.55, f"Expected near 0.5, got {result.risk}"

    def test_signal_response_probs_populated(self):
        """AggregationResult should contain per-signal response probabilities."""
        aggregator = PromptAnchoredAggregator(active_signals={"pos", "dps"})
        prompt_stats = {
            "pos": {"mu": 0.3, "sigma": 0.1},
            "dps": {"mu": 0.5, "sigma": 0.2},
        }
        response_signals = {
            "pos": np.full(5, 0.4),
            "dps": np.full(5, 0.6),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert "pos" in result.signal_response_probs
        assert "dps" in result.signal_response_probs
        # POS mean=0.4, z=(0.4-0.3)/0.1=1.0, sigmoid(1.0)≈0.731
        assert result.signal_response_probs["pos"] > 0.5
        # DPS mean=0.6, z=(0.6-0.5)/0.2=0.5, sigmoid(0.5)≈0.622
        assert result.signal_response_probs["dps"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
