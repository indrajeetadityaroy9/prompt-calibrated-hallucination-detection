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
        assert config.response_flag_threshold == 0.5

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
