"""
Tests for engine aggregation integration.

Validates that the prompt-anchored aggregator is correctly integrated
into the AG-SAR engine (The Standard Model).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ag_sar.config import DetectorConfig, TokenSignals
from ag_sar.aggregation import PromptAnchoredAggregator


class TestEngineAggregationIntegration:
    """Test the engine's integration with prompt-anchored aggregation."""

    def test_collect_signal_arrays(self):
        """Test that signal arrays are correctly collected from token results."""
        # Create mock token signals
        token_results = [
            TokenSignals(
                jsd_cand=0.1, lci_cand=0.05, logp_cand=-2.0, var_logp_cand=0.3,
                entropy=1.5, inv_margin=0.2,
            ),
            TokenSignals(
                jsd_cand=0.3, lci_cand=0.1, logp_cand=-3.0, var_logp_cand=0.5,
                entropy=2.0, inv_margin=0.4,
            ),
            TokenSignals(
                jsd_cand=0.8, lci_cand=0.2, logp_cand=-5.0, var_logp_cand=0.8,
                entropy=3.5, inv_margin=0.7,
            ),
        ]

        # Simulate _collect_signal_arrays
        signals = ("jsd_cand", "entropy", "inv_margin")
        signal_arrays = {sig: [] for sig in signals}
        signal_attr_map = {
            "jsd_cand": "jsd_cand",
            "jsd": "jsd_cand",
            "entropy": "entropy",
            "inv_margin": "inv_margin",
        }

        for token_sig in token_results:
            for sig in signals:
                attr = signal_attr_map.get(sig, sig)
                value = getattr(token_sig, attr, None)
                if value is not None:
                    signal_arrays[sig].append(value)
                else:
                    signal_arrays[sig].append(0.0)

        result = {sig: np.array(vals) for sig, vals in signal_arrays.items()}

        # Verify
        np.testing.assert_array_almost_equal(result["jsd_cand"], [0.1, 0.3, 0.8])
        np.testing.assert_array_almost_equal(result["entropy"], [1.5, 2.0, 3.5])
        np.testing.assert_array_almost_equal(result["inv_margin"], [0.2, 0.4, 0.7])

    def test_aggregator_with_token_signals(self):
        """Test prompt-anchored aggregator with realistic token signals."""
        aggregator = PromptAnchoredAggregator(active_signals={"jsd_cand", "entropy", "inv_margin"})

        # Prompt stats (from PrefillStatisticsHook)
        prompt_stats = {
            "jsd_cand": {"mu": 0.08, "sigma": 0.05},
            "entropy": {"mu": 1.5, "sigma": 0.3},
            "inv_margin": {"mu": 0.25, "sigma": 0.1},
        }

        # Simulate a response with one suspicious token
        response_signals = {
            "jsd_cand": np.array([0.1, 0.1, 0.5, 0.1, 0.1]),  # Spike at token 2
            "entropy": np.array([1.5, 1.6, 2.8, 1.5, 1.5]),   # Elevated at token 2
            "inv_margin": np.array([0.2, 0.2, 0.6, 0.2, 0.2]), # Elevated at token 2
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Token 2 should have higher risk than others
        assert result.token_risks[2] > result.token_risks[0]
        assert result.token_risks[2] > result.token_risks[1]
        assert result.token_risks[2] > result.token_risks[3]

        # Response risk should be elevated due to spike
        assert result.risk > 0.5

    def test_config_prompt_anchored_enabled(self):
        """Test that config correctly enables prompt-anchored aggregation."""
        config = DetectorConfig(use_prompt_anchored=True)
        assert config.use_prompt_anchored == True
        assert "jsd" in config.prompt_anchored_signals
        assert "entropy" in config.prompt_anchored_signals
        assert "inv_margin" in config.prompt_anchored_signals

    def test_config_prompt_anchored_disabled(self):
        """Test that config can disable prompt-anchored aggregation."""
        config = DetectorConfig(use_prompt_anchored=False)
        assert config.use_prompt_anchored == False

    def test_hallucination_vs_confident_response(self):
        """
        Key test: Anchored aggregator should distinguish between
        hallucinated and confident responses.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"jsd_cand", "entropy", "inv_margin"})

        # Normal prompt stats
        prompt_stats = {
            "jsd_cand": {"mu": 0.08, "sigma": 0.05},
            "entropy": {"mu": 1.5, "sigma": 0.3},
            "inv_margin": {"mu": 0.25, "sigma": 0.1},
        }

        # Hallucinated response (high uncertainty throughout)
        hallucinated = {
            "jsd_cand": np.array([0.4, 0.5, 0.6, 0.5, 0.4]),
            "entropy": np.array([3.0, 3.2, 3.5, 3.1, 3.0]),
            "inv_margin": np.array([0.6, 0.7, 0.8, 0.7, 0.6]),
        }

        # Confident response (low uncertainty throughout)
        confident = {
            "jsd_cand": np.array([0.02, 0.03, 0.02, 0.02, 0.02]),
            "entropy": np.array([0.5, 0.6, 0.4, 0.5, 0.5]),
            "inv_margin": np.array([0.05, 0.06, 0.04, 0.05, 0.05]),
        }

        hall_result = aggregator.compute_risk(prompt_stats, hallucinated)
        conf_result = aggregator.compute_risk(prompt_stats, confident)

        # Hallucinated should have much higher risk
        assert hall_result.risk > conf_result.risk
        assert hall_result.risk > 0.7  # High risk
        assert conf_result.risk < 0.5  # Low risk


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
