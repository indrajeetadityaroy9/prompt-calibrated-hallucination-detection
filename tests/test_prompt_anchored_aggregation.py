"""
Tests for Prompt-Anchored Aggregation.

These tests validate the prompt-anchored normalization and Noisy-OR fusion.
No magic numbers - uses prompt statistics as the anchor point.
"""

import numpy as np
import pytest
from ag_sar.aggregation import PromptAnchoredAggregator


class TestRelativityTrapFix:
    """Test cases for the Relativity Trap scenarios."""

    def test_scenario_a_total_hallucination(self):
        """
        Scenario A: Model confused throughout (all high entropy).

        OLD METHOD would give low risk because within-response z-scores are small.
        NEW METHOD should give HIGH risk because entropy is much higher than prompt.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Clear prompt stats (low entropy)
        prompt_stats = {"entropy": {"mu": 0.5, "sigma": 0.1}}

        # Confused response (all high entropy)
        response_signals = {"entropy": np.array([2.0, 2.1, 1.9, 2.0, 2.0])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Risk should be HIGH (> 0.9) because response entropy is much higher than prompt
        assert result.risk > 0.9, f"Expected high risk for total hallucination, got {result.risk}"

        # Z-scores should be large positive
        avg_z = np.mean(result.z_scores["entropy"])
        assert avg_z > 5, f"Expected large positive z-scores, got {avg_z}"

    def test_scenario_b_perfect_fact(self):
        """
        Scenario B: Model confident throughout (very low entropy).

        Should give LOW risk because entropy is lower than prompt.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Normal prompt stats
        prompt_stats = {"entropy": {"mu": 0.5, "sigma": 0.1}}

        # Very confident response (low entropy)
        response_signals = {"entropy": np.array([0.01, 0.02, 0.05, 0.01, 0.01])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Risk should be LOW (< 0.3) because response is more confident than prompt
        assert result.risk < 0.3, f"Expected low risk for perfect fact, got {result.risk}"

        # Z-scores should be negative (response better than prompt)
        avg_z = np.mean(result.z_scores["entropy"])
        assert avg_z < -1, f"Expected negative z-scores, got {avg_z}"

    def test_scenario_c_hard_prompt_confused_response(self):
        """
        Scenario C: Hard prompt (high entropy) with confused response.

        Risk should be MEDIUM because response is only slightly worse than prompt.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Hard prompt stats (high entropy)
        prompt_stats = {"entropy": {"mu": 2.0, "sigma": 0.1}}

        # Confused response (similar high entropy)
        response_signals = {"entropy": np.array([2.1, 2.2, 2.0, 2.1, 2.1])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Risk should be MEDIUM-HIGH (0.4-0.9) - response is slightly worse, p90 aggregation
        assert 0.4 < result.risk < 0.9, f"Expected medium risk, got {result.risk}"

    def test_scenario_d_hard_prompt_confident_response(self):
        """
        Scenario D: Hard prompt but model gives confident response.

        Risk should be LOW - model "figured it out" despite hard prompt.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Hard prompt stats
        prompt_stats = {"entropy": {"mu": 2.0, "sigma": 0.3}}

        # Confident response
        response_signals = {"entropy": np.array([0.3, 0.4, 0.3, 0.35, 0.32])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Risk should be LOW
        assert result.risk < 0.3, f"Expected low risk, got {result.risk}"


class TestNoisyORFusion:
    """Test Noisy-OR fusion of multiple signals."""

    def test_single_signal_spike(self):
        """If one signal spikes, risk should be high."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})

        prompt_stats = {
            "cus": {"mu": 0.1, "sigma": 0.05},
            "pos": {"mu": 0.1, "sigma": 0.05},
            "dps": {"mu": 0.3, "sigma": 0.1},
        }

        # CUS spikes, others normal
        response_signals = {
            "cus": np.array([0.1, 0.1, 0.8, 0.1, 0.1]),  # Spike at token 2
            "pos": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            "dps": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Token 2 should have high risk
        assert result.token_risks[2] > 0.7, f"Expected high risk at spike token"

        # Response risk should be elevated
        assert result.risk > 0.5, f"Expected elevated response risk"

    def test_all_signals_agree_low(self):
        """If all signals are low, risk should be low."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos"})

        prompt_stats = {
            "cus": {"mu": 0.3, "sigma": 0.1},
            "pos": {"mu": 0.2, "sigma": 0.1},
        }

        # Both signals lower than prompt
        response_signals = {
            "cus": np.array([0.1] * 5),
            "pos": np.array([0.05] * 5),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should be low risk
        assert result.risk < 0.5, f"Expected low risk, got {result.risk}"

    def test_all_signals_agree_high(self):
        """If all signals spike, risk should be very high."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos"})

        prompt_stats = {
            "cus": {"mu": 0.1, "sigma": 0.05},
            "pos": {"mu": 0.1, "sigma": 0.05},
        }

        # Both signals much higher than prompt
        response_signals = {
            "cus": np.array([0.8] * 5),
            "pos": np.array([0.7] * 5),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk > 0.9, f"Expected very high risk, got {result.risk}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_response(self):
        """Single token responses should be handled."""
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"dps": {"mu": 0.3, "sigma": 0.1}}
        response_signals = {"dps": np.array([0.9])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert len(result.token_risks) == 1
        assert result.risk > 0.5, "Should detect high DPS token"

    def test_empty_response(self):
        """Empty responses should return zero risk."""
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"dps": {"mu": 0.3, "sigma": 0.1}}
        response_signals = {"dps": np.array([])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk == 0.0
        assert len(result.token_risks) == 0

    def test_no_matching_signals(self):
        """No matching signals should return zero risk."""
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"cus": {"mu": 0.1, "sigma": 0.05}}  # Wrong signal
        response_signals = {"cus": np.array([0.5] * 5)}  # Wrong signal

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk == 0.0

    def test_eps_prevents_division_by_zero(self):
        """Zero sigma should use EPS to prevent division by zero."""
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        # Zero sigma - would cause division by zero without EPS
        prompt_stats = {"dps": {"mu": 0.5, "sigma": 0.0}}
        response_signals = {"dps": np.array([0.6, 0.6, 0.6])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should not crash and produce finite results
        assert 0 <= result.risk <= 1
        assert np.all(np.isfinite(result.z_scores["dps"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
