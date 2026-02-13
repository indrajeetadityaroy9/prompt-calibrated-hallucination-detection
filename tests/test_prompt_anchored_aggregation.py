"""
Tests for Prompt-Anchored Aggregation (The Standard Model).

These tests validate the prompt-anchored normalization and Noisy-OR fusion.
No magic numbers - uses prompt statistics as the anchor point.
"""

import numpy as np
import pytest
from ag_sar.aggregation import PromptAnchoredAggregator, compute_prompt_statistics


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

        # Risk should be MEDIUM (0.4-0.7) - response is only slightly worse
        assert 0.4 < result.risk < 0.8, f"Expected medium risk, got {result.risk}"

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
        aggregator = PromptAnchoredAggregator(active_signals={"jsd", "entropy", "inv_margin"})

        prompt_stats = {
            "jsd": {"mu": 0.1, "sigma": 0.05},
            "entropy": {"mu": 1.0, "sigma": 0.2},
            "inv_margin": {"mu": 0.3, "sigma": 0.1},
        }

        # JSD spikes, others normal
        response_signals = {
            "jsd": np.array([0.1, 0.1, 0.8, 0.1, 0.1]),  # Spike at token 2
            "entropy": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            "inv_margin": np.array([0.3, 0.3, 0.3, 0.3, 0.3]),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Token 2 should have high risk
        assert result.token_risks[2] > 0.7, f"Expected high risk at spike token"

        # Response risk should be elevated
        assert result.risk > 0.5, f"Expected elevated response risk"

    def test_all_signals_agree_low(self):
        """If all signals are low, risk should be low."""
        aggregator = PromptAnchoredAggregator(active_signals={"jsd", "entropy"})

        prompt_stats = {
            "jsd": {"mu": 0.1, "sigma": 0.05},
            "entropy": {"mu": 1.0, "sigma": 0.3},
        }

        # Both signals lower than prompt
        response_signals = {
            "jsd": np.array([0.05] * 5),
            "entropy": np.array([0.5] * 5),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should be low risk
        assert result.risk < 0.5, f"Expected low risk, got {result.risk}"

    def test_all_signals_agree_high(self):
        """If all signals spike, risk should be very high."""
        aggregator = PromptAnchoredAggregator(active_signals={"jsd", "entropy"})

        prompt_stats = {
            "jsd": {"mu": 0.1, "sigma": 0.05},
            "entropy": {"mu": 1.0, "sigma": 0.3},
        }

        # Both signals much higher than prompt
        response_signals = {
            "jsd": np.array([0.5] * 5),
            "entropy": np.array([3.0] * 5),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk > 0.9, f"Expected very high risk, got {result.risk}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_response(self):
        """Single token responses should be handled."""
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        prompt_stats = {"entropy": {"mu": 0.5, "sigma": 0.1}}
        response_signals = {"entropy": np.array([2.0])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert len(result.token_risks) == 1
        assert result.risk > 0.5, "Should detect high entropy token"

    def test_empty_response(self):
        """Empty responses should return zero risk."""
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        prompt_stats = {"entropy": {"mu": 0.5, "sigma": 0.1}}
        response_signals = {"entropy": np.array([])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk == 0.0
        assert len(result.token_risks) == 0

    def test_no_matching_signals(self):
        """No matching signals should return zero risk."""
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        prompt_stats = {"jsd": {"mu": 0.1, "sigma": 0.05}}  # Wrong signal
        response_signals = {"jsd": np.array([0.5] * 5)}  # Wrong signal

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk == 0.0

    def test_eps_prevents_division_by_zero(self):
        """Zero sigma should use EPS to prevent division by zero."""
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Zero sigma - would cause division by zero without EPS
        prompt_stats = {"entropy": {"mu": 1.0, "sigma": 0.0}}
        response_signals = {"entropy": np.array([1.2, 1.2, 1.2])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should not crash and produce finite results
        assert 0 <= result.risk <= 1
        assert np.all(np.isfinite(result.z_scores["entropy"]))


class TestComputePromptStatistics:
    """Test the compute_prompt_statistics helper function."""

    def test_computes_mu_sigma(self):
        """Should compute mean and standard deviation."""
        signal_values = {
            "entropy": np.array([1.0, 1.2, 0.8, 1.1, 0.9]),
        }

        stats = compute_prompt_statistics(signal_values, tail_fraction=1.0)

        assert "entropy" in stats
        assert "mu" in stats["entropy"]
        assert "sigma" in stats["entropy"]
        assert abs(stats["entropy"]["mu"] - 1.0) < 0.01  # Mean should be ~1.0

    def test_tail_sampling(self):
        """Should use only tail of sequence."""
        # First half low, second half high
        signal_values = {
            "entropy": np.array([0.5] * 10 + [2.0] * 10),
        }

        # Use only last 50%
        stats = compute_prompt_statistics(signal_values, tail_fraction=0.5)

        # Mean should be ~2.0 (only uses second half)
        assert stats["entropy"]["mu"] > 1.5

    def test_empty_input(self):
        """Empty input should return empty stats."""
        stats = compute_prompt_statistics({})
        assert stats == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
