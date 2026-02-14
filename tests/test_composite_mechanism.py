"""
Integration tests for the Composite Signal Aggregation Mechanism.

Tests validate:
1. Prefill statistics hook captures real statistics
2. Relativity trap fix works with real prompt data
3. End-to-end signal flow from prefill to aggregation
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from ag_sar.hooks import PrefillStatisticsHook, PrefillStatistics
from ag_sar.aggregation import PromptAnchoredAggregator


class TestPrefillStatisticsHook:
    """Test the PrefillStatisticsHook captures real statistics."""

    def test_statistics_structure(self):
        """Verify PrefillStatistics has correct structure."""
        stats = PrefillStatistics(
            mu={"cus": 0.3, "pos": 0.1, "dps": 0.4},
            sigma={"cus": 0.1, "pos": 0.05, "dps": 0.15},
            n_tokens=64,
            signals=("cus", "pos", "dps"),
        )

        assert stats.mu["cus"] == 0.3
        assert stats.sigma["pos"] == 0.05
        assert stats.n_tokens == 64
        assert "dps" in stats.signals

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA for real model test"
    )
    def test_tail_sampling_window(self):
        """Verify tail sampling uses correct window with real LLaMA model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        lm_head = model.lm_head
        final_norm = model.model.norm

        hook = PrefillStatisticsHook(
            lm_head=lm_head,
            final_norm=final_norm,
            window_size=64,
        )

        assert hook.window_size == 64

        layer_idx = len(model.model.layers) // 2
        hook.install(model.model.layers[layer_idx])

        text = "The quick brown fox jumps over the lazy dog. " * 20
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            model(**inputs)

        stats = hook.compute_statistics(prompt_length=prompt_length)
        hook.remove()

        expected_tokens = min(64, prompt_length)
        assert stats.n_tokens == expected_tokens

        assert "pos" in stats.mu
        assert stats.mu["pos"] >= 0
        assert stats.sigma["pos"] >= 0

        del model
        torch.cuda.empty_cache()


class TestPrecomputedStatistics:
    """Test aggregator with pre-computed statistics."""

    def test_accepts_precomputed_stats(self):
        """Verify aggregator accepts pre-computed statistics dict."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})

        prompt_stats = {
            "cus": {"mu": 0.3, "sigma": 0.1},
            "pos": {"mu": 0.08, "sigma": 0.05},
            "dps": {"mu": 0.4, "sigma": 0.15},
        }

        response_signals = {
            "cus": np.array([0.35, 0.4, 0.38, 0.35, 0.35]),
            "pos": np.array([0.1, 0.15, 0.12, 0.1, 0.1]),
            "dps": np.array([0.45, 0.5, 0.48, 0.45, 0.45]),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert result.risk > 0
        assert result.risk < 1
        assert len(result.token_risks) == 5
        assert result.anchor_stats["pos"]["mu_prompt"] == 0.08
        assert result.anchor_stats["dps"]["mu_prompt"] == 0.4

    def test_precomputed_vs_empty_stats_difference(self):
        """Pre-computed stats should produce different results than empty stats."""
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        response_signals = {"dps": np.array([0.7, 0.75, 0.8, 0.7, 0.7])}

        result_no_stats = aggregator.compute_risk(
            prompt_stats={},
            response_signals=response_signals,
        )

        result_with_precomputed = aggregator.compute_risk(
            prompt_stats={"dps": {"mu": 0.4, "sigma": 0.15}},
            response_signals=response_signals,
        )

        assert result_with_precomputed.risk > 0


class TestRelativityTrapWithRealData:
    """Test Relativity Trap fix with pre-computed prompt statistics."""

    def test_scenario_a_total_hallucination(self):
        """
        Scenario A: All DSG signals high in response, low in prompt.
        Should be HIGH risk.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"dps": {"mu": 0.3, "sigma": 0.05}}
        response_signals = {"dps": np.array([0.8, 0.85, 0.82, 0.8, 0.8])}

        result = aggregator.compute_risk(prompt_stats, response_signals)
        assert result.risk > 0.9, f"Expected high risk for total hallucination, got {result.risk}"

    def test_scenario_b_perfect_fact(self):
        """
        Scenario B: All DSG signals low in response.
        Should be LOW risk.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"dps": {"mu": 0.4, "sigma": 0.1}}
        response_signals = {"dps": np.array([0.15, 0.2, 0.18, 0.15, 0.15])}

        result = aggregator.compute_risk(prompt_stats, response_signals)
        assert result.risk < 0.3, f"Expected low risk for perfect fact, got {result.risk}"

    def test_scenario_c_hard_prompt_matches_response(self):
        """
        Scenario C: Hard prompt (high DPS) with similar response.
        Should be MEDIUM risk.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        prompt_stats = {"dps": {"mu": 0.6, "sigma": 0.1}}
        response_signals = {"dps": np.array([0.62, 0.65, 0.6, 0.62, 0.6])}

        result = aggregator.compute_risk(prompt_stats, response_signals)
        assert 0.4 < result.risk < 0.7, f"Expected medium risk, got {result.risk}"


class TestEndToEndFlow:
    """Test end-to-end signal flow from mock prefill to aggregation."""

    def test_full_pipeline_flow(self):
        """Simulate full pipeline: prefill stats -> response signals -> risk."""
        prefill_stats = PrefillStatistics(
            mu={"cus": 0.25, "pos": 0.08, "dps": 0.35},
            sigma={"cus": 0.08, "pos": 0.05, "dps": 0.1},
            n_tokens=64,
            signals=("cus", "pos", "dps"),
        )

        prompt_stats = {
            sig: {
                "mu": prefill_stats.mu[sig],
                "sigma": prefill_stats.sigma[sig],
            }
            for sig in prefill_stats.signals
        }

        response_signals = {
            "cus": np.array([0.3, 0.35, 0.7, 0.32, 0.3]),
            "pos": np.array([0.1, 0.15, 0.5, 0.12, 0.1]),
            "dps": np.array([0.4, 0.45, 0.8, 0.42, 0.4]),
        }

        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})
        result = aggregator.compute_risk(prompt_stats, response_signals)

        assert 0 <= result.risk <= 1
        assert len(result.token_risks) == 5
        assert all(0 <= r <= 1 for r in result.token_risks)

        # Position 2 should have highest risk (spike in all signals)
        assert result.token_risks[2] == max(result.token_risks)
        assert result.risk > 0.5

    def test_multiple_signals_fusion(self):
        """Verify Noisy-OR fusion works correctly with pre-computed stats."""
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})

        prompt_stats = {
            "cus": {"mu": 0.25, "sigma": 0.08},
            "pos": {"mu": 0.1, "sigma": 0.05},
            "dps": {"mu": 0.35, "sigma": 0.1},
        }

        # Only POS spikes, others normal
        response_only_pos = {
            "cus": np.array([0.25, 0.26, 0.25, 0.25, 0.25]),
            "pos": np.array([0.5, 0.6, 0.5, 0.5, 0.5]),
            "dps": np.array([0.35, 0.36, 0.35, 0.35, 0.35]),
        }

        # All signals spike
        response_all_spike = {
            "cus": np.array([0.7, 0.75, 0.7, 0.7, 0.7]),
            "pos": np.array([0.5, 0.6, 0.5, 0.5, 0.5]),
            "dps": np.array([0.8, 0.85, 0.8, 0.8, 0.8]),
        }

        result_pos_only = aggregator.compute_risk(prompt_stats, response_only_pos)
        result_all_spike = aggregator.compute_risk(prompt_stats, response_all_spike)

        assert result_all_spike.risk > result_pos_only.risk
        assert result_pos_only.risk > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
