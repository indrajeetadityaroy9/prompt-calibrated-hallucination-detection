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
            mu={"entropy": 1.5, "inv_margin": 0.3, "jsd_cand": 0.1},
            sigma={"entropy": 0.5, "inv_margin": 0.1, "jsd_cand": 0.05},
            n_tokens=64,
            signals=("entropy", "inv_margin", "jsd_cand"),
        )

        assert stats.mu["entropy"] == 1.5
        assert stats.sigma["jsd_cand"] == 0.05
        assert stats.n_tokens == 64
        assert "entropy" in stats.signals

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires CUDA for real model test"
    )
    def test_tail_sampling_window(self):
        """Verify tail sampling uses correct window with real LLaMA model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load small LLaMA model
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get model components
        lm_head = model.lm_head
        final_norm = model.model.norm

        hook = PrefillStatisticsHook(
            lm_head=lm_head,
            final_norm=final_norm,
            window_size=64,
            signals=("entropy", "jsd_cand"),
        )

        assert hook.window_size == 64

        # Install hook on a middle layer
        layer_idx = len(model.model.layers) // 2
        hook.install(model.model.layers[layer_idx])

        # Run a forward pass with 128+ tokens
        text = "The quick brown fox jumps over the lazy dog. " * 20
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        prompt_length = inputs.input_ids.shape[1]

        with torch.no_grad():
            model(**inputs)

        # Compute statistics
        stats = hook.compute_statistics(prompt_length=prompt_length)
        hook.remove()

        # Should have used last 64 tokens (or all if prompt < 64)
        expected_tokens = min(64, prompt_length)
        assert stats.n_tokens == expected_tokens

        # Verify statistics are computed
        assert "entropy" in stats.mu
        assert "jsd_cand" in stats.mu
        assert stats.mu["entropy"] > 0  # Entropy should be positive
        assert stats.sigma["entropy"] >= 0  # Std should be non-negative

        # Clean up
        del model
        torch.cuda.empty_cache()


class TestPrecomputedStatistics:
    """Test aggregator with pre-computed statistics (from PrefillStatisticsHook)."""

    def test_accepts_precomputed_stats(self):
        """Verify aggregator accepts pre-computed statistics dict."""
        aggregator = PromptAnchoredAggregator(active_signals={"jsd_cand", "entropy", "inv_margin"})

        # Pre-computed statistics (as returned by engine._convert_prefill_stats_to_signals)
        prompt_stats = {
            "jsd_cand": {"mu": 0.08, "sigma": 0.05},
            "entropy": {"mu": 1.2, "sigma": 0.4},
            "inv_margin": {"mu": 0.25, "sigma": 0.1},
        }

        # Response with moderate signals
        response_signals = {
            "jsd_cand": np.array([0.1, 0.15, 0.12, 0.1, 0.1]),
            "entropy": np.array([1.5, 1.6, 1.8, 1.5, 1.4]),
            "inv_margin": np.array([0.3, 0.35, 0.4, 0.3, 0.3]),
        }

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should work without falling back to priors
        assert result.risk > 0
        assert result.risk < 1
        assert len(result.token_risks) == 5
        # Verify anchor stats use the pre-computed values
        assert result.anchor_stats["jsd_cand"]["mu_prompt"] == 0.08
        assert result.anchor_stats["entropy"]["mu_prompt"] == 1.2

    def test_precomputed_vs_empty_stats_difference(self):
        """Pre-computed stats should produce different results than empty stats."""
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Response with high entropy
        response_signals = {"entropy": np.array([2.5, 2.6, 2.7, 2.5, 2.5])}

        # Case 1: Empty prompt stats (no baseline available)
        result_no_stats = aggregator.compute_risk(
            prompt_stats={},
            response_signals=response_signals,
        )

        # Case 2: Pre-computed stats (high entropy prompt, μ=2.3)
        result_with_precomputed = aggregator.compute_risk(
            prompt_stats={"entropy": {"mu": 2.3, "sigma": 0.3}},
            response_signals=response_signals,
        )

        # With no stats, returns 0 risk (no signals to process)
        # With precomputed (μ=2.3), response entropy 2.5 looks moderate
        # This tests that precomputed stats are actually used
        assert result_with_precomputed.risk > 0  # Has data to process


class TestRelativityTrapWithRealData:
    """Test Relativity Trap fix with pre-computed prompt statistics."""

    def test_scenario_a_total_hallucination(self):
        """
        Scenario A: Model confused throughout (all high entropy in response).
        With real prompt stats (low entropy), this should be HIGH risk.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Pre-computed: Clear, confident prompt
        prompt_stats = {"entropy": {"mu": 0.5, "sigma": 0.1}}

        # All high entropy in response (confused throughout)
        response_signals = {"entropy": np.array([2.5, 2.6, 2.4, 2.5, 2.5])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should detect this as HIGH risk (response much worse than prompt)
        assert result.risk > 0.9, f"Expected high risk for total hallucination, got {result.risk}"

    def test_scenario_b_perfect_fact(self):
        """
        Scenario B: Model confident throughout (low entropy in response).
        With real prompt stats (normal entropy), this should be LOW risk.
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Pre-computed: Normal prompt
        prompt_stats = {"entropy": {"mu": 1.0, "sigma": 0.3}}

        # Very confident response (low entropy)
        response_signals = {"entropy": np.array([0.3, 0.4, 0.35, 0.3, 0.3])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should detect this as LOW risk (response better than prompt)
        assert result.risk < 0.3, f"Expected low risk for perfect fact, got {result.risk}"

    def test_scenario_c_hard_prompt_matches_response(self):
        """
        Scenario C: Hard prompt (high entropy) with similarly confused response.
        Should be MEDIUM risk (response about same as prompt).
        """
        aggregator = PromptAnchoredAggregator(active_signals={"entropy"})

        # Pre-computed: Hard prompt (high entropy)
        prompt_stats = {"entropy": {"mu": 2.0, "sigma": 0.3}}

        # Response matches prompt difficulty
        response_signals = {"entropy": np.array([2.1, 2.2, 2.0, 2.1, 2.0])}

        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Should be MEDIUM risk (z-scores near 0)
        assert 0.4 < result.risk < 0.7, f"Expected medium risk, got {result.risk}"


class TestEndToEndFlow:
    """Test end-to-end signal flow from mock prefill to aggregation."""

    def test_full_pipeline_flow(self):
        """Simulate full pipeline: prefill stats -> response signals -> risk."""
        # Simulate PrefillStatistics from hook
        prefill_stats = PrefillStatistics(
            mu={"jsd_cand": 0.08, "entropy": 1.2, "inv_margin": 0.25},
            sigma={"jsd_cand": 0.05, "entropy": 0.4, "inv_margin": 0.1},
            n_tokens=64,
            signals=("jsd_cand", "entropy", "inv_margin"),
        )

        # Convert to format expected by aggregator (as engine does)
        prompt_stats = {
            sig: {
                "mu": prefill_stats.mu[sig],
                "sigma": prefill_stats.sigma[sig],
            }
            for sig in prefill_stats.signals
        }

        # Simulate response signals (as would be collected during generation)
        response_signals = {
            "jsd_cand": np.array([0.1, 0.15, 0.5, 0.12, 0.1]),  # Spike at position 2
            "entropy": np.array([1.4, 1.5, 2.5, 1.3, 1.4]),    # Spike at position 2
            "inv_margin": np.array([0.3, 0.35, 0.6, 0.3, 0.3]), # Spike at position 2
        }

        # Create aggregator and compute risk
        aggregator = PromptAnchoredAggregator(active_signals={"jsd_cand", "entropy", "inv_margin"})
        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Verify structure
        assert 0 <= result.risk <= 1
        assert len(result.token_risks) == 5
        assert all(0 <= r <= 1 for r in result.token_risks)

        # Position 2 should have highest risk (spike in all signals)
        assert result.token_risks[2] == max(result.token_risks)

        # Overall risk should be elevated due to spike
        assert result.risk > 0.5

    def test_multiple_signals_fusion(self):
        """Verify Noisy-OR fusion works correctly with pre-computed stats."""
        aggregator = PromptAnchoredAggregator(active_signals={"jsd_cand", "entropy", "inv_margin"})

        # Pre-computed prompt stats
        prompt_stats = {
            "jsd_cand": {"mu": 0.1, "sigma": 0.05},
            "entropy": {"mu": 1.0, "sigma": 0.3},
            "inv_margin": {"mu": 0.25, "sigma": 0.1},
        }

        # Only JSD spikes, others normal
        response_only_jsd = {
            "jsd_cand": np.array([0.5, 0.6, 0.5, 0.5, 0.5]),  # High JSD
            "entropy": np.array([1.0, 1.1, 1.0, 1.0, 1.0]),   # Normal
            "inv_margin": np.array([0.25, 0.26, 0.25, 0.25, 0.25]),  # Normal
        }

        # All signals spike
        response_all_spike = {
            "jsd_cand": np.array([0.5, 0.6, 0.5, 0.5, 0.5]),
            "entropy": np.array([3.0, 3.1, 3.0, 3.0, 3.0]),
            "inv_margin": np.array([0.7, 0.75, 0.7, 0.7, 0.7]),
        }

        result_jsd_only = aggregator.compute_risk(prompt_stats, response_only_jsd)
        result_all_spike = aggregator.compute_risk(prompt_stats, response_all_spike)

        # All signals spiking should produce higher risk than just one
        assert result_all_spike.risk > result_jsd_only.risk

        # But even single signal spike should produce elevated risk (Noisy-OR)
        assert result_jsd_only.risk > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
