"""
Tests for the signal metadata registry and calibration utilities.

Validates that:
- Registry has correct metadata for each signal
- Fallback stats follow bounded/unbounded rules
- Sigma floors are data-derived
- Shared calibration utilities work correctly
"""

import numpy as np
import pytest

from ag_sar.config import (
    NormMode,
    SignalMetadata,
    SIGNAL_REGISTRY,
    DSGConfig,
)
from ag_sar.calibration import get_layer_indices, adaptive_window


class TestSignalRegistry:
    """Test signal registry contents and properties."""

    def test_all_five_signals_registered(self):
        """All 5 DSG signals must be in the registry."""
        assert set(SIGNAL_REGISTRY.keys()) == {"cus", "pos", "dps", "dola", "cgd"}

    def test_cus_is_direct(self):
        """CUS uses direct mode (value IS the probability)."""
        assert SIGNAL_REGISTRY["cus"].norm_mode == NormMode.DIRECT

    def test_zscore_signals(self):
        """POS, DPS, DoLa, CGD use z-score normalization."""
        for sig in ["pos", "dps", "dola", "cgd"]:
            assert SIGNAL_REGISTRY[sig].norm_mode == NormMode.ZSCORE

    def test_bounded_signals(self):
        """CUS, POS, DPS, CGD are bounded [0,1]. DoLa is unbounded."""
        for sig in ["cus", "pos", "dps", "cgd"]:
            assert SIGNAL_REGISTRY[sig].bounded is True, f"{sig} should be bounded"
        assert SIGNAL_REGISTRY["dola"].bounded is False

    def test_all_higher_is_riskier(self):
        """All signals have higher_is_riskier=True."""
        for sig, meta in SIGNAL_REGISTRY.items():
            assert meta.higher_is_riskier is True

    def test_neutral_values(self):
        """Neutral values are correct."""
        assert SIGNAL_REGISTRY["cus"].neutral == 0.5
        assert SIGNAL_REGISTRY["pos"].neutral == 0.0
        assert SIGNAL_REGISTRY["dps"].neutral == 0.5
        assert SIGNAL_REGISTRY["dola"].neutral == 0.0
        assert SIGNAL_REGISTRY["cgd"].neutral == 0.5


class TestFallbackStats:
    """Test registry-derived fallback statistics."""

    def test_bounded_signal_falls_back_to_direct(self):
        """Bounded signals fall back to direct mode (no model-specific constants)."""
        for sig in ["cus", "pos", "dps", "cgd"]:
            stats = SIGNAL_REGISTRY[sig].fallback_stats()
            assert stats.get("mode") == "direct", f"{sig} fallback should be direct mode"
            assert "mu" not in stats, f"{sig} direct fallback should not have mu"

    def test_unbounded_signal_falls_back_to_wide_sigma(self):
        """Unbounded signals fall back to neutral + wide sigma."""
        stats = SIGNAL_REGISTRY["dola"].fallback_stats()
        assert stats["mu"] == 0.0
        assert stats["sigma"] == 1.0
        assert "mode" not in stats

    def test_no_model_specific_values(self):
        """No fallback contains model-specific values like mu=5.0 or sigma=3.0."""
        for sig, meta in SIGNAL_REGISTRY.items():
            stats = meta.fallback_stats()
            if "mu" in stats:
                assert stats["mu"] == meta.neutral
            if "sigma" in stats:
                assert stats["sigma"] == 1.0  # Wide/uninformative


class TestSigmaFloor:
    """Test data-derived sigma floor computation."""

    def test_sigma_floor_is_10_percent(self):
        """Sigma floor = 10% of observed sigma."""
        meta = SIGNAL_REGISTRY["dps"]
        assert meta.sigma_floor(1.0) == pytest.approx(0.1)
        assert meta.sigma_floor(0.5) == pytest.approx(0.05)

    def test_sigma_floor_with_zero_sigma(self):
        """Zero observed sigma should produce small positive floor."""
        meta = SIGNAL_REGISTRY["dps"]
        floor = meta.sigma_floor(0.0)
        assert floor == 0.01

    def test_sigma_floor_with_tiny_sigma(self):
        """Very small sigma should produce proportional floor."""
        meta = SIGNAL_REGISTRY["pos"]
        floor = meta.sigma_floor(0.001)
        assert floor == pytest.approx(0.0001)


class TestSharedUtilities:
    """Test shared calibration utilities."""

    def test_get_layer_indices_all(self):
        """layer_subset='all' returns all layers."""
        config = DSGConfig(layer_subset="all")
        indices = get_layer_indices(config, 32)
        assert indices == list(range(32))

    def test_get_layer_indices_last_third(self):
        """layer_subset='last_third' returns last third."""
        config = DSGConfig(layer_subset="last_third")
        indices = get_layer_indices(config, 30)
        assert indices == list(range(20, 30))

    def test_get_layer_indices_last_quarter(self):
        """layer_subset='last_quarter' returns last quarter."""
        config = DSGConfig(layer_subset="last_quarter")
        indices = get_layer_indices(config, 32)
        assert indices == list(range(24, 32))

    def test_get_layer_indices_explicit(self):
        """layer_subset as list returns those exact layers."""
        config = DSGConfig(layer_subset=[5, 10, 15])
        indices = get_layer_indices(config, 32)
        assert indices == [5, 10, 15]

    def test_adaptive_window_short_prompt(self):
        """Short prompts get minimum 16 window."""
        assert adaptive_window(20) == 16

    def test_adaptive_window_medium_prompt(self):
        """Medium prompts: sqrt(n)."""
        w = adaptive_window(256)
        assert w == 16  # sqrt(256) = 16

    def test_adaptive_window_long_prompt(self):
        """Long prompts: capped at prompt_len//2."""
        w = adaptive_window(1000)
        assert w == 32  # sqrt(1000) ≈ 32


class TestAggregatorDirectMode:
    """Test that aggregator correctly handles direct mode from registry fallback."""

    def test_direct_mode_from_fallback_stats(self):
        """When a bounded signal calibration fails, fallback to direct mode."""
        from ag_sar.aggregation.prompt_anchored import PromptAnchoredAggregator

        aggregator = PromptAnchoredAggregator(active_signals={"dps"})

        # Simulate fallback: DPS falls back to direct mode
        prompt_stats = {"dps": SIGNAL_REGISTRY["dps"].fallback_stats()}
        assert prompt_stats["dps"]["mode"] == "direct"

        response_signals = {"dps": np.array([0.8, 0.85, 0.9])}
        result = aggregator.compute_risk(prompt_stats, response_signals)

        # Direct mode: risk ≈ mean(values) since they're treated as probabilities
        assert result.risk > 0.7

    def test_unbounded_fallback_uses_zscore(self):
        """When DoLa calibration fails, fallback to wide z-score."""
        from ag_sar.aggregation.prompt_anchored import PromptAnchoredAggregator

        aggregator = PromptAnchoredAggregator(active_signals={"dola"})

        # Simulate fallback: DoLa mu=0.0, sigma=1.0 (wide/uninformative)
        prompt_stats = {"dola": SIGNAL_REGISTRY["dola"].fallback_stats()}
        assert "mode" not in prompt_stats["dola"]

        # Values near neutral (0.0) with wide sigma → near 0.5 probability
        response_signals = {"dola": np.array([0.1, 0.2, 0.15])}
        result = aggregator.compute_risk(prompt_stats, response_signals)

        # With sigma=1.0 and mean≈0.15, z≈0.15, sigmoid(0.15)≈0.537
        assert 0.45 < result.risk < 0.65


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
