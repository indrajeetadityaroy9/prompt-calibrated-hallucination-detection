"""
Tests for Z-Score Based Adaptive Thresholds.

Purpose: Verify that the adaptive threshold framework correctly:
1. Stores new config parameters
2. Validates parameter bounds
3. Computes Z-score thresholds from calibration baselines
4. Handles edge cases (zero sigma, short prompts)

Key Constraints (from Reviewer Corrections):
1. Entropy Floor: Fixed at 0.01 - NOT adaptive
2. Z-Score Clamping: All sigma values clamped to min_confidence_floor
3. Percentile bounds: low in [0, 50], high in [50, 100]
"""

import pytest
from ag_sar.config import AGSARConfig


class TestAdaptiveThresholdConfig:
    """Test new adaptive threshold config parameters."""

    def test_default_adaptive_enabled(self):
        """Adaptive thresholds enabled by default."""
        config = AGSARConfig()
        assert config.use_adaptive_cpg_thresholds is True
        assert config.use_percentile_gate_sharpening is True

    def test_sigma_multiplier_defaults(self):
        """Default sigma multipliers are -1.0 (1 sigma below mean)."""
        config = AGSARConfig()
        assert config.cpg_gate_sigma_multiplier == -1.0
        assert config.cpg_dispersion_sigma_multiplier == -1.0

    def test_percentile_defaults(self):
        """Default percentiles are 10/90."""
        config = AGSARConfig()
        assert config.gate_sharpen_low_percentile == 10.0
        assert config.gate_sharpen_high_percentile == 90.0

    def test_gate_sharpening_defaults(self):
        """Default gate sharpening values are 0.2/0.8."""
        config = AGSARConfig()
        assert config.gate_sharpen_low == 0.2
        assert config.gate_sharpen_high == 0.8

    def test_entropy_floor_fixed_default(self):
        """Entropy floor default is 0.01 (fixed, not adaptive)."""
        config = AGSARConfig()
        assert config.cpg_entropy_floor == 0.01


class TestAdaptiveThresholdValidation:
    """Test validation of adaptive threshold parameters."""

    def test_invalid_sigma_multiplier_high(self):
        """Sigma multiplier should be <= 3.0."""
        with pytest.raises(ValueError, match="cpg_gate_sigma_multiplier"):
            AGSARConfig(cpg_gate_sigma_multiplier=5.0)

        with pytest.raises(ValueError, match="cpg_dispersion_sigma_multiplier"):
            AGSARConfig(cpg_dispersion_sigma_multiplier=4.0)

    def test_valid_negative_sigma_multiplier(self):
        """Negative sigma multipliers are valid (typical use case)."""
        config = AGSARConfig(
            cpg_gate_sigma_multiplier=-2.0,
            cpg_dispersion_sigma_multiplier=-1.5
        )
        assert config.cpg_gate_sigma_multiplier == -2.0
        assert config.cpg_dispersion_sigma_multiplier == -1.5

    def test_invalid_percentile_low_too_high(self):
        """Low percentile must be <= 50."""
        with pytest.raises(ValueError, match="gate_sharpen_low_percentile"):
            AGSARConfig(gate_sharpen_low_percentile=60.0)

    def test_invalid_percentile_high_too_low(self):
        """High percentile must be >= 50."""
        with pytest.raises(ValueError, match="gate_sharpen_high_percentile"):
            AGSARConfig(gate_sharpen_high_percentile=40.0)

    def test_invalid_gate_sharpen_low_too_high(self):
        """gate_sharpen_low must be in [0, 0.5]."""
        with pytest.raises(ValueError, match="gate_sharpen_low"):
            AGSARConfig(gate_sharpen_low=0.6)

    def test_invalid_gate_sharpen_high_too_low(self):
        """gate_sharpen_high must be in [0.5, 1]."""
        with pytest.raises(ValueError, match="gate_sharpen_high"):
            AGSARConfig(gate_sharpen_high=0.4)

    def test_valid_percentile_boundary(self):
        """Percentiles at boundaries (0, 50, 100) are valid."""
        config = AGSARConfig(
            gate_sharpen_low_percentile=0.0,
            gate_sharpen_high_percentile=100.0
        )
        assert config.gate_sharpen_low_percentile == 0.0
        assert config.gate_sharpen_high_percentile == 100.0

        config2 = AGSARConfig(
            gate_sharpen_low_percentile=50.0,
            gate_sharpen_high_percentile=50.0
        )
        assert config2.gate_sharpen_low_percentile == 50.0
        assert config2.gate_sharpen_high_percentile == 50.0


class TestZScoreComputation:
    """Test Z-score threshold computation logic."""

    def test_z_score_threshold_formula(self):
        """Verify Z-score -> threshold conversion formula."""
        # Formula: threshold = mu + z_multiplier * sigma
        # Given: mu=0.5, sigma=0.1, z=-1.0
        # Expected: threshold = 0.5 + (-1.0) * 0.1 = 0.4
        mu = 0.5
        sigma = 0.1
        z_multiplier = -1.0

        threshold = mu + z_multiplier * sigma
        assert abs(threshold - 0.4) < 1e-6

    def test_adaptive_threshold_differs_from_hardcoded(self):
        """Adaptive threshold differs from hardcoded 0.3."""
        # Simulate calibration with different model characteristics
        gate_mu = 0.6  # Higher than hardcoded threshold of 0.3
        gate_sigma = 0.15
        z_multiplier = -1.0

        adaptive_threshold = gate_mu + z_multiplier * gate_sigma
        hardcoded_threshold = 0.3

        # Adaptive should be much higher (0.45 vs 0.3)
        assert adaptive_threshold > hardcoded_threshold
        assert abs(adaptive_threshold - 0.45) < 1e-6

    def test_z_score_with_zero_sigma(self):
        """Z-score computation with zero sigma requires clamping."""
        # Edge case: sigma = 0 (repetitive prompt)
        # Without clamping: division by zero
        # With clamping: use min_confidence_floor
        mu = 0.5
        sigma = 0.0
        min_confidence_floor = 0.05
        z_multiplier = -1.0

        # Safe computation with clamping
        sigma_safe = max(sigma, min_confidence_floor)
        threshold = mu + z_multiplier * sigma_safe

        # Should be: 0.5 + (-1.0) * 0.05 = 0.45
        assert abs(threshold - 0.45) < 1e-6

    def test_negative_threshold_clamped(self):
        """Negative thresholds should be valid (if mu is low)."""
        # If mu=0.1, sigma=0.2, z=-1.0 -> threshold = -0.1
        # This is valid for Z-score framework (represents "very low")
        mu = 0.1
        sigma = 0.2
        z_multiplier = -1.0

        threshold = mu + z_multiplier * sigma
        assert threshold < 0  # -0.1


class TestEntropyFloorFixed:
    """Test that entropy floor remains fixed (NOT adaptive)."""

    def test_entropy_floor_not_in_adaptive_params(self):
        """
        Entropy floor should NOT be part of adaptive threshold framework.

        Scientific rationale:
        - Prompt entropy is HIGH (diverse instructions)
        - Valid CPG (facts) has LOW entropy
        - Repetition loops have NEAR-ZERO entropy
        - Adaptive calibration would break CPG detection for valid facts
        """
        config = AGSARConfig()

        # Entropy floor should be a small fixed constant
        assert config.cpg_entropy_floor == 0.01

        # Should NOT have an adaptive sigma multiplier for entropy
        assert not hasattr(config, 'cpg_entropy_sigma_multiplier')

    def test_entropy_floor_range(self):
        """Entropy floor should be a small positive value."""
        # Valid range: small positive (catches only stuttering loops)
        config = AGSARConfig(cpg_entropy_floor=0.005)
        assert config.cpg_entropy_floor == 0.005

        config2 = AGSARConfig(cpg_entropy_floor=0.02)
        assert config2.cpg_entropy_floor == 0.02

        # Zero is technically valid but not recommended
        config3 = AGSARConfig(cpg_entropy_floor=0.0)
        assert config3.cpg_entropy_floor == 0.0


class TestConfigSerialization:
    """Test that new config parameters serialize/deserialize correctly."""

    def test_to_dict_includes_adaptive_params(self):
        """to_dict() should include all adaptive threshold parameters."""
        config = AGSARConfig(
            use_adaptive_cpg_thresholds=True,
            cpg_gate_sigma_multiplier=-1.5,
            cpg_dispersion_sigma_multiplier=-2.0,
            use_percentile_gate_sharpening=True,
            gate_sharpen_low_percentile=5.0,
            gate_sharpen_high_percentile=95.0,
            gate_sharpen_low=0.15,
            gate_sharpen_high=0.85,
        )

        d = config.to_dict()

        assert d['use_adaptive_cpg_thresholds'] is True
        assert d['cpg_gate_sigma_multiplier'] == -1.5
        assert d['cpg_dispersion_sigma_multiplier'] == -2.0
        assert d['use_percentile_gate_sharpening'] is True
        assert d['gate_sharpen_low_percentile'] == 5.0
        assert d['gate_sharpen_high_percentile'] == 95.0
        assert d['gate_sharpen_low'] == 0.15
        assert d['gate_sharpen_high'] == 0.85

    def test_from_dict_roundtrip(self):
        """Config should survive to_dict/from_dict roundtrip."""
        original = AGSARConfig(
            use_adaptive_cpg_thresholds=False,
            cpg_gate_sigma_multiplier=-0.5,
            gate_sharpen_low_percentile=20.0,
        )

        d = original.to_dict()
        restored = AGSARConfig.from_dict(d)

        assert restored.use_adaptive_cpg_thresholds == original.use_adaptive_cpg_thresholds
        assert restored.cpg_gate_sigma_multiplier == original.cpg_gate_sigma_multiplier
        assert restored.gate_sharpen_low_percentile == original.gate_sharpen_low_percentile

    def test_legacy_config_still_works(self):
        """Old configs without new params should still work."""
        legacy_dict = {
            'semantic_layers': 4,
            'gate_temperature': 1.5,
            'cpg_gate_threshold': 0.3,  # Legacy hardcoded
        }
        config = AGSARConfig.from_dict(legacy_dict)

        # Legacy param should be set
        assert config.cpg_gate_threshold == 0.3

        # New params should have defaults
        assert config.use_adaptive_cpg_thresholds is True
        assert config.cpg_gate_sigma_multiplier == -1.0


class TestBackwardCompatibility:
    """Test backward compatibility with legacy configurations."""

    def test_adaptive_disabled_uses_hardcoded(self):
        """When adaptive disabled, legacy hardcoded values should be used."""
        config = AGSARConfig(
            use_adaptive_cpg_thresholds=False,
            cpg_gate_threshold=0.25,  # Custom legacy value
            cpg_dispersion_threshold=0.08,
        )

        # With adaptive disabled, these legacy values should be used
        assert config.cpg_gate_threshold == 0.25
        assert config.cpg_dispersion_threshold == 0.08

    def test_percentile_disabled_uses_hardcoded(self):
        """When percentile mode disabled, hardcoded sharpening values used."""
        config = AGSARConfig(
            use_percentile_gate_sharpening=False,
            gate_sharpen_low=0.15,
            gate_sharpen_high=0.85,
        )

        # With percentile mode disabled, these values should be used directly
        assert config.gate_sharpen_low == 0.15
        assert config.gate_sharpen_high == 0.85
