"""Tests for AGSARConfig."""

import pytest
import torch
from ag_sar.config import AGSARConfig


class TestAGSARConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AGSARConfig()

        assert config.entropy_threshold_low == 0.3
        assert config.entropy_threshold_high == 0.95
        assert config.semantic_layers == 4
        assert config.residual_weight == 0.5
        assert config.power_iteration_steps == 3  # Uses fast unrolled version
        assert config.power_iteration_tol == 1e-4  # Relaxed tolerance
        assert config.hallucination_threshold == 0.7
        assert config.preferred_dtype == torch.bfloat16

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AGSARConfig(
            entropy_threshold_low=0.2,
            entropy_threshold_high=0.8,
            semantic_layers=6,
            hallucination_threshold=0.5
        )

        assert config.entropy_threshold_low == 0.2
        assert config.entropy_threshold_high == 0.8
        assert config.semantic_layers == 6
        assert config.hallucination_threshold == 0.5

    def test_invalid_entropy_thresholds(self):
        """Test validation of entropy thresholds."""
        # Low threshold out of range
        with pytest.raises(ValueError, match="entropy_threshold_low"):
            AGSARConfig(entropy_threshold_low=-0.1)

        with pytest.raises(ValueError, match="entropy_threshold_low"):
            AGSARConfig(entropy_threshold_low=1.5)

        # High threshold out of range
        with pytest.raises(ValueError, match="entropy_threshold_high"):
            AGSARConfig(entropy_threshold_high=1.5)

        # Low >= High
        with pytest.raises(ValueError, match="must be <"):
            AGSARConfig(entropy_threshold_low=0.9, entropy_threshold_high=0.3)

    def test_invalid_residual_weight(self):
        """Test validation of residual weight."""
        with pytest.raises(ValueError, match="residual_weight"):
            AGSARConfig(residual_weight=-0.1)

        with pytest.raises(ValueError, match="residual_weight"):
            AGSARConfig(residual_weight=1.5)

    def test_invalid_power_iteration(self):
        """Test validation of power iteration parameters."""
        with pytest.raises(ValueError, match="power_iteration_steps"):
            AGSARConfig(power_iteration_steps=0)

        with pytest.raises(ValueError, match="power_iteration_tol"):
            AGSARConfig(power_iteration_tol=-1e-6)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AGSARConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert 'entropy_threshold_low' in d
        assert 'preferred_dtype' in d
        assert d['entropy_threshold_low'] == 0.3

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'entropy_threshold_low': 0.2,
            'entropy_threshold_high': 0.9,
            'semantic_layers': 3
        }
        config = AGSARConfig.from_dict(d)

        assert config.entropy_threshold_low == 0.2
        assert config.entropy_threshold_high == 0.9
        assert config.semantic_layers == 3

    def test_fp16_warning(self):
        """Test warning for float16 usage."""
        with pytest.warns(UserWarning, match="float16"):
            AGSARConfig(preferred_dtype=torch.float16)
