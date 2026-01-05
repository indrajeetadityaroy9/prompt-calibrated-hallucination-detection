"""Tests for AGSARConfig (v8.5)."""

import pytest
import torch
from ag_sar.config import AGSARConfig


class TestAGSARConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AGSARConfig()

        assert config.semantic_layers == 4
        assert config.residual_weight == 0.5
        assert config.power_iteration_steps == 3
        assert config.power_iteration_tol == 1e-4
        assert config.hallucination_threshold == 0.7
        assert config.preferred_dtype == torch.bfloat16
        # Unified Gating and Semantic Dispersion enabled by default
        assert config.enable_unified_gating is True
        assert config.enable_semantic_dispersion is True
        # Stability Gate parameters
        assert config.stability_sensitivity == 1.0
        assert config.parametric_weight == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AGSARConfig(
            semantic_layers=6,
            hallucination_threshold=0.5,
            stability_sensitivity=2.0,
        )

        assert config.semantic_layers == 6
        assert config.hallucination_threshold == 0.5
        assert config.stability_sensitivity == 2.0

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

    def test_invalid_semantic_layers(self):
        """Test validation of semantic layers."""
        with pytest.raises(ValueError, match="semantic_layers"):
            AGSARConfig(semantic_layers=0)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AGSARConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert 'semantic_layers' in d
        assert 'preferred_dtype' in d
        assert d['semantic_layers'] == 4
        assert d['enable_unified_gating'] is True
        assert d['enable_semantic_dispersion'] is True
        assert d['stability_sensitivity'] == 1.0
        assert d['parametric_weight'] == 0.5

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'semantic_layers': 3,
            'stability_sensitivity': 2.0,
        }
        config = AGSARConfig.from_dict(d)

        assert config.semantic_layers == 3
        assert config.stability_sensitivity == 2.0

    def test_fp16_warning(self):
        """Test warning for float16 usage."""
        with pytest.warns(UserWarning, match="float16"):
            AGSARConfig(preferred_dtype=torch.float16)
