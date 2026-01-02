"""Tests for AGSARConfig."""

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
        assert config.power_iteration_steps == 3  # Uses fast unrolled version
        assert config.power_iteration_tol == 1e-4  # Relaxed tolerance
        assert config.hallucination_threshold == 0.7
        assert config.preferred_dtype == torch.bfloat16
        # MC-SS defaults
        assert config.uncertainty_metric == "gse"
        assert config.mcss_beta == 5.0
        assert config.mcss_hebbian_tau == 0.1
        assert config.mcss_penalty_weight == 1.0
        # SGSS defaults
        assert config.use_spectral_steering is False
        assert config.steering_alpha == 2.0
        assert config.steering_beta == 5.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AGSARConfig(
            semantic_layers=6,
            hallucination_threshold=0.5,
            uncertainty_metric="mcss",
            use_spectral_steering=True,
        )

        assert config.semantic_layers == 6
        assert config.hallucination_threshold == 0.5
        assert config.uncertainty_metric == "mcss"
        assert config.use_spectral_steering is True

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

    def test_invalid_mcss_params(self):
        """Test validation of MC-SS parameters."""
        with pytest.raises(ValueError, match="mcss_beta"):
            AGSARConfig(mcss_beta=0)

        with pytest.raises(ValueError, match="mcss_hebbian_tau"):
            AGSARConfig(mcss_hebbian_tau=1.5)

        with pytest.raises(ValueError, match="mcss_penalty_weight"):
            AGSARConfig(mcss_penalty_weight=-1.0)

    def test_invalid_sgss_params(self):
        """Test validation of SGSS parameters."""
        with pytest.raises(ValueError, match="steering_alpha"):
            AGSARConfig(steering_alpha=-1.0)

        with pytest.raises(ValueError, match="steering_beta"):
            AGSARConfig(steering_beta=0)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = AGSARConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert 'semantic_layers' in d
        assert 'preferred_dtype' in d
        assert 'mcss_beta' in d
        assert 'steering_alpha' in d
        assert d['semantic_layers'] == 4

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            'semantic_layers': 3,
            'uncertainty_metric': 'mcss',
            'use_spectral_steering': True,
        }
        config = AGSARConfig.from_dict(d)

        assert config.semantic_layers == 3
        assert config.uncertainty_metric == 'mcss'
        assert config.use_spectral_steering is True

    def test_from_dict_deprecated_keys(self):
        """Test that deprecated keys are filtered out."""
        d = {
            'semantic_layers': 3,
            'entropy_threshold_low': 0.3,  # Deprecated
            'entropy_threshold_high': 0.9,  # Deprecated
            'use_flash_attn': True,  # Deprecated
            'use_head_weighting': True,  # Deprecated (replaced by SGSS)
            'head_weights_path': '/path/to/weights.json',  # Deprecated
            'align_with_tokens': False,  # Deprecated (now always True)
        }
        # Should not raise, deprecated keys should be filtered
        config = AGSARConfig.from_dict(d)
        assert config.semantic_layers == 3

    def test_fp16_warning(self):
        """Test warning for float16 usage."""
        with pytest.warns(UserWarning, match="float16"):
            AGSARConfig(preferred_dtype=torch.float16)
