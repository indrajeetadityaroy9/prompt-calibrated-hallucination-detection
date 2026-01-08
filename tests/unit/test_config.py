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
        assert config.varentropy_lambda == 1.0
        assert config.sigma_multiplier == -1.0
        assert config.calibration_window == 64
        assert config.hallucination_threshold == 0.5
        assert config.dtype == "bfloat16"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AGSARConfig(
            semantic_layers=6,
            varentropy_lambda=0.5,
            hallucination_threshold=0.7,
        )

        assert config.semantic_layers == 6
        assert config.varentropy_lambda == 0.5
        assert config.hallucination_threshold == 0.7

    def test_invalid_semantic_layers(self):
        """Test validation of semantic layers."""
        with pytest.raises(ValueError, match="semantic_layers"):
            AGSARConfig(semantic_layers=0)

    def test_invalid_varentropy_lambda(self):
        """Test validation of varentropy_lambda."""
        with pytest.raises(ValueError, match="varentropy_lambda"):
            AGSARConfig(varentropy_lambda=-1.0)

        with pytest.raises(ValueError, match="varentropy_lambda"):
            AGSARConfig(varentropy_lambda=6.0)

    def test_invalid_hallucination_threshold(self):
        """Test validation of hallucination_threshold."""
        with pytest.raises(ValueError, match="hallucination_threshold"):
            AGSARConfig(hallucination_threshold=-0.1)

        with pytest.raises(ValueError, match="hallucination_threshold"):
            AGSARConfig(hallucination_threshold=1.5)

    def test_invalid_calibration_window(self):
        """Test validation of calibration_window."""
        with pytest.raises(ValueError, match="calibration_window"):
            AGSARConfig(calibration_window=0)

    def test_torch_dtype(self):
        """Test torch dtype conversion."""
        config = AGSARConfig(dtype="bfloat16")
        assert config.torch_dtype == torch.bfloat16

        config = AGSARConfig(dtype="float16")
        assert config.torch_dtype == torch.float16

        config = AGSARConfig(dtype="float32")
        assert config.torch_dtype == torch.float32
