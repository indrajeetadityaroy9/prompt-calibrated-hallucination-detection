"""
Tests for token entropy computation.

Tests the entropy measure used as a baseline for uncertainty quantification.
"""

import pytest
import torch

from ag_sar.measures.entropy import compute_token_entropy


class TestTokenEntropy:
    """Tests for compute_token_entropy function."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Uniform logits (equal probability for all tokens)
        logits = torch.zeros(batch_size, seq_len, vocab_size)

        entropy = compute_token_entropy(logits)

        # Normalized entropy should be close to 1 for uniform
        assert entropy.shape == (batch_size, seq_len)
        assert torch.all(entropy > 0.95), f"Expected entropy > 0.95, got {entropy}"

    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution should have low entropy."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Make one token have much higher probability
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        logits[:, :, 0] = 100.0  # Very high logit for first token

        entropy = compute_token_entropy(logits)

        # Entropy should be near 0 for peaked distribution
        assert entropy.shape == (batch_size, seq_len)
        assert torch.all(entropy < 0.1), f"Expected entropy < 0.1, got {entropy}"

    def test_entropy_bounds(self):
        """Entropy should be bounded in [0, 1] when normalized."""
        batch_size = 4
        seq_len = 20
        vocab_size = 1000

        # Random logits
        logits = torch.randn(batch_size, seq_len, vocab_size)

        entropy = compute_token_entropy(logits)

        assert entropy.shape == (batch_size, seq_len)
        assert torch.all(entropy >= 0.0), "Entropy should be >= 0"
        assert torch.all(entropy <= 1.0), "Normalized entropy should be <= 1"

    def test_2d_input(self):
        """Should handle 2D input (B, V)."""
        batch_size = 5
        vocab_size = 100

        logits = torch.randn(batch_size, vocab_size)

        entropy = compute_token_entropy(logits)

        assert entropy.shape == (batch_size,)

    def test_device_consistency(self):
        """Output should be on same device as input."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        logits = torch.randn(2, 10, 100, device="cuda")

        entropy = compute_token_entropy(logits)

        assert entropy.device == logits.device

    def test_gradient_flow(self):
        """Entropy computation should support gradients."""
        logits = torch.randn(2, 10, 100, requires_grad=True)

        entropy = compute_token_entropy(logits)
        loss = entropy.sum()
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_numerical_stability_extreme_logits(self):
        """Should handle extreme logit values without NaN/Inf."""
        batch_size = 2
        seq_len = 10
        vocab_size = 100

        # Very large positive logits
        logits = torch.ones(batch_size, seq_len, vocab_size) * 1000
        entropy = compute_token_entropy(logits)
        assert not torch.any(torch.isnan(entropy))
        assert not torch.any(torch.isinf(entropy))

        # Very large negative logits
        logits = torch.ones(batch_size, seq_len, vocab_size) * -1000
        entropy = compute_token_entropy(logits)
        assert not torch.any(torch.isnan(entropy))
        assert not torch.any(torch.isinf(entropy))
