"""Tests for head filtering module."""

import pytest
import torch
from ag_sar.head_filter import (
    compute_attention_entropy,
    compute_head_entropy,
    create_head_mask,
    filter_heads_by_entropy,
)


class TestAttentionEntropy:
    """Test entropy computation."""

    def test_uniform_attention_max_entropy(self):
        """Uniform attention should have maximum entropy (normalized to 1)."""
        batch_size, num_heads, seq_len = 2, 4, 8

        # Uniform attention
        attention = torch.ones(batch_size, num_heads, seq_len, seq_len) / seq_len

        entropy = compute_attention_entropy(attention)

        # Should be close to 1.0 (max normalized entropy)
        assert entropy.shape == (batch_size, num_heads, seq_len)
        assert torch.allclose(entropy, torch.ones_like(entropy), atol=0.01)

    def test_focused_attention_low_entropy(self):
        """Focused attention (attending to one token) should have low entropy."""
        batch_size, num_heads, seq_len = 2, 4, 8

        # Each query attends only to position 0
        attention = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        attention[:, :, :, 0] = 1.0

        entropy = compute_attention_entropy(attention)

        # Should be close to 0.0 (minimum entropy)
        assert entropy.shape == (batch_size, num_heads, seq_len)
        assert torch.all(entropy < 0.1)

    def test_entropy_with_mask(self):
        """Test entropy computation with attention mask."""
        batch_size, num_heads, seq_len = 1, 2, 4

        attention = torch.ones(batch_size, num_heads, seq_len, seq_len) / seq_len
        mask = torch.tensor([[1, 1, 0, 0]])  # Only first 2 tokens valid

        entropy = compute_attention_entropy(attention, attention_mask=mask)

        assert entropy.shape == (batch_size, num_heads, seq_len)


class TestHeadEntropy:
    """Test per-head entropy computation."""

    def test_head_entropy_shape(self):
        """Test output shape of head entropy."""
        batch_size, num_heads, seq_len = 2, 4, 8
        attention = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        head_entropy = compute_head_entropy(attention)

        assert head_entropy.shape == (batch_size, num_heads)

    def test_head_entropy_range(self):
        """Test that head entropy is in valid range [0, 1]."""
        batch_size, num_heads, seq_len = 2, 4, 8
        attention = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)

        head_entropy = compute_head_entropy(attention)

        assert torch.all(head_entropy >= 0)
        assert torch.all(head_entropy <= 1.1)  # Allow small numerical error


class TestHeadMask:
    """Test head mask creation."""

    def test_mask_filters_extremes(self):
        """Test that mask filters heads at threshold boundaries."""
        # Entropies: [0.1, 0.5, 0.9, 0.99]
        head_entropy = torch.tensor([[0.1, 0.5, 0.9, 0.99]])

        mask = create_head_mask(head_entropy, entropy_low=0.3, entropy_high=0.95)

        # Should keep only [0.5, 0.9]
        expected = torch.tensor([[False, True, True, False]])
        assert torch.equal(mask, expected)


class TestFilterHeads:
    """Test full head filtering pipeline."""

    def test_filter_returns_correct_shapes(self):
        """Test that filter returns correctly shaped outputs."""
        batch_size, num_heads, seq_len = 2, 4, 8
        num_layers = 3

        attention_weights = {
            i: torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for i in range(num_layers)
        }

        filtered, head_mask = filter_heads_by_entropy(attention_weights)

        assert head_mask.shape == (num_layers, num_heads)
        assert len(filtered) == num_layers
        for layer_idx in filtered:
            assert filtered[layer_idx].shape == (batch_size, num_heads, seq_len, seq_len)

    def test_filter_zeros_excluded_heads(self):
        """Test that filtered heads are zeroed out."""
        batch_size, num_heads, seq_len = 1, 4, 4

        # Create attention with known entropy patterns
        attention_weights = {
            0: torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
        }

        filtered, head_mask = filter_heads_by_entropy(
            attention_weights,
            entropy_low=0.0,
            entropy_high=0.5  # Will filter out high entropy heads
        )

        # Filtered heads should be zeroed
        for h in range(num_heads):
            if not head_mask[0, h]:
                assert torch.all(filtered[0][:, h] == 0)
