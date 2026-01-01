"""Tests for attention graph construction."""

import pytest
import torch
from ag_sar.attention_graph import (
    add_residual_connection,
    compute_attention_rollout,
    apply_attention_mask,
    build_global_attention_graph,
)


class TestResidualConnection:
    """Test residual connection addition."""

    def test_residual_preserves_shape(self):
        """Test that residual connection preserves tensor shape."""
        batch_size, seq_len = 2, 8
        attention = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)

        result = add_residual_connection(attention, residual_weight=0.5)

        assert result.shape == attention.shape

    def test_residual_weight_zero(self):
        """Test that weight=0 returns original attention."""
        batch_size, seq_len = 2, 8
        attention = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)

        result = add_residual_connection(attention, residual_weight=0.0)

        assert torch.allclose(result, attention, atol=1e-5)

    def test_residual_weight_one(self):
        """Test that weight=1 returns identity."""
        batch_size, seq_len = 2, 8
        attention = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)

        result = add_residual_connection(attention, residual_weight=1.0)

        expected = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_residual_weight_half(self):
        """Test that weight=0.5 mixes attention and identity equally."""
        batch_size, seq_len = 2, 8
        attention = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)

        result = add_residual_connection(attention, residual_weight=0.5)

        identity = torch.eye(seq_len).unsqueeze(0).expand(batch_size, -1, -1)
        expected = 0.5 * attention + 0.5 * identity
        assert torch.allclose(result, expected, atol=1e-5)


class TestAttentionRollout:
    """Test attention rollout computation."""

    def test_rollout_single_layer(self):
        """Test rollout with single layer."""
        batch_size, seq_len = 2, 8
        attention = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)

        rollout = compute_attention_rollout([attention], residual_weight=0.5)

        # Single layer rollout = residual-corrected attention
        expected = add_residual_connection(attention, 0.5)
        # Normalize rows
        expected = expected / expected.sum(dim=-1, keepdim=True)
        assert torch.allclose(rollout, expected, atol=1e-5)

    def test_rollout_multiple_layers(self):
        """Test rollout with multiple layers."""
        batch_size, seq_len = 2, 8
        layers = [
            torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)
            for _ in range(3)
        ]

        rollout = compute_attention_rollout(layers, residual_weight=0.5)

        assert rollout.shape == (batch_size, seq_len, seq_len)
        # Rows should sum to 1 (valid probability distribution)
        row_sums = rollout.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_rollout_empty_list_raises(self):
        """Test that empty layer list raises error."""
        with pytest.raises(ValueError, match="Empty"):
            compute_attention_rollout([])


class TestAttentionMask:
    """Test attention mask application."""

    def test_mask_zeros_padding(self):
        """Test that mask zeros out padding rows and columns."""
        batch_size, seq_len = 1, 4
        attention = torch.ones(batch_size, seq_len, seq_len) / seq_len
        mask = torch.tensor([[1, 1, 0, 0]])  # Last 2 positions are padding

        masked = apply_attention_mask(attention, mask, normalize=False)

        # Padding rows and columns should be zero
        assert torch.all(masked[:, 2:, :] == 0)
        assert torch.all(masked[:, :, 2:] == 0)
        # Valid region should still have values
        assert torch.all(masked[:, :2, :2] > 0)

    def test_mask_renormalization(self):
        """Test that masked attention is renormalized."""
        batch_size, seq_len = 1, 4
        attention = torch.ones(batch_size, seq_len, seq_len) / seq_len
        mask = torch.tensor([[1, 1, 0, 0]])

        masked = apply_attention_mask(attention, mask, normalize=True)

        # Valid rows should sum to 1
        valid_row_sums = masked[:, :2, :].sum(dim=-1)
        assert torch.allclose(valid_row_sums, torch.ones_like(valid_row_sums), atol=1e-5)


class TestBuildGlobalGraph:
    """Test global attention graph construction."""

    def test_global_graph_shape(self):
        """Test output shape of global graph."""
        batch_size, num_heads, seq_len = 2, 4, 8
        num_layers = 3

        attention_weights = {
            i: torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            for i in range(num_layers)
        }

        global_attn = build_global_attention_graph(attention_weights)

        assert global_attn.shape == (batch_size, seq_len, seq_len)

    def test_global_graph_with_mask(self):
        """Test global graph with attention mask."""
        batch_size, num_heads, seq_len = 1, 4, 4

        attention_weights = {
            0: torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
        }
        mask = torch.tensor([[1, 1, 0, 0]])

        global_attn = build_global_attention_graph(
            attention_weights,
            attention_mask=mask
        )

        # After masking, valid rows should only attend to valid positions
        # (columns 2,3 should be near-zero for valid rows)
        assert global_attn[:, :2, 2:].sum() < 0.1  # Minimal attention to padded positions

    def test_semantic_layers_selection(self):
        """Test that only semantic layers are used."""
        batch_size, num_heads, seq_len = 2, 4, 8

        # Create 6 layers with distinct patterns
        attention_weights = {}
        for i in range(6):
            attn = torch.randn(batch_size, num_heads, seq_len, seq_len).softmax(dim=-1)
            attention_weights[i] = attn

        # Use only last 2 layers
        global_attn = build_global_attention_graph(
            attention_weights,
            semantic_layers=2
        )

        assert global_attn.shape == (batch_size, seq_len, seq_len)
