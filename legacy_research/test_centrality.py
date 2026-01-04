"""Tests for centrality computation."""

import pytest
import torch
from ag_sar.measures import (
    compute_sink_aware_centrality,
    aggregate_value_norms,
)


class TestAggregateValueNorms:
    """Test value norm aggregation."""

    def test_aggregate_shape(self):
        """Test output shape of aggregation."""
        batch_size, num_heads, seq_len = 2, 4, 8
        value_norms_dict = {
            i: torch.rand(batch_size, num_heads, seq_len)
            for i in range(3)
        }

        aggregated = aggregate_value_norms(value_norms_dict)

        assert aggregated.shape == (batch_size, seq_len)

    def test_semantic_layers_selection(self):
        """Test that only semantic layers are used."""
        batch_size, num_heads, seq_len = 2, 4, 8

        # Create 4 layers with distinct values
        value_norms_dict = {
            0: torch.ones(batch_size, num_heads, seq_len) * 1.0,
            1: torch.ones(batch_size, num_heads, seq_len) * 2.0,
            2: torch.ones(batch_size, num_heads, seq_len) * 3.0,
            3: torch.ones(batch_size, num_heads, seq_len) * 4.0,
        }

        # Use only last 2 layers
        aggregated = aggregate_value_norms(value_norms_dict, semantic_layers=2)

        # Should average 3.0 and 4.0 = 3.5
        expected = torch.ones(batch_size, seq_len) * 3.5
        assert torch.allclose(aggregated, expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMatrixFreeCentrality:
    """Test matrix-free centrality computation (requires CUDA for Triton)."""

    def test_sink_aware_centrality_shape(self):
        """Test output shapes of sink-aware centrality."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 16
        device = torch.device('cuda')
        dtype = torch.bfloat16

        # Create Q and K stacks for matrix-free path
        Q_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        value_norms = (torch.rand(batch_size, seq_len, device=device, dtype=dtype) + 0.1)

        relevance, centrality, per_head_contrib = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
        )

        assert relevance.shape == (batch_size, seq_len)
        assert centrality.shape == (batch_size, seq_len)

    def test_centrality_sums_to_one(self):
        """Test that centrality sums to 1 (normalized)."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 16
        device = torch.device('cuda')
        dtype = torch.bfloat16

        Q_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        value_norms = (torch.rand(batch_size, seq_len, device=device, dtype=dtype) + 0.1)

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
        )

        sums = centrality.sum(dim=-1)
        assert torch.allclose(sums.float(), torch.ones_like(sums.float()), atol=1e-3)

    def test_sink_filtering(self):
        """Test that sinks (high centrality, low value norm) get low relevance."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 4, 16
        device = torch.device('cuda')
        dtype = torch.bfloat16

        # Random Q/K for realistic attention patterns
        Q_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Node 0 has very low value norm (sink), others have high
        value_norms = torch.tensor([[0.01, 1.0, 1.0, 1.0]], device=device, dtype=dtype)

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
        )

        # Sink should have lower relevance due to low value norm
        # Even if it has high centrality, R = C * ||v|| will be low
        assert relevance[0, 0] < relevance[0, 1:].max()

    def test_with_attention_mask(self):
        """Test sink-aware centrality with attention mask."""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 4, 16
        device = torch.device('cuda')
        dtype = torch.bfloat16

        Q_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        K_stack = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        value_norms = (torch.rand(batch_size, seq_len, device=device, dtype=dtype) + 0.1)
        mask = torch.tensor([[1, 1, 0, 0]], device=device)

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
            attention_mask=mask,
        )

        # Masked positions should have zero relevance
        assert torch.all(relevance[:, 2:] == 0)
