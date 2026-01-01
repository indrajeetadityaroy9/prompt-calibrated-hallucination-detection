"""Tests for centrality computation."""

import pytest
import torch
from ag_sar.centrality import (
    power_iteration,
    compute_sink_aware_centrality,
    aggregate_value_norms,
)


class TestPowerIteration:
    """Test power iteration for eigenvector centrality."""

    def test_power_iteration_output_shape(self):
        """Test output shape of power iteration."""
        batch_size, n = 2, 8
        adj_matrix = torch.randn(batch_size, n, n).softmax(dim=-1)

        centrality = power_iteration(adj_matrix, num_iterations=50)

        assert centrality.shape == (batch_size, n)

    def test_power_iteration_positive_values(self):
        """Test that centrality values are positive."""
        batch_size, n = 2, 8
        adj_matrix = torch.randn(batch_size, n, n).softmax(dim=-1)

        centrality = power_iteration(adj_matrix, num_iterations=50)

        assert torch.all(centrality >= 0)

    def test_power_iteration_sums_to_one(self):
        """Test that centrality sums to 1 (normalized)."""
        batch_size, n = 2, 8
        adj_matrix = torch.randn(batch_size, n, n).softmax(dim=-1)

        centrality = power_iteration(adj_matrix, num_iterations=50)

        sums = centrality.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_power_iteration_uniform_matrix(self):
        """Test centrality on uniform adjacency matrix."""
        batch_size, n = 2, 8
        adj_matrix = torch.ones(batch_size, n, n) / n

        centrality = power_iteration(adj_matrix, num_iterations=100)

        # All nodes should have equal centrality
        expected = torch.ones(batch_size, n) / n
        assert torch.allclose(centrality, expected, atol=1e-3)

    def test_power_iteration_star_graph(self):
        """Test centrality on star graph (one central node)."""
        batch_size, n = 1, 5
        # All nodes point to node 0
        adj_matrix = torch.zeros(batch_size, n, n)
        adj_matrix[:, :, 0] = 1.0

        centrality = power_iteration(adj_matrix, num_iterations=100)

        # Node 0 should have highest centrality
        assert centrality[0, 0] == centrality[0].max()

    def test_power_iteration_convergence(self):
        """Test that power iteration converges."""
        batch_size, n = 2, 8
        adj_matrix = torch.randn(batch_size, n, n).softmax(dim=-1)

        # Run with different iteration counts
        centrality_50 = power_iteration(adj_matrix, num_iterations=50)
        centrality_100 = power_iteration(adj_matrix, num_iterations=100)

        # Should be similar (converged)
        assert torch.allclose(centrality_50, centrality_100, atol=1e-3)


class TestSinkAwareCentrality:
    """Test sink-aware centrality computation."""

    def test_sink_aware_centrality_shape(self):
        """Test output shapes of sink-aware centrality."""
        batch_size, seq_len = 2, 8
        attention_graph = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)
        value_norms = torch.rand(batch_size, seq_len) + 0.1

        relevance, centrality = compute_sink_aware_centrality(
            attention_graph, value_norms
        )

        assert relevance.shape == (batch_size, seq_len)
        assert centrality.shape == (batch_size, seq_len)

    def test_sink_filtering(self):
        """Test that sinks (high centrality, low value norm) get low relevance."""
        batch_size, seq_len = 1, 4

        # Create graph with varying centrality (not uniform)
        # Node 0 receives more attention
        attention_graph = torch.tensor([[[0.7, 0.1, 0.1, 0.1],
                                          [0.7, 0.1, 0.1, 0.1],
                                          [0.7, 0.1, 0.1, 0.1],
                                          [0.7, 0.1, 0.1, 0.1]]])

        # Node 0 has low value norm (sink), others have high
        value_norms = torch.tensor([[0.01, 1.0, 1.0, 1.0]])

        relevance, centrality = compute_sink_aware_centrality(
            attention_graph, value_norms
        )

        # Node 0 should have relatively high centrality
        # but low relevance due to low value norm
        # The key is that relevance[0] < what it would be without value norm weighting
        assert relevance[0, 0] < relevance[0, 1:].max()  # Sink has lower relevance than at least one other node

    def test_with_attention_mask(self):
        """Test sink-aware centrality with attention mask."""
        batch_size, seq_len = 1, 4
        attention_graph = torch.randn(batch_size, seq_len, seq_len).softmax(dim=-1)
        value_norms = torch.rand(batch_size, seq_len) + 0.1
        mask = torch.tensor([[1, 1, 0, 0]])

        relevance, centrality = compute_sink_aware_centrality(
            attention_graph, value_norms, attention_mask=mask
        )

        # Masked positions should have zero relevance
        assert torch.all(relevance[:, 2:] == 0)


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


