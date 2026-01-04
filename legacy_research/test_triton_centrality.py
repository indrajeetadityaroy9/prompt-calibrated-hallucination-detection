"""
Tests for Triton-based matrix-free centrality computation.

These tests require a CUDA device and will be skipped if unavailable.
Run with: pytest tests/test_triton_centrality.py -v
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# Note: TestCentralityFlashKernel and TestMatrixFreePowerIteration classes
# have been removed as they tested internal kernel functions that have been
# refactored. The public API is tested via TestSinkAwareCentrality below.


class TestSinkAwareCentrality:
    """Tests for compute_sink_aware_centrality with matrix-free path."""

    @pytest.fixture
    def setup_full_inputs(self):
        """Create full inputs for sink-aware centrality."""
        B, L, H, S, D = 2, 4, 12, 64, 32
        device = torch.device('cuda')
        dtype = torch.bfloat16

        Q_stack = torch.randn(B, L * H, S, D, device=device, dtype=dtype)
        K_stack = torch.randn(B, L * H, S, D, device=device, dtype=dtype)
        value_norms = torch.rand(B, S, device=device, dtype=dtype) + 0.1

        return Q_stack, K_stack, value_norms, B, S

    def test_relevance_output(self, setup_full_inputs):
        """Test that relevance has correct shape and is non-negative."""
        from ag_sar.measures import compute_sink_aware_centrality

        Q_stack, K_stack, value_norms, B, S = setup_full_inputs

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
        )

        assert relevance.shape == (B, S)
        assert centrality.shape == (B, S)
        assert (relevance >= 0).all(), "Relevance should be non-negative"

    def test_sink_filtering_effect(self, setup_full_inputs):
        """Test that low value norms reduce relevance (sink filtering)."""
        from ag_sar.measures import compute_sink_aware_centrality

        Q_stack, K_stack, value_norms, B, S = setup_full_inputs

        # Set first token to have very low value norm (simulating sink)
        value_norms_sink = value_norms.clone()
        value_norms_sink[:, 0] = 0.01

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms_sink,
            Q_stack=Q_stack,
            K_stack=K_stack,
        )

        # First token should have lower relevance than average
        avg_relevance = relevance[:, 1:].mean(dim=-1)
        first_token_relevance = relevance[:, 0]

        assert (first_token_relevance < avg_relevance).all(), \
            "Sink token (low value norm) should have below-average relevance"
