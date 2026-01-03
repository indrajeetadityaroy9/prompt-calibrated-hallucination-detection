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


class TestCentralityFlashKernel:
    """Tests for the Triton centrality_flash_fwd kernel."""

    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors on GPU."""
        B, H, S, D = 2, 4, 64, 32
        device = torch.device('cuda')
        dtype = torch.bfloat16

        Q = torch.randn(B, H, S, D, device=device, dtype=dtype)
        K = torch.randn(B, H, S, D, device=device, dtype=dtype)
        v = torch.randn(B, S, device=device, dtype=dtype)
        v = v / v.sum(dim=-1, keepdim=True)  # Normalize like centrality vector

        return Q, K, v, B, H, S, D

    def test_output_shape(self, setup_tensors):
        """Test that kernel output has correct shape (B, H, S)."""
        from ag_sar.kernels import centrality_flash_fwd

        Q, K, v, B, H, S, D = setup_tensors
        out = centrality_flash_fwd(Q, K, v)

        assert out.shape == (B, H, S), f"Expected {(B, H, S)}, got {out.shape}"

    def test_output_dtype(self, setup_tensors):
        """Test that output dtype matches input dtype."""
        from ag_sar.kernels import centrality_flash_fwd

        Q, K, v, B, H, S, D = setup_tensors
        out = centrality_flash_fwd(Q, K, v)

        assert out.dtype == Q.dtype, f"Expected {Q.dtype}, got {out.dtype}"

    def test_numerical_accuracy_vs_reference(self, setup_tensors):
        """Test kernel matches reference implementation within tolerance."""
        from ag_sar.kernels.centrality_flash import (
            centrality_flash_fwd,
            centrality_flash_reference
        )

        Q, K, v, B, H, S, D = setup_tensors

        # Use smaller tensors for reference comparison
        Q_small = Q[:1, :2, :32, :]
        K_small = K[:1, :2, :32, :]
        v_small = v[:1, :32]

        out_kernel = centrality_flash_fwd(Q_small, K_small, v_small)
        out_ref = centrality_flash_reference(Q_small, K_small, v_small)

        # Allow some tolerance for numerical differences
        torch.testing.assert_close(
            out_kernel.float(),
            out_ref.float(),
            atol=1e-2,
            rtol=1e-2
        )

    def test_causal_masking(self, setup_tensors):
        """Test that causal masking is correctly applied (future tokens don't contribute)."""
        from ag_sar.kernels import centrality_flash_fwd

        Q, K, v, B, H, S, D = setup_tensors

        # Set v to be a one-hot at the last position
        v_onehot = torch.zeros_like(v)
        v_onehot[:, -1] = 1.0

        out = centrality_flash_fwd(Q, K, v_onehot)

        # Only the last position should have non-zero output (due to causal mask)
        # All positions before the last should attend only to positions before them
        # So their weighted sum of v (which is 0 except at last) should be 0
        assert torch.allclose(out[:, :, :-1], torch.zeros_like(out[:, :, :-1]), atol=1e-5)


class TestMatrixFreePowerIteration:
    """Tests for matrix_free_power_iteration function."""

    @pytest.fixture
    def setup_qk_stacks(self):
        """Create Q/K stacks as if from multiple layers."""
        B, L, H, S, D = 2, 4, 12, 64, 32
        device = torch.device('cuda')
        dtype = torch.bfloat16

        # Stack as (B, L*H, S, D)
        Q_stack = torch.randn(B, L * H, S, D, device=device, dtype=dtype)
        K_stack = torch.randn(B, L * H, S, D, device=device, dtype=dtype)

        return Q_stack, K_stack, B, S

    def test_output_shape(self, setup_qk_stacks):
        """Test that output has shape (B, S)."""
        from ag_sar.centrality import matrix_free_power_iteration

        Q_stack, K_stack, B, S = setup_qk_stacks
        centrality, _ = matrix_free_power_iteration(Q_stack, K_stack)

        assert centrality.shape == (B, S), f"Expected {(B, S)}, got {centrality.shape}"

    def test_normalization(self, setup_qk_stacks):
        """Test that output sums to 1 (probability distribution)."""
        from ag_sar.centrality import matrix_free_power_iteration

        Q_stack, K_stack, B, S = setup_qk_stacks
        centrality, _ = matrix_free_power_iteration(Q_stack, K_stack)

        sums = centrality.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_convergence_iterations(self, setup_qk_stacks):
        """Test that more iterations converge to stable result."""
        from ag_sar.centrality import matrix_free_power_iteration

        Q_stack, K_stack, B, S = setup_qk_stacks

        c1, _ = matrix_free_power_iteration(Q_stack, K_stack, num_iterations=1)
        c3, _ = matrix_free_power_iteration(Q_stack, K_stack, num_iterations=3)
        c10, _ = matrix_free_power_iteration(Q_stack, K_stack, num_iterations=10)

        # More iterations should converge (c3 closer to c10 than c1 is)
        # Use <= because algorithm may converge in 1 iteration for some inputs
        diff_1_10 = (c1 - c10).abs().max()
        diff_3_10 = (c3 - c10).abs().max()

        assert diff_3_10 <= diff_1_10, "More iterations should converge or stabilize"

    def test_attention_mask_support(self, setup_qk_stacks):
        """Test that attention mask zeros out padded positions."""
        from ag_sar.centrality import matrix_free_power_iteration

        Q_stack, K_stack, B, S = setup_qk_stacks

        # Create mask with last 10 positions as padding
        attention_mask = torch.ones(B, S, device=Q_stack.device)
        attention_mask[:, -10:] = 0

        centrality, _ = matrix_free_power_iteration(
            Q_stack, K_stack, attention_mask=attention_mask
        )

        # Padded positions should have zero centrality
        assert torch.allclose(
            centrality[:, -10:],
            torch.zeros_like(centrality[:, -10:]),
            atol=1e-5
        )


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
        from ag_sar.centrality import compute_sink_aware_centrality

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
        from ag_sar.centrality import compute_sink_aware_centrality

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
