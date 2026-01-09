"""Tests for fused softmax + varentropy Triton kernel."""

import pytest
import torch


class TestFusedVarentropy:
    """Test fused varentropy computation parity and numerical stability."""

    @pytest.fixture
    def logits_small(self):
        """Small logits tensor for basic tests."""
        return torch.randn(2, 10, 1000)

    @pytest.fixture
    def logits_large(self):
        """Larger logits tensor for stress tests."""
        return torch.randn(4, 64, 32000)

    def test_pytorch_implementation_output_shape(self, logits_small):
        """PyTorch reference implementation returns correct shape."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        B, S, V = logits_small.shape
        result = fused_softmax_varentropy_torch(logits_small)

        assert result.shape == (B, S)
        assert result.dtype == logits_small.dtype

    def test_pytorch_implementation_non_negative(self, logits_small):
        """Varentropy should always be non-negative."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        result = fused_softmax_varentropy_torch(logits_small)

        assert (result >= 0).all(), "Varentropy should be non-negative"

    def test_pytorch_vs_direct_computation(self, logits_small):
        """Compare fused PyTorch implementation with direct computation."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        # Direct computation
        log_probs = torch.log_softmax(logits_small, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
        surprisal = -log_probs
        squared_deviation = (surprisal - entropy) ** 2
        expected = torch.sum(probs * squared_deviation, dim=-1)

        # Fused implementation
        result = fused_softmax_varentropy_torch(logits_small)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)

    def test_numerical_stability_extreme_logits(self):
        """Test numerical stability with extreme logit values."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        # Very large positive logits (could cause exp overflow without care)
        large_logits = torch.randn(2, 5, 100) * 100
        result = fused_softmax_varentropy_torch(large_logits)

        assert torch.isfinite(result).all(), "Result should be finite for large logits"
        assert (result >= 0).all(), "Varentropy should be non-negative"

        # Very large negative logits (should still work)
        small_logits = torch.randn(2, 5, 100) - 100
        result = fused_softmax_varentropy_torch(small_logits)

        assert torch.isfinite(result).all(), "Result should be finite for negative logits"

    def test_uniform_distribution_low_varentropy(self):
        """Uniform distribution should have relatively low varentropy."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        # Logits that produce near-uniform distribution
        uniform_logits = torch.zeros(1, 1, 100)
        result = fused_softmax_varentropy_torch(uniform_logits)

        # For uniform distribution, varentropy = 0 (all tokens have same surprisal)
        assert result[0, 0].item() < 0.01, "Uniform distribution should have near-zero varentropy"

    def test_peaked_distribution_low_varentropy(self):
        """Very peaked distribution should also have low varentropy."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        # Logits that produce peaked distribution (one dominant token)
        peaked_logits = torch.zeros(1, 1, 100)
        peaked_logits[0, 0, 0] = 100  # Huge peak at first token

        result = fused_softmax_varentropy_torch(peaked_logits)

        # Very peaked = almost all probability on one token = low variance in surprisal
        assert result[0, 0].item() < 0.1, "Very peaked distribution should have low varentropy"

    def test_mixed_distribution_higher_varentropy(self):
        """Mixed distribution (neither uniform nor peaked) should have higher varentropy."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy_torch

        # Bimodal-ish distribution
        bimodal_logits = torch.zeros(1, 1, 100)
        bimodal_logits[0, 0, :10] = 5  # 10 tokens with high prob
        bimodal_logits[0, 0, 10:] = -5  # Rest with low prob

        result = fused_softmax_varentropy_torch(bimodal_logits)

        # Mixed distribution should have higher varentropy than uniform or peaked
        # The exact value depends on the distribution shape
        assert result[0, 0].item() > 0, "Mixed distribution should have positive varentropy"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_parity_with_pytorch(self, logits_small):
        """Triton kernel should match PyTorch reference."""
        from ag_sar.ops.triton_logits import (
            fused_softmax_varentropy,
            fused_softmax_varentropy_torch,
        )

        logits_gpu = logits_small.cuda()

        triton_result = fused_softmax_varentropy(logits_gpu)
        torch_result = fused_softmax_varentropy_torch(logits_gpu)

        torch.testing.assert_close(triton_result, torch_result, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_large_vocabulary(self, logits_large):
        """Test Triton kernel with large vocabulary (typical LLM size)."""
        from ag_sar.ops.triton_logits import (
            fused_softmax_varentropy,
            fused_softmax_varentropy_torch,
        )

        logits_gpu = logits_large.cuda()

        triton_result = fused_softmax_varentropy(logits_gpu)
        torch_result = fused_softmax_varentropy_torch(logits_gpu)

        torch.testing.assert_close(triton_result, torch_result, rtol=1e-3, atol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_dtype_bf16(self):
        """Test Triton kernel with bfloat16 input."""
        from ag_sar.ops.triton_logits import fused_softmax_varentropy

        logits = torch.randn(2, 10, 1000, dtype=torch.bfloat16, device='cuda')

        result = fused_softmax_varentropy(logits)

        assert result.dtype == torch.bfloat16
        assert torch.isfinite(result).all()
        assert (result >= 0).all()


class TestImportFallback:
    """Test that import fallback works correctly."""

    def test_import_from_ops(self):
        """fused_softmax_varentropy should be importable from ops package."""
        from ag_sar.ops import fused_softmax_varentropy

        assert callable(fused_softmax_varentropy)

    def test_fallback_flag(self):
        """_LOGITS_TRITON_AVAILABLE flag should be set correctly."""
        from ag_sar.ops import _LOGITS_TRITON_AVAILABLE

        # On CPU-only or non-Triton systems, this should be False
        # On Triton-capable systems, this should be True
        # We just check it's a boolean
        assert isinstance(_LOGITS_TRITON_AVAILABLE, bool)

    def test_cpu_fallback(self):
        """CPU tensors should use PyTorch fallback."""
        from ag_sar.ops import fused_softmax_varentropy

        logits = torch.randn(2, 5, 100)  # CPU tensor
        result = fused_softmax_varentropy(logits)

        assert result.device.type == 'cpu'
        assert result.shape == (2, 5)
