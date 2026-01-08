"""
Unit Tests: Fused Semantic Dispersion Triton Kernel.

Verifies numerical parity between Triton kernel and PyTorch fallback.

Tolerance requirements from plan:
- TopK indices: exact match
- Cosine similarity: rtol=1e-4, atol=1e-6
- Dispersion output: rtol=1e-3, atol=1e-5
"""

import torch
import pytest

# Import directly from submodules to avoid triggering full package import
# (which requires transformers). This allows testing ops in isolation.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from ag_sar.ops.torch_functional import semantic_dispersion_fallback

# Import backend detection and fused kernels from ops/__init__
# These imports don't require transformers
import ag_sar.ops as ops_module
fused_semantic_dispersion = ops_module.fused_semantic_dispersion
_SEMANTIC_TRITON_AVAILABLE = ops_module._SEMANTIC_TRITON_AVAILABLE


class TestSemanticDispersionParity:
    """Verify Triton kernel matches PyTorch fallback."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_parity_top1_projection(self):
        """Top-1 projection method should match PyTorch fallback."""
        torch.manual_seed(42)

        N, V, D = 128, 32000, 4096
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        # PyTorch reference
        result_pytorch = semantic_dispersion_fallback(
            probs, embed_matrix, k=5, method="top1_projection"
        )

        # Triton kernel (or fallback if Triton unavailable)
        result_triton = fused_semantic_dispersion(
            probs, embed_matrix, k=5, method="top1_projection"
        )

        torch.testing.assert_close(
            result_triton.cpu(),
            result_pytorch.cpu(),
            rtol=1e-3,
            atol=1e-5,
            msg="Triton top1_projection should match PyTorch fallback",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_parity_centroid_variance(self):
        """Centroid variance method should match PyTorch fallback."""
        torch.manual_seed(42)

        N, V, D = 128, 32000, 4096
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        # PyTorch reference
        result_pytorch = semantic_dispersion_fallback(
            probs, embed_matrix, k=5, method="centroid_variance"
        )

        # Triton kernel
        result_triton = fused_semantic_dispersion(
            probs, embed_matrix, k=5, method="centroid_variance"
        )

        torch.testing.assert_close(
            result_triton.cpu(),
            result_pytorch.cpu(),
            rtol=1e-3,
            atol=1e-5,
            msg="Triton centroid_variance should match PyTorch fallback",
        )


class TestSemanticDispersionBounds:
    """Verify output bounds and numerical stability."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_output_bounded_zero_one(self):
        """Dispersion output should be in [0, 1]."""
        torch.manual_seed(42)

        N, V, D = 256, 32000, 4096
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        dispersion = fused_semantic_dispersion(probs, embed_matrix, k=5)

        assert (dispersion >= 0.0).all(), f"Dispersion should be >= 0, min={dispersion.min()}"
        assert (dispersion <= 1.0).all(), f"Dispersion should be <= 1, max={dispersion.max()}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_deterministic_output(self):
        """Same input should give same output."""
        torch.manual_seed(42)

        N, V, D = 64, 32000, 4096
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        result1 = fused_semantic_dispersion(probs.clone(), embed_matrix, k=5)
        result2 = fused_semantic_dispersion(probs.clone(), embed_matrix, k=5)

        torch.testing.assert_close(
            result1, result2, rtol=0, atol=0,
            msg="Dispersion should be deterministic",
        )


class TestSemanticDispersionShapes:
    """Verify handling of different tensor shapes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_small_vocab(self):
        """Should work with smaller vocab (uses PyTorch fallback)."""
        torch.manual_seed(42)

        N, V, D = 64, 1000, 768  # Small vocab like GPT-2
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        # Should not raise, uses fallback for small vocab
        dispersion = fused_semantic_dispersion(probs, embed_matrix, k=5)

        assert dispersion.shape == (N,)
        assert (dispersion >= 0.0).all()
        assert (dispersion <= 1.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_large_vocab(self):
        """Should work with large vocab (128K, Llama-3 scale)."""
        torch.manual_seed(42)

        N, V, D = 32, 128000, 4096  # Llama-3 scale vocab
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        dispersion = fused_semantic_dispersion(probs, embed_matrix, k=5)

        assert dispersion.shape == (N,)
        assert (dispersion >= 0.0).all()
        assert (dispersion <= 1.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_various_k_values(self):
        """Should work with different TopK values."""
        torch.manual_seed(42)

        N, V, D = 64, 32000, 4096
        probs = torch.softmax(torch.randn(N, V, device="cuda"), dim=-1)
        embed_matrix = torch.randn(V, D, device="cuda")

        for k in [3, 5, 10, 20]:
            dispersion = fused_semantic_dispersion(probs, embed_matrix, k=k)
            assert dispersion.shape == (N,), f"k={k}: Wrong output shape"
            assert (dispersion >= 0.0).all(), f"k={k}: Min below 0"
            assert (dispersion <= 1.0).all(), f"k={k}: Max above 1"


class TestSemanticDispersionCPU:
    """Verify CPU fallback works."""

    def test_cpu_fallback(self):
        """Should work on CPU (uses PyTorch fallback)."""
        torch.manual_seed(42)

        N, V, D = 32, 10000, 768
        probs = torch.softmax(torch.randn(N, V), dim=-1)
        embed_matrix = torch.randn(V, D)

        # Force CPU by passing CPU tensors
        dispersion = semantic_dispersion_fallback(probs, embed_matrix, k=5)

        assert dispersion.shape == (N,)
        assert dispersion.device.type == "cpu"
        assert (dispersion >= 0.0).all()
        assert (dispersion <= 1.0).all()


class TestTritonAvailability:
    """Test Triton availability detection."""

    def test_triton_available_flag(self):
        """_SEMANTIC_TRITON_AVAILABLE should be boolean."""
        assert isinstance(_SEMANTIC_TRITON_AVAILABLE, bool)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_triton_on_cuda(self):
        """When CUDA available, Triton should be detected (if installed)."""
        # This test just verifies the flag exists and can be checked
        # Actual Triton availability depends on installation
        if _SEMANTIC_TRITON_AVAILABLE:
            # Triton is available, kernel should use optimized path
            pass
        else:
            # Triton not available, fallback should be used
            pass
        # Test passes either way - just ensures no import errors
