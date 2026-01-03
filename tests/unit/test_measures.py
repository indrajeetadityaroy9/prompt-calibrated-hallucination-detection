"""Unit tests for measures module."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ag_sar.measures import (
    compute_token_entropy,
    compute_graph_shifted_entropy,
    normalize_relevance,
    detect_hallucination,
    compute_bounded_surprisal,
)


class TestEntropy:
    """Tests for entropy computation."""

    def test_compute_token_entropy_shape(self, batch_size, seq_len, device, dtype):
        """Test entropy output shape matches input."""
        vocab_size = 50257
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)

        entropy = compute_token_entropy(logits)

        assert entropy.shape == (batch_size, seq_len)

    def test_entropy_is_non_negative(self, batch_size, seq_len, device, dtype):
        """Entropy must be non-negative."""
        vocab_size = 1000
        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)

        entropy = compute_token_entropy(logits)

        assert (entropy >= 0).all()

    def test_entropy_alignment(self, device, dtype):
        """Test entropy is properly aligned with tokens."""
        # entropy[i] should represent uncertainty about token[i]
        # Since logits[i] predicts token[i+1], we shift
        batch_size = 1
        seq_len = 5
        vocab_size = 100

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device, dtype=dtype)
        entropy = compute_token_entropy(logits)

        # First position should be 0 (no prediction for first token)
        assert entropy[0, 0] == 0


class TestGSE:
    """Tests for Graph-Shifted Entropy."""

    def test_gse_shape(self, batch_size, seq_len, device, dtype):
        """Test GSE returns scalar per batch."""
        entropy = torch.rand(batch_size, seq_len, device=device, dtype=dtype)
        relevance = torch.rand(batch_size, seq_len, device=device, dtype=dtype)

        gse = compute_graph_shifted_entropy(entropy, relevance)

        assert gse.shape == (batch_size,)

    def test_gse_is_non_negative(self, batch_size, seq_len, device, dtype):
        """GSE must be non-negative."""
        entropy = torch.rand(batch_size, seq_len, device=device, dtype=dtype)
        relevance = torch.rand(batch_size, seq_len, device=device, dtype=dtype)

        gse = compute_graph_shifted_entropy(entropy, relevance)

        assert (gse >= 0).all()

    def test_normalize_relevance_sums_to_one(self, batch_size, seq_len, device, dtype):
        """Normalized relevance should sum to 1."""
        relevance = torch.rand(batch_size, seq_len, device=device, dtype=dtype)

        normalized = normalize_relevance(relevance)

        sums = normalized.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestHallucinationDetection:
    """Tests for hallucination detection."""

    def test_detect_returns_boolean(self, device):
        """Detection should return boolean tensor."""
        gse = torch.tensor([0.5, 0.8, 0.3], device=device)
        threshold = 0.6

        is_hall, conf = detect_hallucination(gse, threshold)

        assert is_hall.dtype == torch.bool
        assert is_hall.shape == gse.shape

    def test_detection_threshold(self, device):
        """Values above threshold should be detected."""
        gse = torch.tensor([0.5, 0.8, 0.3], device=device)
        threshold = 0.6

        is_hall, conf = detect_hallucination(gse, threshold)

        assert is_hall[0] == False  # 0.5 < 0.6
        assert is_hall[1] == True   # 0.8 > 0.6
        assert is_hall[2] == False  # 0.3 < 0.6

    def test_confidence_range(self, device):
        """Confidence should be in [0, 1]."""
        gse = torch.tensor([0.5, 0.8, 0.3], device=device)
        threshold = 0.6

        is_hall, conf = detect_hallucination(gse, threshold)

        assert (conf >= 0).all()
        assert (conf <= 1).all()
