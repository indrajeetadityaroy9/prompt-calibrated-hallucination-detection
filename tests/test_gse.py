"""Tests for Graph-Shifted Entropy (uncertainty) computation."""

import pytest
import torch
from ag_sar.uncertainty import (
    compute_token_entropy,
    normalize_relevance,
    compute_graph_shifted_entropy,
    detect_hallucination,
)


class TestTokenEntropy:
    """Test per-token entropy computation."""

    def test_entropy_shape(self):
        """Test output shape of token entropy."""
        batch_size, seq_len, vocab_size = 2, 8, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        entropy = compute_token_entropy(logits)

        assert entropy.shape == (batch_size, seq_len)

    def test_entropy_positive(self):
        """Test that entropy is always positive."""
        batch_size, seq_len, vocab_size = 2, 8, 1000
        logits = torch.randn(batch_size, seq_len, vocab_size)

        entropy = compute_token_entropy(logits)

        assert torch.all(entropy >= 0)

    def test_uniform_logits_max_entropy(self):
        """Test that uniform logits produce maximum entropy."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.zeros(batch_size, seq_len, vocab_size)  # Uniform distribution

        entropy = compute_token_entropy(logits)

        # Max entropy for 100-class uniform = log(100) ≈ 4.6
        expected = torch.log(torch.tensor(float(vocab_size)))
        assert torch.allclose(entropy, expected.expand_as(entropy), atol=0.01)

    def test_peaked_logits_low_entropy(self):
        """Test that peaked logits produce low entropy."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.full((batch_size, seq_len, vocab_size), -1000.0)
        logits[:, :, 0] = 0.0  # Only first class has probability

        entropy = compute_token_entropy(logits)

        # Should be close to 0
        assert torch.all(entropy < 0.1)

    def test_entropy_with_mask(self):
        """Test entropy computation with attention mask."""
        batch_size, seq_len, vocab_size = 1, 4, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        mask = torch.tensor([[1, 1, 0, 0]])

        entropy = compute_token_entropy(logits, attention_mask=mask)

        # Masked positions should be zero
        assert torch.all(entropy[:, 2:] == 0)


class TestNormalizeRelevance:
    """Test relevance normalization."""

    def test_normalize_sums_to_one(self):
        """Test that normalized relevance sums to 1."""
        batch_size, seq_len = 2, 8
        relevance = torch.rand(batch_size, seq_len) + 0.1

        normalized = normalize_relevance(relevance)

        sums = normalized.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_normalize_with_mask(self):
        """Test normalization with attention mask."""
        batch_size, seq_len = 1, 4
        relevance = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        mask = torch.tensor([[1, 1, 0, 0]])

        normalized = normalize_relevance(relevance, attention_mask=mask)

        # Valid positions should sum to 1
        valid_sum = normalized[:, :2].sum()
        assert torch.allclose(valid_sum, torch.tensor(1.0), atol=1e-5)

        # Masked positions should be 0
        assert torch.all(normalized[:, 2:] == 0)


class TestGraphShiftedEntropy:
    """Test GSE computation."""

    def test_gse_shape(self):
        """Test output shape of GSE."""
        batch_size, seq_len = 2, 8
        token_entropy = torch.rand(batch_size, seq_len)
        relevance = torch.rand(batch_size, seq_len) + 0.1

        gse = compute_graph_shifted_entropy(token_entropy, relevance)

        assert gse.shape == (batch_size,)

    def test_gse_positive(self):
        """Test that GSE is positive."""
        batch_size, seq_len = 2, 8
        token_entropy = torch.rand(batch_size, seq_len) + 0.1
        relevance = torch.rand(batch_size, seq_len) + 0.1

        gse = compute_graph_shifted_entropy(token_entropy, relevance)

        assert torch.all(gse >= 0)

    def test_gse_weighted_average(self):
        """Test that GSE equals weighted average of entropy."""
        batch_size, seq_len = 1, 4
        token_entropy = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        relevance = torch.tensor([[0.25, 0.25, 0.25, 0.25]])  # Equal weights

        gse = compute_graph_shifted_entropy(token_entropy, relevance)

        # Equal weights should give mean
        expected = token_entropy.mean()
        assert torch.allclose(gse, expected.unsqueeze(0), atol=1e-5)

    def test_gse_high_relevance_high_entropy(self):
        """Test that GSE is high when high-relevance tokens have high entropy."""
        batch_size, seq_len = 1, 4

        # Token 0: high entropy, high relevance
        # Token 1: low entropy, low relevance
        token_entropy = torch.tensor([[5.0, 0.1, 0.1, 0.1]])
        relevance = torch.tensor([[0.9, 0.03, 0.03, 0.04]])

        gse = compute_graph_shifted_entropy(token_entropy, relevance)

        # GSE should be high (close to 5.0 * 0.9 = 4.5)
        assert gse.item() > 4.0

    def test_gse_low_relevance_high_entropy(self):
        """Test that GSE is low when high-entropy tokens have low relevance."""
        batch_size, seq_len = 1, 4

        # Token 0: high entropy, low relevance
        # Token 1: low entropy, high relevance
        token_entropy = torch.tensor([[5.0, 0.1, 0.1, 0.1]])
        relevance = torch.tensor([[0.03, 0.9, 0.03, 0.04]])

        gse = compute_graph_shifted_entropy(token_entropy, relevance)

        # GSE should be low (approximately 0.1 * 0.9 + 5.0 * 0.03 ≈ 0.24)
        assert gse.item() < 0.5


class TestDetectHallucination:
    """Test hallucination detection."""

    def test_detect_above_threshold(self):
        """Test detection when GSE exceeds threshold."""
        gse = torch.tensor([0.8])
        threshold = 0.5

        is_hall, confidence = detect_hallucination(gse, threshold)

        assert is_hall.item() is True
        assert confidence.item() > 0.5

    def test_detect_below_threshold(self):
        """Test detection when GSE is below threshold."""
        gse = torch.tensor([0.3])
        threshold = 0.5

        is_hall, confidence = detect_hallucination(gse, threshold)

        assert is_hall.item() is False
        assert confidence.item() < 0.5

    def test_confidence_at_threshold(self):
        """Test confidence at threshold boundary."""
        gse = torch.tensor([0.5])
        threshold = 0.5

        is_hall, confidence = detect_hallucination(gse, threshold)

        # At threshold, confidence should be ~0.5
        assert abs(confidence.item() - 0.5) < 0.01

    def test_batch_detection(self):
        """Test batch hallucination detection."""
        gse = torch.tensor([0.3, 0.6, 0.9])
        threshold = 0.5

        is_hall, confidence = detect_hallucination(gse, threshold)

        assert is_hall.shape == (3,)
        assert is_hall[0].item() is False
        assert is_hall[1].item() is True
        assert is_hall[2].item() is True
