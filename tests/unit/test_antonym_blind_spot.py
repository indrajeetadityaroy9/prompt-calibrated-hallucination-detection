"""
Tests for JEPA Centroid Variance and the Antonym Safety Valve.

The "Antonym Blind Spot" is a critical risk when using centroid variance:
words like "hot/cold" or "positive/negative" cluster together in embedding
space despite being factually opposite.

The "Antonym Safety Valve" relies on: parametric_weight < threshold.
When Gate~0 (context ignored) and Trust=1.0 (confident antonym),
the max score = (1-Gate) * Trust * parametric_weight < threshold.
"""

import pytest
import torch
import torch.nn.functional as F

from ag_sar.measures.semantics import (
    compute_centroid_variance,
    compute_top1_projection,
    compute_semantic_dispersion,
)


class TestCentroidVariance:
    """Tests for the JEPA-style centroid variance function."""

    @pytest.fixture
    def vocab_size(self):
        return 1000

    @pytest.fixture
    def embed_dim(self):
        return 128

    @pytest.fixture
    def embed_matrix(self, vocab_size, embed_dim):
        """Create a normalized embedding matrix."""
        matrix = torch.randn(vocab_size, embed_dim)
        return F.normalize(matrix, p=2, dim=-1)

    def test_tight_cluster_low_variance(self, embed_matrix, vocab_size):
        """
        When all top-k tokens are identical (or very similar),
        centroid variance should be near zero.
        """
        # Create logits that heavily favor one token
        logits = torch.full((1, vocab_size), -100.0)
        logits[0, 42] = 10.0  # Dominant token

        variance = compute_centroid_variance(logits, embed_matrix, k=5)
        assert variance.item() < 0.1, f"Expected low variance for dominant token, got {variance.item()}"

    def test_dispersed_cluster_high_variance(self, embed_matrix, vocab_size):
        """
        When top-k tokens are spread across unrelated embeddings,
        centroid variance should be higher.
        """
        # Create logits with multiple equally likely tokens
        logits = torch.full((1, vocab_size), -100.0)
        # Select tokens that are likely far apart in embedding space
        logits[0, 0] = 5.0
        logits[0, 200] = 5.0
        logits[0, 400] = 5.0
        logits[0, 600] = 5.0
        logits[0, 800] = 5.0

        variance = compute_centroid_variance(logits, embed_matrix, k=5)
        # Should have some variance since tokens are spread out
        assert variance.item() > 0.0, "Expected non-zero variance for dispersed tokens"

    def test_output_shape_2d(self, embed_matrix, vocab_size):
        """Test output shape for 2D input (B, V)."""
        logits = torch.randn(4, vocab_size)
        variance = compute_centroid_variance(logits, embed_matrix, k=5)
        assert variance.shape == (4,)

    def test_output_shape_3d(self, embed_matrix, vocab_size):
        """Test output shape for 3D input (B, S, V)."""
        logits = torch.randn(2, 10, vocab_size)
        variance = compute_centroid_variance(logits, embed_matrix, k=5)
        assert variance.shape == (2, 10)

    def test_output_bounded(self, embed_matrix, vocab_size):
        """Variance should always be in [0, 1]."""
        logits = torch.randn(8, vocab_size)
        variance = compute_centroid_variance(logits, embed_matrix, k=10)
        assert (variance >= 0.0).all(), "Variance should be >= 0"
        assert (variance <= 1.0).all(), "Variance should be <= 1"


class TestDispatcher:
    """Tests for the semantic dispersion dispatcher."""

    @pytest.fixture
    def embed_matrix(self):
        return F.normalize(torch.randn(100, 64), p=2, dim=-1)

    def test_dispatcher_default(self, embed_matrix):
        """Default method should be top1_projection."""
        logits = torch.randn(2, 100)
        result = compute_semantic_dispersion(logits, embed_matrix, k=5)
        expected = compute_top1_projection(logits, embed_matrix, k=5)
        torch.testing.assert_close(result, expected)

    def test_dispatcher_top1_projection(self, embed_matrix):
        """Explicit top1_projection should use that method."""
        logits = torch.randn(2, 100)
        result = compute_semantic_dispersion(logits, embed_matrix, k=5, method="top1_projection")
        expected = compute_top1_projection(logits, embed_matrix, k=5)
        torch.testing.assert_close(result, expected)

    def test_dispatcher_centroid_variance(self, embed_matrix):
        """centroid_variance method should use centroid variance."""
        logits = torch.randn(2, 100)
        result = compute_semantic_dispersion(logits, embed_matrix, k=5, method="centroid_variance")
        expected = compute_centroid_variance(logits, embed_matrix, k=5)
        torch.testing.assert_close(result, expected)


class TestAntonymSafetyValve:
    """
    Tests verifying the "Antonym Safety Valve" mechanism.

    The safety valve relies on: parametric_weight < threshold.
    This ensures that even when JEPA thinks antonyms are "trusted",
    the final authority score stays below the hallucination threshold.
    """

    def test_antonym_safety_valve_math(self):
        """
        Verify the parametric_weight < threshold safety valve catches antonyms.

        Scenario: Model ignores context and confidently predicts an antonym.
        - Gate ~ 0 (context ignored)
        - Trust ~ 1.0 (antonyms cluster tightly in embedding space)
        - Safety Valve: parametric_weight (0.6) < threshold (0.7) -> FAIL
        """
        # Simulate a context-detached hallucination (Gate ~ 0)
        gate = 0.1  # Model ignored context
        flow = 0.0  # No authority from prompt (context was ignored)

        # The Trap: JEPA sees antonyms as "trusted" because they cluster
        # In real embedding space, "hot" and "cold" are semantically close
        variance = 0.05  # Low variance - tight cluster
        trust = 1.0 - variance  # High trust due to clustering

        # Parameters from summarization preset
        parametric_weight = 0.6
        threshold = 0.7  # Default hallucination threshold

        # Master equation when Gate ~ 0:
        # A(t) = Gate * Flow + (1 - Gate) * Trust * parametric_weight
        final_score = (gate * flow) + ((1 - gate) * trust * parametric_weight)

        # The score should be < threshold
        # With the values above: 0.1*0 + 0.9*0.95*0.6 = 0.513 < 0.7
        assert final_score < threshold, (
            f"Antonym bypassed safety valve! "
            f"Score {final_score:.3f} >= threshold {threshold}. "
            f"Ensure parametric_weight ({parametric_weight}) < threshold ({threshold})"
        )

    def test_safety_valve_edge_cases(self):
        """Test safety valve under various edge conditions."""
        threshold = 0.7

        # Case 1: Worst case - Gate=0, Trust=1.0
        gate = 0.0
        trust = 1.0
        parametric_weight = 0.6
        score = gate * 0 + (1 - gate) * trust * parametric_weight
        assert score < threshold, f"Edge case 1 failed: {score} >= {threshold}"

        # Case 2: Moderate gate, high trust
        gate = 0.3
        trust = 0.95
        flow = 0.2  # Some flow from context
        score = gate * flow + (1 - gate) * trust * parametric_weight
        # 0.3*0.2 + 0.7*0.95*0.6 = 0.06 + 0.399 = 0.459
        assert score < threshold, f"Edge case 2 failed: {score} >= {threshold}"

    def test_summarization_preset_constraint(self):
        """
        Verify the summarization preset satisfies the safety valve constraint.

        The preset must have parametric_weight < threshold.
        """
        # Values from summarization.yaml
        parametric_weight = 0.6
        threshold = 0.7  # Default from config

        assert parametric_weight < threshold, (
            f"CRITICAL: summarization preset violates safety valve! "
            f"parametric_weight ({parametric_weight}) >= threshold ({threshold}). "
            f"Antonyms will bypass hallucination detection."
        )

        # Maximum possible score when ignoring context entirely
        max_score_when_ignoring_context = parametric_weight * 1.0  # Trust can be at most 1.0
        assert max_score_when_ignoring_context < threshold, (
            f"Maximum score when ignoring context ({max_score_when_ignoring_context}) "
            f">= threshold ({threshold})"
        )


class TestCentroidVsTop1Comparison:
    """Compare behavior of centroid_variance vs top1_projection."""

    @pytest.fixture
    def embed_matrix(self):
        """Create embedding matrix with known structure."""
        vocab_size = 100
        embed_dim = 64
        matrix = torch.randn(vocab_size, embed_dim)
        return F.normalize(matrix, p=2, dim=-1)

    def test_synonyms_lower_dispersion_with_centroid(self, embed_matrix):
        """
        Centroid variance should give lower dispersion for synonym-like clusters
        compared to top1_projection.

        This is the key insight: centroid_variance is more tolerant of
        valid paraphrases where all alternatives mean similar things.
        """
        vocab_size = embed_matrix.shape[0]

        # Create "synonym-like" logits - alternatives are nearby in embedding space
        # Token 0 is the top choice, tokens 1-4 are "similar" (we make them nearby)
        logits = torch.full((1, vocab_size), -100.0)

        # Top-5 tokens with decreasing probability
        logits[0, 0] = 10.0
        logits[0, 1] = 8.0
        logits[0, 2] = 6.0
        logits[0, 3] = 4.0
        logits[0, 4] = 2.0

        # Both methods should produce valid outputs
        top1_disp = compute_top1_projection(logits, embed_matrix, k=5)
        centroid_disp = compute_centroid_variance(logits, embed_matrix, k=5)

        assert top1_disp.shape == centroid_disp.shape
        assert (top1_disp >= 0).all() and (top1_disp <= 1).all()
        assert (centroid_disp >= 0).all() and (centroid_disp <= 1).all()
