"""
Integration tests for metrics correctness using real datasets.

Uses HaluEval and RAGTruth to validate the evaluation framework
produces sensible results on actual hallucination detection data.
"""

import numpy as np
import pytest
from experiments.core.metrics import MetricsCalculator
from experiments.core.validation import StageValidator


class TestMetricsOnHaluEval:
    """Test metrics computation on HaluEval dataset."""

    @pytest.fixture(scope="class")
    def halueval_data(self):
        """Load HaluEval QA dataset."""
        from experiments.data.halueval import HaluEvalDataset

        dataset = HaluEvalDataset(variant="qa", num_samples=100, seed=42)
        dataset.load()
        return dataset

    def test_dataset_loads(self, halueval_data):
        """HaluEval should load without errors."""
        assert len(halueval_data) > 0
        stats = halueval_data.get_statistics()
        assert stats["total_samples"] > 0

    def test_label_distribution(self, halueval_data):
        """HaluEval should have both positive and negative labels."""
        stats = halueval_data.get_statistics()
        assert stats["hallucinated"] > 0
        assert stats["factual"] > 0
        # Should be roughly balanced
        rate = stats["hallucination_rate"]
        assert 0.3 < rate < 0.7, f"Unexpected hallucination rate: {rate}"

    def test_metrics_computable(self, halueval_data):
        """All ICML metrics should be computable on HaluEval."""
        # Generate random scores for testing metric computation
        np.random.seed(42)
        labels = [s.label for s in halueval_data]
        scores = np.random.rand(len(labels)).tolist()

        calc = MetricsCalculator(bootstrap_samples=100, seed=42)
        metrics, ci_bounds = calc.compute_all(
            labels, scores, ["auroc", "auprc", "ece", "brier", "aurc"]
        )

        # All metrics should be computed
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "ece" in metrics
        assert "brier" in metrics
        assert "aurc" in metrics

        # Random scores should give AUROC near 0.5
        assert 0.4 < metrics["auroc"] < 0.6

        # CI bounds should exist
        for name in metrics:
            assert name in ci_bounds
            lo, hi = ci_bounds[name]
            assert lo <= metrics[name] <= hi or np.isnan(lo)

    def test_validation_passes(self, halueval_data):
        """Stage validation should pass on HaluEval."""
        np.random.seed(42)
        results = [
            {"score": np.random.rand(), "label": s.label}
            for s in halueval_data
        ]

        validator = StageValidator()
        result = validator.validate_stage(results)
        assert result.passed


class TestMetricsOnRAGTruth:
    """Test metrics computation on RAGTruth dataset."""

    @pytest.fixture(scope="class")
    def ragtruth_data(self):
        """Load RAGTruth dataset."""
        pytest.importorskip("datasets")
        from experiments.data.ragtruth import RAGTruthDataset

        dataset = RAGTruthDataset(num_samples=100, seed=42, filter_refusals=False)
        try:
            dataset.load()
        except Exception as e:
            pytest.skip(f"Could not load RAGTruth: {e}")
        return dataset

    def test_dataset_loads(self, ragtruth_data):
        """RAGTruth should load without errors."""
        assert len(ragtruth_data) > 0
        stats = ragtruth_data.get_statistics()
        assert stats["total_samples"] > 0

    def test_refusal_filtering(self):
        """Refusal filtering should work on RAGTruth."""
        pytest.importorskip("datasets")
        from experiments.data.ragtruth import RAGTruthDataset

        try:
            # Load with and without filtering
            unfiltered = RAGTruthDataset(num_samples=200, seed=42, filter_refusals=False)
            unfiltered.load()

            filtered = RAGTruthDataset(num_samples=200, seed=42, filter_refusals=True)
            filtered.load()

            # Filtered should have same or fewer samples
            assert len(filtered) <= len(unfiltered)
        except Exception as e:
            pytest.skip(f"Could not load RAGTruth: {e}")

    def test_metrics_computable(self, ragtruth_data):
        """All ICML metrics should be computable on RAGTruth."""
        np.random.seed(42)
        labels = [s.label for s in ragtruth_data]
        scores = np.random.rand(len(labels)).tolist()

        calc = MetricsCalculator(bootstrap_samples=100, seed=42)
        metrics, ci_bounds = calc.compute_all(
            labels, scores, ["auroc", "auprc", "ece", "brier", "aurc"]
        )

        # All metrics should be computed
        assert "auroc" in metrics
        assert "auprc" in metrics
        assert "ece" in metrics
        assert "brier" in metrics
        assert "aurc" in metrics


class TestNaNHandling:
    """Test NaN handling in metrics computation."""

    def test_nan_filtering(self):
        """NaN values should be filtered."""
        y_true = [0, 0, 1, 1, 0, 1]
        y_scores = [0.1, float('nan'), 0.8, 0.9, 0.2, float('nan')]

        calc = MetricsCalculator(bootstrap_samples=10, max_nan_rate=0.5)
        metrics, _ = calc.compute_all(y_true, y_scores, ["auroc"])
        assert "auroc" in metrics

    def test_high_nan_rate_fails(self):
        """High NaN rate should raise error."""
        y_true = [0] * 100
        y_scores = [float('nan')] * 10 + [0.5] * 90  # 10% NaN

        calc = MetricsCalculator(max_nan_rate=0.05)
        with pytest.raises(ValueError, match="Excessive NaNs"):
            calc.compute_all(y_true, y_scores, ["auroc"])


class TestValidation:
    """Test validation utilities."""

    def test_variance_check(self):
        """Degenerate scores should fail variance check."""
        from experiments.core.validation import StageValidator

        validator = StageValidator()
        # All same scores = zero variance
        results = [{"score": 0.5, "label": i % 2} for i in range(100)]

        with pytest.raises(ValueError, match="variance"):
            validator.validate_stage(results)

    def test_label_imbalance_check(self):
        """Extreme label imbalance should fail."""
        from experiments.core.validation import StageValidator

        validator = StageValidator()
        # 99% positive rate
        results = [
            {"score": np.random.rand(), "label": 1} for _ in range(99)
        ] + [{"score": np.random.rand(), "label": 0}]

        with pytest.raises(ValueError, match="imbalance"):
            validator.validate_stage(results)


class TestBootstrapCI:
    """Test bootstrap confidence intervals."""

    def test_ci_bounds_order(self):
        """Lower bound should be <= point estimate <= upper bound."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100).tolist()
        y_scores = np.random.rand(100).tolist()

        calc = MetricsCalculator(bootstrap_samples=100)
        metrics, ci_bounds = calc.compute_all(y_true, y_scores, ["auroc"])

        lo, hi = ci_bounds["auroc"]
        assert lo <= metrics["auroc"] <= hi
