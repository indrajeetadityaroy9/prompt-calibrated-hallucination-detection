"""
Integration tests for metrics correctness using real datasets.

Uses HaluEval and RAGTruth to validate the evaluation framework
produces sensible results on actual hallucination detection data.
"""

import numpy as np
import pytest
from experiments.evaluation.metrics import MetricsCalculator


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


class TestCorrelationMetrics:
    """Test Spearman/Pearson/Point-Biserial correlation metrics."""

    def test_perfect_monotonic_correlation(self):
        """Perfect monotonic relationship should give high Spearman/Pearson."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Perfect monotonic: low scores for factual, high for hallucinated
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["spearman", "pearson", "pointbiserial"]
        )

        assert metrics["spearman"] > 0.85, f"Expected Spearman > 0.85, got {metrics['spearman']}"
        assert metrics["pearson"] > 0.85, f"Expected Pearson > 0.85, got {metrics['pearson']}"
        assert metrics["pointbiserial"] > 0.85, f"Expected Point-biserial > 0.85, got {metrics['pointbiserial']}"

    def test_random_scores_low_correlation(self):
        """Random scores should have correlation near zero."""
        np.random.seed(42)
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.random.rand(10)

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["spearman", "pearson", "pointbiserial"]
        )

        # Random should be near zero (within ±0.5 for small samples)
        assert -0.5 < metrics["spearman"] < 0.5
        assert -0.5 < metrics["pearson"] < 0.5

    def test_negative_correlation(self):
        """Inverted relationship should give negative correlation."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Inverted: high scores for factual, low for hallucinated
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["spearman", "pearson"]
        )

        assert metrics["spearman"] < -0.85
        assert metrics["pearson"] < -0.85


class TestAUPRCFactual:
    """Test inverted AUPRC (Positive Class = Factual) metric."""

    def test_auprc_factual_well_separated(self):
        """Well-separated data should have high AUPRC and AUPRC-Factual."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Well-separated: low uncertainty = factual, high = hallucinated
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.9])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["auprc", "auprc_factual"]
        )

        # Both should be high for well-separated data
        assert metrics["auprc"] > 0.8, f"Expected AUPRC > 0.8, got {metrics['auprc']}"
        assert metrics["auprc_factual"] > 0.8, f"Expected AUPRC-Factual > 0.8, got {metrics['auprc_factual']}"

    def test_auprc_factual_symmetry(self):
        """AUPRC-Factual should be high when low uncertainty = factual."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Perfect: factual (0) has lowest scores, hallucinated (1) highest
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["auprc", "auprc_factual"]
        )

        # Both should be high (≈1.0) for perfectly separated data
        assert metrics["auprc"] > 0.95
        assert metrics["auprc_factual"] > 0.95


class TestRiskAtCoverage:
    """Test Risk at Coverage metrics (consistent with AURC - lower=better)."""

    def test_risk_at_coverage_perfect_ranking(self):
        """Perfect ranking should have low risk at high coverage."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Perfect: factual samples have lowest uncertainty
        # Labels: 0=factual (good), 1=hallucinated (bad)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Scores: factual samples have low uncertainty, hallucinated have high
        scores = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["risk_80", "risk_90", "risk_95"]
        )

        # At 80% coverage (8 samples), keeping lowest-uncertainty 8
        # Should keep 5 factual + 3 hallucinated → risk = 3/8 = 0.375
        assert 0.3 <= metrics["risk_80"] <= 0.45, f"Expected risk_80 ≈ 0.375, got {metrics['risk_80']}"

        # At 90% coverage (9 samples), risk slightly higher
        assert metrics["risk_90"] >= metrics["risk_80"], "risk_90 should be >= risk_80"

        # At 95% coverage, risk even higher
        assert metrics["risk_95"] >= metrics["risk_90"], "risk_95 should be >= risk_90"

    def test_risk_at_coverage_poor_ranking(self):
        """Poor ranking should have high risk at all coverage levels."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Inverted: hallucinated samples have LOWEST uncertainty (worst case)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Scores are inverted - hallucinated have low uncertainty
        scores = np.array([0.7, 0.75, 0.8, 0.85, 0.9, 0.1, 0.15, 0.2, 0.25, 0.3])

        metrics, _ = calc.compute_all(
            labels.tolist(), scores.tolist(),
            ["risk_80", "risk_90", "risk_95"]
        )

        # At 80% coverage, keeping 8 lowest-uncertainty samples
        # Lowest 8 are: 5 hallucinated + 3 factual → risk = 5/8 = 0.625
        assert metrics["risk_80"] > 0.5, f"Poor ranking should have risk_80 > 0.5, got {metrics['risk_80']}"

    def test_risk_lower_is_better(self):
        """Verify that lower risk values indicate better ranking."""
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Good ranking
        labels_good = [0, 0, 0, 1, 1, 1]
        scores_good = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]

        # Poor ranking (inverted)
        labels_poor = [0, 0, 0, 1, 1, 1]
        scores_poor = [0.7, 0.8, 0.9, 0.1, 0.2, 0.3]

        metrics_good, _ = calc.compute_all(labels_good, scores_good, ["risk_80"])
        metrics_poor, _ = calc.compute_all(labels_poor, scores_poor, ["risk_80"])

        # Good ranking should have lower risk
        assert metrics_good["risk_80"] < metrics_poor["risk_80"], \
            f"Good ranking risk ({metrics_good['risk_80']}) should be < poor ranking ({metrics_poor['risk_80']})"


class TestLatencyHelpers:
    """Test latency computation helper functions."""

    def test_compute_latency_metrics(self):
        """Test latency statistics computation."""
        from experiments.evaluation.metrics import compute_latency_metrics

        latencies_ms = [10.0, 15.0, 20.0, 25.0, 100.0]  # P95 should catch the 100ms outlier
        token_counts = [50, 60, 55, 70, 65]

        stats = compute_latency_metrics(latencies_ms, token_counts)

        assert "latency_mean_ms" in stats
        assert "latency_std_ms" in stats
        assert "latency_p50_ms" in stats
        assert "latency_p95_ms" in stats
        assert "ms_per_token" in stats
        assert "tokens_per_sec" in stats

        # Mean should be 34.0
        assert abs(stats["latency_mean_ms"] - 34.0) < 0.01

        # P50 should be median (20.0)
        assert abs(stats["latency_p50_ms"] - 20.0) < 0.01

        # P95 should be high due to 100ms outlier
        assert stats["latency_p95_ms"] >= 25.0

        # Total time = 170ms, total tokens = 300 → ms/token ≈ 0.567
        total_tokens = sum(token_counts)
        total_time = sum(latencies_ms)
        expected_ms_per_token = total_time / total_tokens
        assert abs(stats["ms_per_token"] - expected_ms_per_token) < 0.01

    def test_compute_overhead_pct(self):
        """Test overhead percentage computation."""
        from experiments.evaluation.metrics import compute_overhead_pct

        method_latencies = [20.0, 25.0, 30.0]  # Mean = 25
        baseline_latencies = [10.0, 12.0, 8.0]  # Mean = 10

        overhead = compute_overhead_pct(method_latencies, baseline_latencies)

        # Expected: (25/10 - 1) * 100 = 150%
        assert abs(overhead - 150.0) < 0.01


class TestNewMetricsIntegration:
    """Integration tests for all new ICML/NeurIPS metrics."""

    def test_all_new_metrics_computable(self):
        """All new metrics should be computable together."""
        np.random.seed(42)
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        # Realistic data: mostly correct ranking with some noise
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        scores = np.array([0.2, 0.25, 0.35, 0.4, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9])

        all_new_metrics = [
            "auprc_factual",
            "spearman", "pearson", "pointbiserial",
            "risk_80", "risk_90", "risk_95",
        ]

        metrics, ci_bounds = calc.compute_all(
            labels.tolist(), scores.tolist(), all_new_metrics
        )

        # All metrics should be present
        for name in all_new_metrics:
            assert name in metrics, f"Missing metric: {name}"
            assert metrics[name] is not None, f"Metric {name} is None"

        # CI bounds should exist for all
        for name in all_new_metrics:
            assert name in ci_bounds, f"Missing CI bounds for: {name}"

    def test_full_icml_metric_suite(self):
        """Test the complete ICML/NeurIPS metric suite."""
        np.random.seed(42)
        calc = MetricsCalculator(bootstrap_samples=100, seed=42)

        labels = np.random.randint(0, 2, 100).tolist()
        scores = np.random.rand(100).tolist()

        full_suite = [
            # Discrimination
            "auroc", "auprc", "auprc_factual",
            # Classification
            "f1", "precision", "recall", "accuracy",
            # Calibration
            "ece", "brier",
            # Coverage & Utility
            "aurc", "risk_80", "risk_90", "risk_95",
            # Correlation
            "spearman", "pearson", "pointbiserial",
        ]

        metrics, ci_bounds = calc.compute_all(labels, scores, full_suite)

        # All 16 metrics should be computed
        assert len(metrics) == 16, f"Expected 16 metrics, got {len(metrics)}"

        for name in full_suite:
            assert name in metrics, f"Missing metric: {name}"
