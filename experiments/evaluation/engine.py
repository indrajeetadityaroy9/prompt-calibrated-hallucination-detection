"""
BenchmarkEngine: Orchestrates M methods × N datasets evaluation.

Phase 5.3 Compliant (Crash Resilience):
- JSONL streaming output (one line per sample)
- Incremental writing with flush() for crash safety
- Resume logic to skip already-processed samples
- Progress tracking with tqdm

Handles:
- Method initialization and lifecycle
- Dataset loading and iteration
- Streaming result logging (JSONL)
- Metric computation with bootstrap CI
- Proper cleanup to prevent hook conflicts

Key design: Uses streaming JSONL writes to prevent OOM on large datasets
and ensure crash recovery (99% of data saved if crash at 99%).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

from experiments.data.base import EvaluationDataset
from experiments.methods.base import UncertaintyMethod
from experiments.evaluation.metrics import MetricsCalculator
from experiments.evaluation.logging import JSONLLogger, load_jsonl_results
from experiments.configs.schema import ExperimentConfig


@dataclass
class BenchmarkResult:
    """
    Results for one method on one dataset.

    Note: This is used for in-memory aggregation after streaming writes.
    Individual results are streamed to JSONL, this holds aggregated metrics.
    """

    method_name: str
    dataset_name: str
    scores: List[float]
    labels: List[int]
    latencies_ms: List[float]
    confidences: List[Optional[float]]
    metrics: Dict[str, float] = field(default_factory=dict)
    ci_bounds: Dict[str, tuple] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.scores)

    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0


class BenchmarkEngine:
    """
    Orchestrates benchmark execution across methods and datasets.

    Features:
    - Streaming JSONL output (crash-safe)
    - Bootstrap confidence intervals
    - Automatic method cleanup between evaluations
    - Progress tracking with tqdm
    - Task-adaptive method factory for per-dataset method configuration

    Example:
        >>> engine = BenchmarkEngine(config, methods, datasets)
        >>> results = engine.run()
        >>> # Results also saved to experiments/results/{exp_name}_{timestamp}.jsonl
    """

    def __init__(
        self,
        config: ExperimentConfig,
        methods: Dict[str, UncertaintyMethod],
        datasets: Dict[str, EvaluationDataset],
        resume_from: Optional[str] = None,
    ):
        """
        Initialize benchmark engine.

        Args:
            config: Experiment configuration
            methods: Dict of {method_name: UncertaintyMethod instance}
            datasets: Dict of {dataset_name: EvaluationDataset instance}
            resume_from: Optional path to JSONL file to resume from (Phase 5.3)
        """
        self.config = config
        self.methods = methods
        self.datasets = datasets
        self.resume_from = resume_from

        self.metrics_calc = MetricsCalculator(
            bootstrap_samples=config.evaluation.bootstrap_samples,
            confidence_level=config.evaluation.confidence_level,
        )

        self.logger = JSONLLogger(config.output.output_dir)
        self.results: List[BenchmarkResult] = []

        # Phase 5.3: Resume logic - track already processed samples
        self._processed_ids: Set[Tuple[str, str, int]] = set()
        if resume_from:
            self._load_processed_ids(resume_from)

    def _load_processed_ids(self, resume_path: str) -> None:
        """
        Load already-processed sample IDs from an existing JSONL file.

        Phase 5.3: Resume Logic - skip already-processed samples on restart.
        """
        path = Path(resume_path)
        if not path.exists():
            print(f"Resume file not found: {resume_path}")
            return

        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        if "_type" not in record:  # Skip summaries
                            key = (
                                record.get("method", ""),
                                record.get("dataset", ""),
                                record.get("sample_idx", -1),
                            )
                            self._processed_ids.add(key)
                    except json.JSONDecodeError:
                        continue

            print(f"Resuming: Found {len(self._processed_ids)} already-processed samples")
        except Exception as e:
            print(f"Warning: Could not load resume file: {e}")

    def run(self) -> List[BenchmarkResult]:
        """
        Run all methods on all datasets.

        Returns:
            List of BenchmarkResult for each (method, dataset) pair
        """
        # Start logging
        self.logger.start_experiment(self.config.model_dump())

        for dataset_name, dataset in self.datasets.items():
            self._print_dataset_header(dataset_name, dataset)

            for method_name, method in self.methods.items():
                print(f"\n  [{method_name}] Running...")

                try:
                    result = self._run_method_on_dataset(
                        method, method_name, dataset, dataset_name
                    )

                    # Compute metrics
                    result.metrics, result.ci_bounds = self.metrics_calc.compute_all(
                        labels=result.labels,
                        scores=result.scores,
                        metric_names=self.config.evaluation.metrics,
                    )

                    # Log summary
                    self.logger.log_method_summary(
                        method_name=method_name,
                        dataset_name=dataset_name,
                        metrics=result.metrics,
                        ci_bounds=result.ci_bounds,
                        n_samples=result.n_samples,
                        mean_latency_ms=result.mean_latency_ms,
                    )

                    self.results.append(result)
                    self._print_result(result)

                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

                finally:
                    # CRITICAL: Clean up method to prevent hook conflicts
                    method.cleanup()

        # Finalize logging
        self.logger.finalize(self._create_final_summary())

        return self.results

    def _run_method_on_dataset(
        self,
        method: UncertaintyMethod,
        method_name: str,
        dataset: EvaluationDataset,
        dataset_name: str,
    ) -> BenchmarkResult:
        """
        Run a single method on a single dataset with streaming output.

        Phase 5.3: Includes resume logic and NaN-based error handling.
        """
        scores = []
        labels = []
        latencies = []
        confidences = []

        dataset.load()

        skipped = 0

        # Use tqdm for progress
        for idx, sample in enumerate(tqdm(dataset, desc=f"    {method_name}", leave=False)):
            # Phase 5.3: Resume logic - skip already-processed samples
            sample_key = (method_name, dataset_name, idx)
            if sample_key in self._processed_ids:
                skipped += 1
                continue

            try:
                result = method.compute_score(sample.prompt, sample.response)

                scores.append(result.score)
                labels.append(sample.label)
                latencies.append(result.latency_ms)
                confidences.append(result.confidence)

                # Stream result to JSONL immediately
                self.logger.log_result(
                    method_name=method_name,
                    dataset_name=dataset_name,
                    sample_idx=idx,
                    score=result.score,
                    label=sample.label,
                    latency_ms=result.latency_ms,
                    confidence=result.confidence,
                    extra=result.extra if result.extra else None,
                )

            except Exception as e:
                # Phase 1.5: Use NaN for errors, not 0.5 (which distorts metrics)
                scores.append(float("nan"))
                labels.append(sample.label)
                latencies.append(0.0)
                confidences.append(None)

                self.logger.log_result(
                    method_name=method_name,
                    dataset_name=dataset_name,
                    sample_idx=idx,
                    score=float("nan"),
                    label=sample.label,
                    latency_ms=0.0,
                    confidence=None,
                    extra={"status": "DROP", "error": str(e)[:100]},
                )

        if skipped > 0:
            print(f"    (Skipped {skipped} already-processed samples)")

        return BenchmarkResult(
            method_name=method_name,
            dataset_name=dataset_name,
            scores=scores,
            labels=labels,
            latencies_ms=latencies,
            confidences=confidences,
        )

    def _print_dataset_header(self, name: str, dataset: EvaluationDataset) -> None:
        """Print dataset information header."""
        print(f"\n{'=' * 60}")
        print(f"Dataset: {name}")
        print(f"{'=' * 60}")

        dataset.load()
        stats = dataset.get_statistics()
        print(
            f"Samples: {stats['total_samples']} "
            f"(Hall: {stats['hallucinated']}, Fact: {stats['factual']}, "
            f"Rate: {stats['hallucination_rate']:.1%})"
        )

    def _print_result(self, result: BenchmarkResult) -> None:
        """Print formatted result for one method."""
        # Only print metrics that were actually computed (in config)
        for metric_name in self.config.evaluation.metrics:
            if metric_name in result.metrics:
                val = result.metrics[metric_name]
                ci = result.ci_bounds.get(metric_name, (0, 0))
                print(f"    {metric_name.upper()}: {val:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"    Mean Latency: {result.mean_latency_ms:.1f} ms")

    def _create_final_summary(self) -> Dict:
        """Create final summary for logging."""
        return {
            "experiment_name": self.config.experiment.get("name", "unnamed"),
            "n_methods": len(self.methods),
            "n_datasets": len(self.datasets),
            "n_results": len(self.results),
            "methods": list(self.methods.keys()),
            "datasets": list(self.datasets.keys()),
        }

    def get_results_table(self) -> str:
        """
        Generate a formatted results table.

        Returns:
            String table of results
        """
        if not self.results:
            return "No results yet."

        # Build header dynamically based on configured metrics
        configured_metrics = self.config.evaluation.metrics
        header_parts = [f"{'Method':<20}", f"{'Dataset':<20}"]
        for metric in configured_metrics:
            header_parts.append(f"{metric.upper():<24}")
        header_parts.append(f"{'Latency (ms)':<12}")

        lines = []
        lines.append(" ".join(header_parts))
        lines.append("-" * (45 + 24 * len(configured_metrics)))

        for r in self.results:
            row_parts = [f"{r.method_name:<20}", f"{r.dataset_name:<20}"]

            for metric in configured_metrics:
                val = r.metrics.get(metric, 0)
                ci = r.ci_bounds.get(metric, (0, 0))
                metric_str = f"{val:.4f} [{ci[0]:.3f}, {ci[1]:.3f}]"
                row_parts.append(f"{metric_str:<24}")

            row_parts.append(f"{r.mean_latency_ms:<12.1f}")
            lines.append(" ".join(row_parts))

        return "\n".join(lines)
