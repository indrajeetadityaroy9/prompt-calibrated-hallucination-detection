"""
AG-SAR Evaluation Infrastructure Module.

This module provides the framework for running hallucination detection experiments
and computing standardized metrics.

Components:
    BenchmarkEngine:
        Orchestrates evaluation by iterating over datasets, running methods,
        collecting scores, and computing metrics. Supports progress tracking,
        checkpointing, and parallel execution.

    BenchmarkResult:
        Container for evaluation results with predictions, ground truth labels,
        metrics, and execution metadata. Serializable to JSON for analysis.

    MetricsCalculator:
        Computes standard binary classification metrics:
        - AUROC: Area Under ROC Curve (ranking quality)
        - AUPRC: Area Under Precision-Recall Curve (class-imbalanced performance)
        - F1 @ optimal threshold
        - Precision/Recall at configurable thresholds

    JSONLLogger:
        Streams results to JSONL files for real-time monitoring and crash recovery.
        Each line is a complete JSON object, enabling incremental parsing.

Reproducibility:
    The BenchmarkEngine accepts `seed` and `deterministic` parameters:
    - seed: Sets random state for all libraries (Python, NumPy, PyTorch)
    - deterministic: Enables PyTorch deterministic algorithms (may reduce performance)

Usage:
    >>> from experiments.evaluation import BenchmarkEngine, MetricsCalculator
    >>> engine = BenchmarkEngine(
    ...     methods=[AGSARMethod()],
    ...     dataset=HaluEvalDataset(variant="qa"),
    ...     seed=42,
    ...     deterministic=True,
    ... )
    >>> results = engine.run()
    >>> metrics = MetricsCalculator.compute(results)
    >>> print(f"AUROC: {metrics['auroc']:.4f}")
"""

from experiments.evaluation.engine import BenchmarkEngine, BenchmarkResult
from experiments.evaluation.metrics import MetricsCalculator
from experiments.evaluation.logging import JSONLLogger

__all__ = [
    "BenchmarkEngine",
    "BenchmarkResult",
    "MetricsCalculator",
    "JSONLLogger",
]
