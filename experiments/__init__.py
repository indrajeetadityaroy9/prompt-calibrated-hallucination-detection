"""
AG-SAR Experiments - The Laboratory.

This package contains the scientific evaluation framework for AG-SAR,
separated from the core library (src/ag_sar/) for clean architecture.

Usage:
    python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml
"""

from experiments.evaluation.engine import BenchmarkEngine
from experiments.configs.schema import ExperimentConfig

__all__ = ["BenchmarkEngine", "ExperimentConfig"]
