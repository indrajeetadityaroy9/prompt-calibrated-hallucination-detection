"""
AG-SAR Evaluation Framework.

PhD-level evaluation for validating AG-SAR's mechanistic correctness,
predictive performance, and computational efficiency.

Chapters:
    1. Mechanistic Verification (sink-awareness, POS correlation)
    2. Hallucination Detection (AUROC, ECE)
    3. Computational Profiling (latency, throughput)
    5. Ablation Studies (component importance)

Usage:
    python scripts/run_eval.py --chapter 3  # Profiling first
    python scripts/run_eval.py --all        # Full evaluation
"""

from .config import EvalConfig

__all__ = ['EvalConfig']
