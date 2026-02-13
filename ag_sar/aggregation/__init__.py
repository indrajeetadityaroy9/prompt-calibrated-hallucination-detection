"""
Aggregation methods for token-to-response risk.

The Standard Model Pipeline (ICML-ready):
    1. Signal Extraction -> raw metrics (entropy, inv_margin, etc.)
    2. Prompt-Relative Z-Scoring: Z_i(t) = (S_i(t) - mu_prompt) / (sigma_prompt + eps)
    3. Probabilistic Mapping: P_i(t) = sigmoid(Z_i(t))
    4. Independence Assumption (Noisy-OR): P(Risk) = 1 - prod(1 - P_i(t))

Key Components:
- PromptAnchoredAggregator: The Standard Model - prompt-anchored normalization + Noisy-OR
- SpanMerger: Group high-risk tokens into contiguous spans
"""

from .prompt_anchored import (
    PromptAnchoredAggregator,
    AggregationResult,
    compute_prompt_statistics,
)
from .span_merger import SpanMerger, RiskySpan

__all__ = [
    # The Standard Model (ICML-ready)
    "PromptAnchoredAggregator",
    "AggregationResult",
    "compute_prompt_statistics",
    # Span utilities
    "SpanMerger",
    "RiskySpan",
]
