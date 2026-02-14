"""
Aggregation methods for token-to-response risk.

DSG Pipeline:
    1. Signal Extraction -> CUS, POS, DPS per token
    2. Prompt-Relative Z-Scoring: Z_i(t) = (S_i(t) - mu_prompt) / (sigma_prompt + eps)
    3. Probabilistic Mapping: P_i(t) = sigmoid(Z_i(t))
    4. Independence Assumption (Noisy-OR): P(Risk) = 1 - prod(1 - P_i(t))
    5. Response aggregation: p90

Key Components:
- PromptAnchoredAggregator: Prompt-anchored normalization + Noisy-OR + p90
- SpanMerger: Group high-risk tokens into contiguous spans (adaptive thresholds)
"""

from .prompt_anchored import (
    PromptAnchoredAggregator,
    AggregationResult,
)
from .span_merger import SpanMerger, RiskySpan
from .conformal import ConformalCalibrator

__all__ = [
    "PromptAnchoredAggregator",
    "AggregationResult",
    "SpanMerger",
    "RiskySpan",
    "ConformalCalibrator",
]
