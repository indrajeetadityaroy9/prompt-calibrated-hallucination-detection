"""
Aggregation methods for token-to-response risk.

DSG Pipeline:
    1. Signal Extraction -> CUS, POS, DPS, DoLa, CGD per token
    2. Signal-specific normalization:
       - Direct (CUS): value IS the probability
       - Z-scored (POS, DPS, DoLa, CGD): prompt-anchored z-score -> sigmoid
    3. Token-level: Entropy-gated fusion (w_i = (1-H_i)^2)
    4. Response-level: Signal-first mean of per-signal probabilities

Key Components:
- PromptAnchoredAggregator: Entropy-gated fusion + signal-first aggregation
- SpanMerger: Group high-risk tokens into contiguous spans (Tukey fence)
"""

from .prompt_anchored import (
    PromptAnchoredAggregator,
    AggregationResult,
)
from .span_merger import SpanMerger, RiskySpan

__all__ = [
    "PromptAnchoredAggregator",
    "AggregationResult",
    "SpanMerger",
    "RiskySpan",
]
