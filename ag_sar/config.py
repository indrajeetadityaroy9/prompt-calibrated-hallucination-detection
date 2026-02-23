"""
Unified configuration for AG-SAR hallucination detector.

All detection thresholds are adaptive — derived from input statistics
and model architecture.
"""

from dataclasses import dataclass
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .aggregation.spans import RiskySpan


@dataclass
class TokenSignals:
    """Per-token signals for AG-SAR detector."""
    cus: float = 0.0   # Context Utilization Score (lookback ratio bimodality)
    pos: float = 0.0   # Parametric Override Score (FFN)
    dps: float = 0.0   # Dual-Subspace Projection Score (representation)
    dola: float = 0.0  # DoLa layer-contrast score (factuality)
    cgd: float = 0.0   # Context-Grounding Direction score (activation steering)


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    # Generated text
    generated_text: str

    # Token-level results
    token_signals: List[TokenSignals]
    token_risks: List[float]

    # Span-level results
    risky_spans: List["RiskySpan"]

    # Response-level results
    response_risk: float
    is_flagged: bool

    # Metadata
    num_tokens: int
    prompt_length: int
