"""
Unified configuration for AG-SAR hallucination detector.

All detection thresholds are adaptive — derived from input statistics.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class TokenSignals:
    """Per-token signals for AG-SAR detector."""
    cus: float = 0.5   # Context Utilization Score (lookback ratio bimodality)
    pos: float = 0.0   # Parametric Override Score (JSD-weighted directional override)
    dps: float = 0.5   # Dual-Subspace Projection Score (context vs reasoning geometry)
    spt: float = 0.5   # Spectral Phase-Transition score (MP BBP threshold)


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    generated_text: str
    token_signals: List[TokenSignals]
    token_risks: List[float]
    risky_spans: list
    response_risk: float
    is_flagged: bool
    num_tokens: int
    prompt_length: int
