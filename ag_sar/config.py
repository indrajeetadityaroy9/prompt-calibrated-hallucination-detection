"""
Unified configuration for AG-SAR hallucination detector.

All detection thresholds are adaptive — derived from input statistics.
"""

from dataclasses import dataclass


@dataclass
class TokenSignals:
    """Per-token signals for AG-SAR detector."""
    cus: float
    pos: float
    dps: float
    spt: float
    spectral_gap: float


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    generated_text: str
    token_signals: list[TokenSignals]
    token_risks: list[float]
    risky_spans: list
    response_risk: float
    is_flagged: bool
    num_tokens: int
    prompt_length: int
