"""
Unified configuration for DSG hallucination detector.

All detection thresholds are adaptive — derived from input statistics
and model architecture. The only user-facing parameters are architectural
choices and operating points.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .aggregation.span_merger import RiskySpan


class NormMode(Enum):
    """Signal normalization mode."""
    DIRECT = "direct"    # Value IS the probability (CUS ∈ [0,1])
    ZSCORE = "zscore"    # Prompt-anchored z-score → sigmoid


@dataclass
class SignalMetadata:
    """Signal properties — single source of truth for normalization and fallbacks."""
    name: str
    norm_mode: NormMode
    bounded: bool           # True if signal is in [0,1]
    neutral: float          # Value when signal is uninformative
    higher_is_riskier: bool

    def sigma_floor(self, sigma_from_data: float) -> float:
        """Data-derived sigma floor: 10% of observed sigma, minimum EPS."""
        from .numerics import EPS
        return max(0.1 * sigma_from_data, EPS) if sigma_from_data > 0 else 0.01

    def fallback_stats(self) -> dict:
        """Fallback when calibration fails — no model-specific constants."""
        if self.bounded:
            # Bounded [0,1] signals can safely use direct mode as fallback
            return {"mode": "direct"}
        # Unbounded: use neutral value with wide sigma (uninformative)
        return {"mu": self.neutral, "sigma": 1.0}


SIGNAL_REGISTRY = {
    "cus": SignalMetadata("cus", NormMode.DIRECT,  bounded=True,  neutral=0.5, higher_is_riskier=True),
    "pos": SignalMetadata("pos", NormMode.ZSCORE,  bounded=True,  neutral=0.0, higher_is_riskier=True),
    "dps": SignalMetadata("dps", NormMode.ZSCORE,  bounded=True,  neutral=0.5, higher_is_riskier=True),
    "dola": SignalMetadata("dola", NormMode.ZSCORE, bounded=False, neutral=0.0, higher_is_riskier=True),
    "cgd": SignalMetadata("cgd", NormMode.ZSCORE,  bounded=True,  neutral=0.5, higher_is_riskier=True),
}


@dataclass
class DSGConfig:
    """
    Zero-configuration DSG detector.

    All signal thresholds, calibration, and aggregation parameters are
    self-derived from input statistics. User-facing knobs:
    - layer_subset: architectural choice for which layers to hook
    """

    # Layer selection: "all", "last_third", "last_quarter", or List[int]
    layer_subset: Union[str, List[int]] = "all"

    def __post_init__(self):
        if isinstance(self.layer_subset, str):
            valid = {"all", "last_third", "last_quarter"}
            if self.layer_subset not in valid:
                raise ValueError(
                    f"layer_subset must be one of {valid} or List[int], "
                    f"got '{self.layer_subset}'"
                )


@dataclass
class DSGTokenSignals:
    """Per-token signals for DSG detector."""
    cus: float = 0.0   # Context Utilization Score (lookback ratio bimodality)
    pos: float = 0.0   # Parametric Override Score (FFN)
    dps: float = 0.0   # Dual-Subspace Projection Score (representation)
    dola: float = 0.0  # DoLa layer-contrast score (factuality)
    cgd: float = 0.0   # Context-Grounding Direction score (activation steering)

    def as_dict(self) -> dict:
        return {
            "cus": self.cus, "pos": self.pos, "dps": self.dps,
            "dola": self.dola, "cgd": self.cgd,
        }


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    # Generated text
    generated_text: str

    # Token-level results
    token_signals: List[DSGTokenSignals]
    token_risks: List[float]

    # Span-level results
    risky_spans: List["RiskySpan"]

    # Response-level results
    response_risk: float
    is_flagged: bool

    # Metadata
    num_tokens: int
    prompt_length: int
