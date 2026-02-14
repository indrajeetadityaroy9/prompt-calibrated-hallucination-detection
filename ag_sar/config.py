"""
Unified configuration for DSG hallucination detector.

All detection thresholds are adaptive — derived from input statistics
and model architecture. The only user-facing parameters are architectural
choices and operating points.
"""

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class DSGConfig:
    """
    Zero-configuration DSG detector.

    All signal thresholds, calibration, and aggregation parameters are
    self-derived from input statistics. User-facing knobs:
    - layer_subset: architectural choice for which layers to hook
    - conformal_alpha: miscoverage rate for conformal prediction (0.10 = 90% coverage)
    - response_flag_threshold: deprecated fallback operating point
    """

    # Layer selection: "all", "last_third", "last_quarter", or List[int]
    layer_subset: Union[str, List[int]] = "all"

    # Conformal miscoverage rate — the recommended path for binary decisions
    conformal_alpha: float = 0.10

    # Deprecated: kept for backward compat when conformal is not calibrated
    response_flag_threshold: float = 0.5

    def __post_init__(self):
        if isinstance(self.layer_subset, str):
            valid = {"all", "last_third", "last_quarter"}
            if self.layer_subset not in valid:
                raise ValueError(
                    f"layer_subset must be one of {valid} or List[int], "
                    f"got '{self.layer_subset}'"
                )
        if not 0 <= self.response_flag_threshold <= 1:
            raise ValueError(
                f"response_flag_threshold must be in [0, 1], "
                f"got {self.response_flag_threshold}"
            )
        if not 0 < self.conformal_alpha < 1:
            raise ValueError(
                f"conformal_alpha must be in (0, 1), "
                f"got {self.conformal_alpha}"
            )


@dataclass
class PrefillCalibration:
    """
    Self-calibrated parameters derived from prefill pass. Not user-facing.

    All values are computed from the actual input — no hardcoded priors.
    Signal normalization strategy:
    - CUS: direct mode (value IS the probability, no z-score)
    - POS/DPS: prompt-anchored z-score with MAD-based robust sigma
    """
    dps_mu: float = 0.0
    dps_sigma: float = 0.0
    dps_sigma_floor: float = 0.0
    pos_mu: float = 0.0
    pos_sigma: float = 0.0
    pos_sigma_floor: float = 0.0
    # CUS uses direct mode (no z-score) — see prompt_anchored.py
    cus_mode: str = "direct"
    # Adaptive layer sets (computed from JSD variance via Otsu)
    dps_layers: List[int] = field(default_factory=list)
    # Number of tokens used for calibration
    n_calibration_tokens: int = 0


@dataclass
class DSGTokenSignals:
    """Per-token signals for DSG detector."""
    cus: float = 0.0  # Context Utilization Score (attention)
    pos: float = 0.0  # Parametric Override Score (FFN)
    dps: float = 0.0  # Dual-Subspace Projection Score (representation)

    def as_dict(self) -> dict:
        return {"cus": self.cus, "pos": self.pos, "dps": self.dps}


@dataclass
class SpanRisk:
    """Risk information for a contiguous span of tokens."""
    start_token: int
    end_token: int
    text: str
    risk_score: float
    token_risks: List[float]


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    # Generated text
    generated_text: str

    # Token-level results
    token_signals: List[DSGTokenSignals]
    token_risks: List[float]

    # Span-level results
    risky_spans: List[SpanRisk]

    # Response-level results
    response_risk: float
    is_flagged: bool

    # Metadata
    num_tokens: int
    prompt_length: int

    # Conformal calibration (None if not calibrated)
    conformal_threshold: float = None
