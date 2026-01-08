"""
AG-SAR Configuration - Minimal Dynamic Architecture.

Core mechanism parameters only. All thresholds are auto-calibrated from prompt statistics.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class AGSARConfig:
    """
    Minimal configuration for AG-SAR hallucination detection.

    Most parameters are auto-derived from prompt calibration.
    Only essential tuning knobs are exposed.

    Core Equation:
        Uncertainty(t) = 1 - Authority(t)
        Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)
        Trust(t) = 1 - Dispersion(t) × (1 + λ × Varentropy(t))
    """

    # === CORE MECHANISM ===
    semantic_layers: int = 4
    """Number of final transformer layers to analyze."""

    varentropy_lambda: float = 1.0
    """Varentropy weighting for dispersion. Higher = penalize oscillating confidence more."""

    # === CALIBRATION ===
    sigma_multiplier: float = -1.0
    """Z-score threshold for adaptive CPG detection. Negative = below mean."""

    calibration_window: int = 64
    """Tokens to analyze during calibration. Captures instruction uncertainty."""

    # === CLASSIFICATION ===
    hallucination_threshold: float = 0.5
    """Decision boundary. Uncertainty > threshold = hallucination."""

    # === HARDWARE ===
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    """Compute precision."""

    def __post_init__(self):
        if self.semantic_layers < 1:
            raise ValueError(f"semantic_layers must be >= 1, got {self.semantic_layers}")
        if not 0.0 <= self.varentropy_lambda <= 5.0:
            raise ValueError(f"varentropy_lambda must be in [0, 5], got {self.varentropy_lambda}")
        if self.calibration_window < 1:
            raise ValueError(f"calibration_window must be >= 1, got {self.calibration_window}")
        if not 0.0 <= self.hallucination_threshold <= 1.0:
            raise ValueError(f"hallucination_threshold must be in [0, 1], got {self.hallucination_threshold}")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[self.dtype]
