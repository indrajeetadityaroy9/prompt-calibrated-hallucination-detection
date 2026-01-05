"""
Calibration module for AG-SAR Truth Vector.

Provides offline calibration of a "truthfulness direction" in the model's
residual stream for intrinsic hallucination detection.
"""

from .truth_vector import (
    TruthVectorConfig,
    TruthVectorCalibrator,
    compute_intrinsic_score,
)

__all__ = [
    "TruthVectorConfig",
    "TruthVectorCalibrator",
    "compute_intrinsic_score",
]
