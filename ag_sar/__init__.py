"""AG-SAR: Zero-Shot Hallucination Detection via Aggregated Signal Architecture for Risk."""

__version__ = "0.1.0"

from .config import TokenSignals, DetectionResult
from .detector import Detector
from .aggregation.spans import RiskySpan
from .signals.std import SemanticTrajectoryDynamics

__all__ = ["Detector", "TokenSignals", "DetectionResult", "RiskySpan", "SemanticTrajectoryDynamics"]
