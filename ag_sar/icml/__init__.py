"""
ICML-Ready Hallucination Detection Module.

DSG (Decoupled Spectral Grounding) detector.
"""

from .dsg_detector import DSGDetector, CandidateSetManager

__all__ = [
    "DSGDetector",
    "CandidateSetManager",
]
