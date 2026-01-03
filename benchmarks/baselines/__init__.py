"""
Baseline methods for hallucination detection comparison.

Provides implementations of:
- Predictive Entropy (simple baseline)
- SelfCheckGPT-Ngram (SOTA approximation)
- LogProb baseline (1 - mean token probability)
- EigenScore (spectral competitor)
- SAPLMA (hidden state analysis)
"""

from .entropy import PredictiveEntropyBaseline
from .selfcheck import SelfCheckNgramBaseline
from .logprob import LogProbBaseline
from .eigenscore import EigenScoreBaseline, SAPLMABaseline

__all__ = [
    "PredictiveEntropyBaseline",
    "SelfCheckNgramBaseline",
    "LogProbBaseline",
    "EigenScoreBaseline",
    "SAPLMABaseline",
]
