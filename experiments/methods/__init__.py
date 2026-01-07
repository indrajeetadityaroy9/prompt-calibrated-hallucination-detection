"""
AG-SAR Baseline Methods Module.

This module provides implementations of uncertainty quantification methods
for hallucination detection, enabling fair comparison against AG-SAR.

Methods by Category:

    Single-Pass Methods (no additional inference):
        - AGSARMethod: AG-SAR v8.0 (Authority Flow + Gating + Dispersion)
        - LogProbMethod: Mean negative log probability
        - PredictiveEntropyMethod: Shannon entropy over vocabulary

    Multi-Pass Methods (require additional forward passes):
        - SelfCheckNgramMethod: N-gram consistency across samples
        - SelfCheckNLIMethod: NLI-based consistency checking
        - SemanticEntropyMethod: Entropy over semantic clusters

    Embedding-Based Methods:
        - EigenScoreMethod: Principal eigenvalue from token embeddings
        - SAPLMAMethod: Self-Assessment via Eigenvalue Analysis

    Hybrid Methods:
        - LLMCheckAttentionMethod: Attention-based internal consistency
        - LLMCheckHiddenMethod: Hidden state trajectory analysis
        - LLMCheckLogitMethod: Logit-based confidence estimation

Interface:
    All methods implement the UncertaintyMethod ABC:
    - setup(model, tokenizer): Initialize method with model
    - compute_uncertainty(prompt, response) -> MethodResult
    - cleanup(): Release resources

    MethodResult contains:
    - score: float in [0, 1] (higher = more likely hallucinated)
    - details: dict with method-specific information

Reproducibility:
    Sampling-based methods (SemanticEntropy, SelfCheck, EigenScore) accept a
    `seed` parameter for deterministic behavior. This seed is reset before
    each generation call to ensure reproducibility.

Usage:
    >>> method = AGSARMethod(config=AGSARConfig())
    >>> method.setup(model, tokenizer)
    >>> result = method.compute_uncertainty(prompt, response)
    >>> print(f"Uncertainty: {result.score:.3f}")
    >>> method.cleanup()
"""

from experiments.methods.base import UncertaintyMethod, MethodResult
from experiments.methods.agsar_wrapper import AGSARMethod
from experiments.methods.logprob import LogProbMethod
from experiments.methods.entropy import PredictiveEntropyMethod
from experiments.methods.selfcheck import SelfCheckNgramMethod, SelfCheckNLIMethod
from experiments.methods.eigenscore import EigenScoreMethod, SAPLMAMethod
from experiments.methods.llm_check import (
    LLMCheckAttentionMethod,
    LLMCheckHiddenMethod,
    LLMCheckLogitMethod,
)
from experiments.methods.semantic_entropy import SemanticEntropyMethod

__all__ = [
    # Base classes
    "UncertaintyMethod",
    "MethodResult",
    # AG-SAR (primary method)
    "AGSARMethod",
    # Single-pass baselines
    "LogProbMethod",
    "PredictiveEntropyMethod",
    # Multi-pass baselines
    "SelfCheckNgramMethod",
    "SelfCheckNLIMethod",
    "SemanticEntropyMethod",
    # Embedding-based baselines
    "EigenScoreMethod",
    "SAPLMAMethod",
    # Hybrid baselines
    "LLMCheckAttentionMethod",
    "LLMCheckHiddenMethod",
    "LLMCheckLogitMethod",
]
