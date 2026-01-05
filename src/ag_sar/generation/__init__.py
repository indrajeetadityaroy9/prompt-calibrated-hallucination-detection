"""
AG-SAR Generation Module: Active Hallucination Prevention.

This module transforms AG-SAR from a passive detector into an active controller
that guides text generation away from hallucination-prone paths.

Key Components:
    - AGSARGuidedGenerator: Stepwise Best-of-N search with trust-based selection
    - GenerationStep: Dataclass recording each step's candidates and decisions

Example:
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> from ag_sar.generation import AGSARGuidedGenerator
    >>>
    >>> config = AGSARConfig(enable_intrinsic_detection=True,
    ...                      truth_vector_path="data/truth_vectors/llama.pt")
    >>> engine = AGSAR(model, tokenizer, config)
    >>> generator = AGSARGuidedGenerator(model, tokenizer, engine)
    >>>
    >>> response = generator.generate(
    ...     "What is the capital of France?",
    ...     step_size=15,
    ...     num_candidates=3
    ... )
"""

from .guided_decoding import AGSARGuidedGenerator, GenerationStep

__all__ = ["AGSARGuidedGenerator", "GenerationStep"]
