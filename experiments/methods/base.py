"""
Abstract base class for uncertainty quantification methods.

All methods (AG-SAR, baselines) implement this interface for uniform comparison.
The interface is designed to be simple and consistent across all methods.

IMPORTANT: MethodResult.extra must contain only JSON-serializable scalars/strings
to prevent memory bloat during large-scale evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


@dataclass
class MethodResult:
    """
    Standardized result from any uncertainty method.

    Attributes:
        score: Uncertainty score in [0, 1] where 0=confident, 1=uncertain
        confidence: Model confidence (e.g., mean token probability) if available
        latency_ms: Time taken for computation in milliseconds
        extra: Method-specific metadata (MUST be JSON-serializable scalars only!)

    WARNING: Do not store tensors, numpy arrays, or large objects in `extra`.
    The BenchmarkEngine accumulates results and will OOM on large datasets.
    """

    score: float
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that extra contains only JSON-serializable types."""
        if self.extra:
            for key, value in self.extra.items():
                if not isinstance(value, (str, int, float, bool, type(None))):
                    raise ValueError(
                        f"MethodResult.extra['{key}'] must be JSON-serializable "
                        f"(str/int/float/bool/None), got {type(value).__name__}. "
                        "Do NOT store tensors or large objects to prevent OOM."
                    )


class UncertaintyMethod(ABC):
    """
    Abstract base class for uncertainty quantification methods.

    All methods implement compute_score() with the same signature.
    The requires_sampling property indicates if the method needs
    multiple forward passes (e.g., SelfCheck) vs single pass (e.g., AG-SAR).

    Lifecycle:
        1. __init__(model, tokenizer) - Initialize with shared model
        2. compute_score(prompt, response) - Compute uncertainty (can be called many times)
        3. cleanup() - Release resources (hooks, caches) when done

    Example:
        >>> method = AGSARMethod(model, tokenizer, config)
        >>> for sample in dataset:
        ...     result = method.compute_score(sample.prompt, sample.response)
        ...     print(f"Uncertainty: {result.score:.3f}")
        >>> method.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the method with a model and tokenizer.

        Args:
            model: Language model (must already be loaded and on device)
            tokenizer: Tokenizer for the model
            device: Compute device (auto-detected from model if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name for the method.

        Used in logging and result files. Should be short and unique.
        Examples: "AG-SAR", "LogProb", "SelfCheck-Ngram"
        """
        pass

    @property
    @abstractmethod
    def requires_sampling(self) -> bool:
        """
        Whether the method requires multiple stochastic samples.

        Returns:
            True if method generates samples (e.g., SelfCheck needs 5 generations)
            False if method uses single forward pass (e.g., AG-SAR, LogProb)

        This is used by the BenchmarkEngine to:
        - Estimate runtime (sampling methods are ~10x slower)
        - Potentially run non-sampling methods in parallel
        """
        pass

    @abstractmethod
    def compute_score(
        self,
        prompt: str,
        response: str,
    ) -> MethodResult:
        """
        Compute uncertainty score for a prompt-response pair.

        This is the core method that all implementations must provide.
        It should be deterministic for non-sampling methods.

        Args:
            prompt: Input prompt (may include context, question, etc.)
            response: Generated response to evaluate for hallucination

        Returns:
            MethodResult with:
                - score: Uncertainty in [0, 1] (higher = more likely hallucinated)
                - confidence: Model confidence if available
                - latency_ms: Computation time
                - extra: Method-specific metadata (JSON-serializable only!)

        Raises:
            ValueError: If prompt or response is empty
            RuntimeError: If model inference fails
        """
        pass

    def cleanup(self) -> None:
        """
        Release any resources held by the method.

        This is critical for methods that register hooks (e.g., AG-SAR).
        Must be called before switching to another method on the same model
        to prevent hook conflicts.

        Default implementation does nothing. Override if needed.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', requires_sampling={self.requires_sampling})"
