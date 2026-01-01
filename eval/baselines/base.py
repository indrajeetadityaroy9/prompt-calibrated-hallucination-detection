"""
Base class for uncertainty quantification baselines.

Provides common functionality for tokenization, device handling,
and interface contract for uncertainty computation methods.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class UncertaintyBaseline(ABC):
    """
    Abstract base class for uncertainty quantification baselines.

    All baseline methods should inherit from this class to ensure
    consistent interface and shared functionality.

    Attributes:
        model: The language model
        tokenizer: Corresponding tokenizer
        device: Computation device (auto-detected from model)
        dtype: Tensor dtype (auto-detected from model)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize baseline.

        Args:
            model: HuggingFace model (GPT2LMHeadModel or similar)
            tokenizer: Corresponding tokenizer
            device: Computation device (auto-detected if None)
            dtype: Tensor dtype (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer

        # Auto-detect device and dtype from model if not specified
        self.device = (
            torch.device(device) if device is not None
            else next(model.parameters()).device
        )
        self.dtype = dtype if dtype is not None else next(model.parameters()).dtype

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenize prompt and response with proper attention mask.

        Args:
            prompt: Input prompt text
            response: Generated response text

        Returns:
            input_ids: (1, seq_len) token IDs
            attention_mask: (1, seq_len) attention mask
            response_start: Index where response begins
        """
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        full_tokens = prompt_tokens + response_tokens
        response_start = len(prompt_tokens)

        input_ids = torch.tensor([full_tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, response_start

    @abstractmethod
    def compute_uncertainty(self, prompt: str, response: str) -> float:
        """
        Compute uncertainty score for prompt-response pair.

        Higher scores indicate higher uncertainty (less confidence).

        Args:
            prompt: Input prompt text
            response: Generated response text

        Returns:
            Uncertainty score (higher = more uncertain)
        """
        pass

    def compute_confidence(self, prompt: str, response: str) -> float:
        """
        Compute confidence score (inverse of uncertainty).

        Default implementation uses sigmoid transformation.
        Subclasses may override for method-specific normalization.

        Args:
            prompt: Input prompt text
            response: Generated response text

        Returns:
            Confidence score in [0, 1] range (higher = more confident)
        """
        uncertainty = self.compute_uncertainty(prompt, response)
        # Sigmoid-like transformation to [0, 1]
        return 1.0 / (1.0 + uncertainty)

    def batch_compute_uncertainty(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> List[float]:
        """
        Compute uncertainty for multiple prompt-response pairs.

        Default implementation processes sequentially.
        Subclasses may override for true batching.

        Args:
            prompts: List of prompt texts
            responses: List of response texts

        Returns:
            List of uncertainty scores
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")
        return [
            self.compute_uncertainty(p, r)
            for p, r in zip(prompts, responses)
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"dtype={self.dtype})"
        )
