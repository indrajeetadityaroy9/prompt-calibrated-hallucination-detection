"""
Base interface for signal computation.
Enforces strict typing and standardized signatures for the AG-SAR pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
import torch
from torch import Tensor

from ..hooks import LayerHiddenStates

class BaseSignal(ABC):
    """
    Abstract base class for all signal computations.

    All signals must:
    1. Inherit from this class.
    2. Implement compute() with the standardized signature.
    3. Return a torch.Tensor (scalar or vector).
    """

    @abstractmethod
    def compute(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        logits: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute the signal value.

        Args:
            layer_states: Dictionary mapping layer indices to hidden states.
                          Captured by LayerHooks during the forward pass.
            logits: Final output logits from the model head [vocab_size] or [batch, vocab].
            **kwargs: Additional context required for specific signals:
                      - emitted_token_id (int/Tensor): The selected token.
                      - candidate_set (Tensor): Top-k candidate indices.
                      - token_hidden (Tensor): Specific hidden state for context support.

        Returns:
            Tensor: The computed signal value. Should be a scalar (0-d) for single-token
                    computations, or 1-d for batch/candidate computations.
        """
        pass
