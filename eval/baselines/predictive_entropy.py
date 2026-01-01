"""
Predictive Entropy baseline for uncertainty estimation.

Simple log-probability entropy averaged over response tokens.
Formula: H = -Σ p(t) log p(t) averaged over response tokens
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveEntropy:
    """
    Predictive Entropy baseline.

    Computes average token-level entropy from model logits.
    This is the simplest uncertainty baseline - no semantic weighting.

    Complexity: O(1) model forward pass

    Example:
        >>> pe = PredictiveEntropy(model, tokenizer)
        >>> uncertainty = pe.compute_uncertainty("The capital of France is", "Paris")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Auto-detect device and dtype from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = next(model.parameters()).dtype
        else:
            self.dtype = dtype

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _tokenize(self, prompt: str, response: str) -> Tuple[torch.Tensor, int]:
        """Tokenize and return input_ids with response start index."""
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        full_tokens = prompt_tokens + response_tokens
        response_start = len(prompt_tokens)

        input_ids = torch.tensor([full_tokens], device=self.device)
        return input_ids, response_start

    @torch.inference_mode()
    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        return_per_token: bool = False
    ) -> float:
        """
        Compute predictive entropy for response.

        Args:
            prompt: Input prompt
            response: Generated response
            return_per_token: If True, return per-token entropies

        Returns:
            Average entropy (bits) over response tokens
        """
        input_ids, response_start = self._tokenize(prompt, response)

        # Forward pass
        output = self.model(input_ids=input_ids, return_dict=True)
        logits = output.logits  # (1, seq, vocab)

        # Compute probabilities for response tokens only
        # Note: logits[t] predicts token[t+1], so we use logits[response_start-1:-1]
        response_logits = logits[0, response_start-1:-1, :]  # (response_len, vocab)

        # Softmax to get probabilities
        probs = F.softmax(response_logits, dim=-1)

        # Entropy: H = -Σ p log p
        log_probs = torch.log(probs + 1e-10)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # (response_len,)

        # Convert to bits (base 2)
        token_entropy = token_entropy / torch.log(torch.tensor(2.0, device=self.device))

        if return_per_token:
            return token_entropy.cpu().tolist()

        # Average entropy
        return token_entropy.mean().item()

    @torch.inference_mode()
    def compute_confidence(self, prompt: str, response: str) -> float:
        """
        Compute confidence score (inverse of uncertainty).

        Returns:
            Confidence in [0, 1] range
        """
        entropy = self.compute_uncertainty(prompt, response)

        # Normalize to [0, 1] using sigmoid-like transformation
        # Higher entropy -> lower confidence
        max_entropy = 10.0  # Typical max for GPT-2 vocab
        confidence = 1.0 - min(entropy / max_entropy, 1.0)

        return confidence

    def batch_compute_uncertainty(
        self,
        prompts: list,
        responses: list
    ) -> list:
        """Compute uncertainty for multiple prompt-response pairs."""
        return [
            self.compute_uncertainty(p, r)
            for p, r in zip(prompts, responses)
        ]
