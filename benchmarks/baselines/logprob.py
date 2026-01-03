"""
LogProb Baseline: 1 - mean token probability.

The simplest baseline - uses model's own confidence as uncertainty proxy.
Expected to perform poorly on "confident hallucinations".
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class LogProbBaseline:
    """
    Baseline using 1 - mean(token_probability) as hallucination score.

    Higher score = higher uncertainty = more likely hallucination.
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute uncertainty score based on token probabilities.

        Returns:
            Dict with 'score' (uncertainty) and 'confidence' (mean prob)
        """
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        full_text = prompt + response
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        response_start = prompt_ids.size(1)

        # Forward pass
        outputs = self.model(full_ids)
        logits = outputs.logits

        # Get probabilities for response tokens
        log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
        response_tokens = full_ids[:, response_start:]

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

        # Mean probability
        mean_prob = token_log_probs.exp().mean().item()

        # Uncertainty = 1 - confidence
        uncertainty = 1.0 - mean_prob

        return {
            "score": uncertainty,
            "confidence": mean_prob,
            "method": "logprob",
        }
