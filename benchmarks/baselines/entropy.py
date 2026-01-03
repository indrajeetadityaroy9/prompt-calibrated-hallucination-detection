"""
Predictive Entropy Baseline.

Uses Shannon entropy of the softmax distribution as uncertainty measure.
Better than LogProb because it captures the full distribution shape.

H = -sum(p * log(p))

Higher entropy = more uncertain = more likely hallucination.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class PredictiveEntropyBaseline:
    """
    Baseline using predictive entropy (Shannon entropy of softmax).

    This is considered a "better" simple baseline than raw logprob
    because it captures uncertainty in the full distribution.
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute predictive entropy over response tokens.

        Returns:
            Dict with 'score' (normalized entropy) and 'raw_entropy'
        """
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        full_text = prompt + response
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        response_start = prompt_ids.size(1)

        # Forward pass
        outputs = self.model(full_ids)
        logits = outputs.logits

        # Get softmax probabilities for response token positions
        # logits[:, response_start-1:-1] predicts tokens at positions response_start:end
        probs = torch.softmax(logits[:, response_start-1:-1, :], dim=-1)

        # Shannon entropy: H = -sum(p * log(p))
        # Add small epsilon for numerical stability
        log_probs = torch.log(probs + 1e-10)
        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)  # (B, T)

        # Mean entropy across response tokens
        mean_entropy = entropy_per_token.mean().item()

        # Normalize by max possible entropy (log(vocab_size))
        vocab_size = logits.size(-1)
        max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32)).item()
        normalized_entropy = mean_entropy / max_entropy

        # Also compute model confidence for comparison
        response_tokens = full_ids[:, response_start:]
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
        model_confidence = token_log_probs.exp().mean().item()

        return {
            "score": normalized_entropy,  # 0-1 range
            "raw_entropy": mean_entropy,
            "confidence": model_confidence,
            "method": "entropy",
        }
