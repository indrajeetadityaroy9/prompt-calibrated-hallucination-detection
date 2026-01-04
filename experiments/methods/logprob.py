"""
LogProb baseline implementing UncertaintyMethod interface.

Simple baseline: uncertainty = 1 - mean(token_probability)
"""

import time
import torch
import torch.nn as nn
from typing import Optional

from experiments.methods.base import UncertaintyMethod, MethodResult


class LogProbMethod(UncertaintyMethod):
    """
    LogProb uncertainty method.

    Computes uncertainty as 1 - mean(token_probability) over response tokens.
    This is the simplest baseline that uses model confidence directly.

    Interpretation:
        - High score = Low average token probability = More uncertain = Likely hallucination
        - Low score = High average token probability = Confident = Likely factual
    """

    @property
    def name(self) -> str:
        return "LogProb"

    @property
    def requires_sampling(self) -> bool:
        return False

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute LogProb uncertainty score.

        Score = 1 - mean(exp(log_prob)) for response tokens
        """
        t0 = time.perf_counter()

        # Tokenize prompt and full text
        prompt_enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        prompt_ids = prompt_enc["input_ids"].to(self.device)

        full_text = prompt + response
        full_enc = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        full_ids = full_enc["input_ids"].to(self.device)

        response_start = prompt_ids.size(1)

        # Get logits
        outputs = self.model(full_ids)
        logits = outputs.logits

        # Compute log probabilities for response tokens
        # logits[:, i, :] predicts token at position i+1
        log_probs = torch.log_softmax(logits[:, response_start - 1 : -1, :], dim=-1)
        response_tokens = full_ids[:, response_start:]

        # Gather log probs of actual tokens
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

        # Mean probability
        mean_prob = token_log_probs.exp().mean().item()
        uncertainty = 1.0 - mean_prob

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=uncertainty,
            confidence=mean_prob,
            latency_ms=latency,
        )
