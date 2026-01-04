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

        # Tokenize full text with offset mapping to find correct response boundary
        # NOTE: BPE tokenizers may merge tokens at prompt/response boundary,
        # so we use offset_mapping to find the first token that starts at or after
        # the prompt end position.
        full_text = prompt + response
        full_enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            return_offsets_mapping=True,
        )
        full_ids = full_enc["input_ids"].to(self.device)
        offsets = full_enc["offset_mapping"][0]  # List of (start, end) tuples

        # Find the first token that starts at or after the prompt length
        prompt_len = len(prompt)
        response_start = None
        for i, (start, end) in enumerate(offsets):
            if start >= prompt_len:
                response_start = i
                break

        # Fallback: if no token starts exactly at prompt_len, find first token
        # that contains characters from the response
        if response_start is None:
            for i, (start, end) in enumerate(offsets):
                if end > prompt_len:
                    response_start = i
                    break

        # If still None, response may be empty
        if response_start is None or response_start >= full_ids.size(1):
            response_start = full_ids.size(1)

        response_tokens = full_ids[:, response_start:]

        # Handle empty response tokens (return NaN sentinel for DROP strategy)
        if response_tokens.numel() == 0:
            latency = (time.perf_counter() - t0) * 1000
            return MethodResult(
                score=float('nan'),
                confidence=0.0,
                latency_ms=latency,
                extra={"status": "DROP", "reason": "empty_response_tokens"},
            )

        # Get logits
        outputs = self.model(full_ids)
        logits = outputs.logits

        # Compute log probabilities for response tokens
        # logits[:, i, :] predicts token at position i+1
        log_probs = torch.log_softmax(logits[:, response_start - 1 : -1, :], dim=-1)

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
