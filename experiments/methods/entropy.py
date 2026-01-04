"""
Predictive Entropy baseline implementing UncertaintyMethod interface.

Uses Shannon entropy of the softmax distribution as uncertainty measure.
Better than LogProb because it captures the full distribution shape.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from experiments.methods.base import UncertaintyMethod, MethodResult


class PredictiveEntropyMethod(UncertaintyMethod):
    """
    Predictive Entropy uncertainty method.

    Computes uncertainty as normalized Shannon entropy of the softmax distribution.
    H = -sum(p * log(p)) / log(vocab_size)

    Advantages over LogProb:
        - Captures full distribution shape, not just peak probability
        - Differentiates between confident-wrong and uncertain-wrong
        - High entropy = flat distribution = model is unsure
        - Low entropy = peaked distribution = model is confident

    Interpretation:
        - High score = High entropy = Uncertain = Likely hallucination
        - Low score = Low entropy = Confident = Likely factual
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)

        # Get vocab size for normalization
        if hasattr(self.model.config, "vocab_size"):
            self.vocab_size = self.model.config.vocab_size
        else:
            self.vocab_size = len(tokenizer)

        self.max_entropy = torch.log(torch.tensor(self.vocab_size, dtype=torch.float32))

    @property
    def name(self) -> str:
        return "Entropy"

    @property
    def requires_sampling(self) -> bool:
        return False

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute predictive entropy uncertainty score.

        Score = mean(H_normalized) for response tokens
        where H_normalized = -sum(p * log(p)) / log(vocab_size)
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

        # Get logits for response positions
        response_logits = logits[:, response_start - 1 : -1, :]

        # Compute softmax probabilities
        probs = F.softmax(response_logits, dim=-1)

        # Compute Shannon entropy: H = -sum(p * log(p))
        log_probs = F.log_softmax(response_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)

        # Normalize by max entropy
        max_ent = self.max_entropy.to(self.device)
        normalized_entropy = entropy / max_ent

        # Mean over response tokens
        mean_entropy = normalized_entropy.mean().item()

        # Also compute mean probability for confidence
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
        mean_prob = token_log_probs.exp().mean().item()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=mean_entropy,
            confidence=mean_prob,
            latency_ms=latency,
            extra={
                "raw_entropy": entropy.mean().item(),
            },
        )
