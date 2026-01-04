"""
SelfCheck methods implementing UncertaintyMethod interface.

SelfCheckGPT-Ngram: Generate multiple samples and check n-gram consistency.
This is a SOTA method but requires multiple forward passes (expensive).
"""

import time
from typing import Optional, Set
import torch
import torch.nn as nn

from experiments.methods.base import UncertaintyMethod, MethodResult
from experiments.configs.schema import SelfCheckMethodConfig


class SelfCheckNgramMethod(UncertaintyMethod):
    """
    SelfCheckGPT-Ngram uncertainty method.

    Generates multiple stochastic samples and measures n-gram inconsistency
    between the original response and the samples.

    Key insight: Hallucinated content is less likely to be reproduced
    across multiple samples because it's not grounded in the prompt.

    Interpretation:
        - High score = Many n-grams in response NOT found in samples = Inconsistent = Hallucination
        - Low score = Most n-grams found in samples = Consistent = Factual

    WARNING: This method is ~10x slower than direct methods due to generation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[SelfCheckMethodConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SelfCheck-Ngram.

        Args:
            model: Language model
            tokenizer: Tokenizer
            config: SelfCheck configuration
            device: Compute device
        """
        super().__init__(model, tokenizer, device)

        config = config or SelfCheckMethodConfig()
        self.num_samples = config.num_samples
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.n = 3  # Trigram by default

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def name(self) -> str:
        return "SelfCheck-Ngram"

    @property
    def requires_sampling(self) -> bool:
        return True

    def _get_ngrams(self, text: str, n: int) -> Set[tuple]:
        """Extract n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return set()
        return set(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute SelfCheck-Ngram inconsistency score.

        1. Generate num_samples stochastic responses
        2. Extract n-grams from each
        3. Score = fraction of response n-grams NOT in any sample
        """
        t0 = time.perf_counter()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        try:
            # Generate multiple samples
            gen_outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_len = inputs["input_ids"].size(1)

            # Decode samples
            samples = [
                self.tokenizer.decode(
                    gen_outputs[i, prompt_len:], skip_special_tokens=True
                )
                for i in range(self.num_samples)
            ]

            # Get n-grams from response and samples
            response_ngrams = self._get_ngrams(response, self.n)

            if not response_ngrams:
                # No n-grams to compare
                inconsistency = 0.5
            else:
                # Collect all sample n-grams
                sample_ngrams = set()
                for sample in samples:
                    sample_ngrams.update(self._get_ngrams(sample, self.n))

                # Count inconsistent n-grams
                inconsistent = sum(
                    1 for ng in response_ngrams if ng not in sample_ngrams
                )
                inconsistency = inconsistent / len(response_ngrams)

        except Exception as e:
            # Fallback on generation failure
            inconsistency = 0.5
            samples = []

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=inconsistency,
            confidence=None,
            latency_ms=latency,
            extra={
                "num_samples_generated": len(samples) if "samples" in dir() else 0,
            },
        )


class SelfCheckBertScoreMethod(UncertaintyMethod):
    """
    SelfCheckGPT-BertScore uncertainty method.

    Uses BERTScore for semantic similarity instead of n-gram matching.
    More robust to paraphrasing but slower.

    Note: Requires bert-score package.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[SelfCheckMethodConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)

        config = config or SelfCheckMethodConfig()
        self.num_samples = config.num_samples
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Lazy load BERTScore
        self._bert_scorer = None

    def _get_bert_scorer(self):
        if self._bert_scorer is None:
            from bert_score import BERTScorer

            self._bert_scorer = BERTScorer(
                lang="en", rescale_with_baseline=True, device=str(self.device)
            )
        return self._bert_scorer

    @property
    def name(self) -> str:
        return "SelfCheck-BertScore"

    @property
    def requires_sampling(self) -> bool:
        return True

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute SelfCheck-BertScore inconsistency score.

        Score = 1 - mean(BertScore(response, sample)) across samples
        """
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        try:
            gen_outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            prompt_len = inputs["input_ids"].size(1)

            samples = [
                self.tokenizer.decode(
                    gen_outputs[i, prompt_len:], skip_special_tokens=True
                )
                for i in range(self.num_samples)
            ]

            # Compute BERTScore between response and each sample
            scorer = self._get_bert_scorer()
            _, _, f1_scores = scorer.score(
                [response] * len(samples), samples, verbose=False
            )

            # Inconsistency = 1 - mean similarity
            mean_similarity = f1_scores.mean().item()
            inconsistency = 1.0 - mean_similarity

        except Exception:
            inconsistency = 0.5

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=inconsistency,
            confidence=None,
            latency_ms=latency,
        )
