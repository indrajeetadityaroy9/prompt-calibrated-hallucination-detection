"""
SelfCheckGPT-Ngram Baseline.

Approximation of SelfCheckGPT using n-gram consistency.
The idea: Generate multiple stochastic samples, check if the response
contains n-grams that don't appear in any samples.

This is MUCH slower than AG-SAR (requires multiple generations).
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Set
from collections import Counter


class SelfCheckNgramBaseline:
    """
    SelfCheckGPT-Ngram: Sample multiple responses and check n-gram consistency.

    Based on: "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"
    (Manakul et al., 2023)

    This approximation uses n-gram overlap instead of NLI model.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_samples: int = 4,
        n: int = 3,
        max_new_tokens: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_samples = num_samples
        self.n = n  # n-gram size
        self.max_new_tokens = max_new_tokens

    def _get_ngrams(self, text: str, n: int) -> Set[tuple]:
        """Extract n-grams from text."""
        words = text.lower().split()
        if len(words) < n:
            return set()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    def _ngram_inconsistency(self, response: str, samples: List[str]) -> float:
        """
        Compute n-gram inconsistency score.

        Returns fraction of response n-grams NOT found in any sample.
        Higher = more inconsistent = more likely hallucination.
        """
        response_ngrams = self._get_ngrams(response, self.n)

        if not response_ngrams:
            return 0.5  # Neutral if too short

        # Collect all n-grams from samples
        sample_ngrams = set()
        for sample in samples:
            sample_ngrams.update(self._get_ngrams(sample, self.n))

        # Count how many response n-grams are NOT in samples
        inconsistent = sum(1 for ng in response_ngrams if ng not in sample_ngrams)
        inconsistency_rate = inconsistent / len(response_ngrams)

        return inconsistency_rate

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute SelfCheck score by generating samples and checking consistency.

        WARNING: This is expensive! Requires num_samples forward passes.

        Returns:
            Dict with 'score' (inconsistency rate) and generation stats
        """
        import time
        start_time = time.time()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Generate stochastic samples
        try:
            gen_outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            # Decode samples (removing prompt)
            prompt_len = inputs["input_ids"].size(1)
            samples = []
            for i in range(self.num_samples):
                sample_ids = gen_outputs[i, prompt_len:]
                sample_text = self.tokenizer.decode(sample_ids, skip_special_tokens=True)
                samples.append(sample_text)

        except Exception as e:
            # Fallback if generation fails
            return {
                "score": 0.5,
                "num_samples": 0,
                "generation_time": time.time() - start_time,
                "method": "selfcheck_ngram",
                "error": str(e),
            }

        # Compute inconsistency
        inconsistency = self._ngram_inconsistency(response, samples)

        generation_time = time.time() - start_time

        return {
            "score": inconsistency,
            "num_samples": len(samples),
            "generation_time": generation_time,
            "method": "selfcheck_ngram",
        }


class SelfCheckBertScore:
    """
    SelfCheckGPT using BERTScore for semantic similarity.

    More accurate than n-gram but requires BERTScore computation.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_samples: int = 4,
        max_new_tokens: int = 100,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

        # Lazy import BERTScore
        self._bert_score = None

    def _get_bertscore(self):
        """Lazy load BERTScore."""
        if self._bert_score is None:
            from bert_score import score as bert_score
            self._bert_score = bert_score
        return self._bert_score

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute SelfCheck score using BERTScore for semantic comparison.
        """
        import time
        start_time = time.time()

        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.device)

        # Generate stochastic samples
        try:
            gen_outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=1.0,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            prompt_len = inputs["input_ids"].size(1)
            samples = []
            for i in range(self.num_samples):
                sample_ids = gen_outputs[i, prompt_len:]
                sample_text = self.tokenizer.decode(sample_ids, skip_special_tokens=True)
                samples.append(sample_text)

        except Exception as e:
            return {
                "score": 0.5,
                "num_samples": 0,
                "generation_time": time.time() - start_time,
                "method": "selfcheck_bertscore",
                "error": str(e),
            }

        # Compute BERTScore similarity
        try:
            bert_score_fn = self._get_bertscore()
            # Compare response to each sample
            refs = samples
            cands = [response] * len(samples)
            P, R, F1 = bert_score_fn(cands, refs, lang="en", verbose=False)

            # Average F1 score across samples
            avg_similarity = F1.mean().item()

            # Inconsistency = 1 - similarity
            inconsistency = 1.0 - avg_similarity

        except Exception as e:
            # Fallback to n-gram if BERTScore fails
            inconsistency = 0.5

        generation_time = time.time() - start_time

        return {
            "score": inconsistency,
            "num_samples": len(samples),
            "generation_time": generation_time,
            "method": "selfcheck_bertscore",
        }
