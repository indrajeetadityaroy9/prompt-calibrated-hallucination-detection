"""
SelfCheck methods implementing UncertaintyMethod interface.

Phase 2.1 Requirements:
- Uses sentence_transformers.CrossEncoder with pretrained NLI model
- NO training loops
- NO custom weight loading
- Inherits global seed from Phase 0

Reference: Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection"
"""

import re
import time
from typing import Optional, Set, List
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SelfCheckNLIMethod(UncertaintyMethod):
    """
    SelfCheckGPT-NLI uncertainty method using CrossEncoder.

    Phase 2.1 Compliant:
    - Uses sentence_transformers.CrossEncoder (pretrained NLI only)
    - NO training loops
    - NO custom weight loading
    - Inherits global seed from Phase 0

    Generates multiple samples from the model and computes consistency
    between the original response and sampled responses using NLI.

    Interpretation:
        - High score = Low consistency = Likely hallucination
        - Low score = High consistency = Likely factual
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[SelfCheckMethodConfig] = None,
        device: Optional[torch.device] = None,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        seed: int = 42,
    ):
        """
        Initialize SelfCheckGPT-NLI method.

        Args:
            model: The language model for generation
            tokenizer: Tokenizer for the model
            config: SelfCheck configuration
            device: Device to run on
            nli_model: Pretrained NLI model from sentence-transformers
            seed: Random seed for reproducibility (Phase 0.2)
        """
        super().__init__(model, tokenizer, device)

        config = config or SelfCheckMethodConfig()
        self.num_samples = config.num_samples
        self.max_new_tokens = config.max_new_tokens
        self.temperature = config.temperature
        self.seed = seed

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Phase 2.1: Use pretrained NLI model only (no training, no custom weights)
        from sentence_transformers import CrossEncoder

        self._nli_model = CrossEncoder(nli_model, device=str(self.device))

        # Set seed for reproducibility (Phase 0.2)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def name(self) -> str:
        return "SelfCheck-NLI"

    @property
    def requires_sampling(self) -> bool:
        return True

    def _generate_samples(self, prompt: str) -> List[str]:
        """Generate multiple samples from the model."""
        # Set seed before each generation batch for reproducibility
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.max_new_tokens,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].size(1)
        samples = [
            self.tokenizer.decode(
                outputs[i, prompt_len:], skip_special_tokens=True
            ).strip()
            for i in range(self.num_samples)
        ]

        return samples

    def _compute_nli_score(self, original: str, samples: List[str]) -> float:
        """
        Compute NLI-based inconsistency score.

        Returns the average contradiction probability across samples.
        Higher score = more inconsistency = more likely hallucination.
        """
        if not samples or not original.strip():
            return float("nan")

        # Split original into sentences
        sentences = re.split(r"(?<=[.!?])\s+", original.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return float("nan")

        total_contradiction = 0.0
        total_pairs = 0

        for sentence in sentences:
            for sample in samples:
                if not sample.strip():
                    continue

                # NLI: (premise, hypothesis)
                # Label order: contradiction=0, entailment=1, neutral=2
                pair = [(sample, sentence)]

                try:
                    # CrossEncoder predict returns logits
                    logits = self._nli_model.predict(pair, convert_to_numpy=True)

                    if len(logits.shape) == 1:
                        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
                    else:
                        probs = F.softmax(torch.tensor(logits[0]), dim=-1).numpy()

                    # Contradiction probability (label 0)
                    contradiction_prob = float(probs[0])
                    total_contradiction += contradiction_prob
                    total_pairs += 1
                except Exception:
                    continue

        if total_pairs == 0:
            return float("nan")

        return total_contradiction / total_pairs

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute SelfCheckGPT-NLI uncertainty score.

        Score = average contradiction probability between original response
        and sampled responses.
        """
        t0 = time.perf_counter()

        # Handle empty response (Phase 1.5: NaN for empty)
        if not response.strip():
            latency = (time.perf_counter() - t0) * 1000
            return MethodResult(
                score=float("nan"),
                confidence=0.0,
                latency_ms=latency,
                extra={"status": "DROP", "reason": "empty_response"},
            )

        # Generate samples
        try:
            samples = self._generate_samples(prompt)
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return MethodResult(
                score=float("nan"),
                confidence=0.0,
                latency_ms=latency,
                extra={"status": "DROP", "reason": f"generation_error: {str(e)}"},
            )

        # Compute NLI consistency score
        score = self._compute_nli_score(response, samples)

        latency = (time.perf_counter() - t0) * 1000

        # Confidence is inverse of contradiction (higher consistency = higher confidence)
        confidence = 1.0 - score if not (score != score) else 0.0  # NaN check

        return MethodResult(
            score=score,
            confidence=confidence,
            latency_ms=latency,
            extra={
                "num_samples": len(samples),
                "sample_lengths": [len(s) for s in samples],
            },
        )
