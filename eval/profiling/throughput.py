"""
Throughput benchmarking for AG-SAR vs baselines.

Measures Tokens Per Second (TPS) across methods.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import torch


@dataclass
class ThroughputResult:
    """Result of throughput benchmark."""
    method: str
    tokens_per_second: float
    samples_per_second: float
    total_tokens: int
    total_samples: int
    total_time_seconds: float
    avg_tokens_per_sample: float

    def __repr__(self):
        return f"{self.method}: {self.tokens_per_second:.1f} TPS ({self.samples_per_second:.2f} samples/s)"


class ThroughputBenchmark:
    """
    Throughput benchmarking utility.

    Compares:
    1. Vanilla GPT-2 (no uncertainty)
    2. AG-SAR (internal graph)
    3. Original SAR (GPT-2 + RoBERTa perturbation)

    Example:
        >>> benchmark = ThroughputBenchmark(model, tokenizer)
        >>> results = benchmark.run(prompts, responses)
        >>> print(results['ag_sar'].tokens_per_second)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        warmup_samples: int = 10
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.warmup_samples = warmup_samples

    def _count_tokens(self, prompt: str, response: str) -> int:
        """Count tokens in prompt + response."""
        return len(self.tokenizer.encode(prompt + response))

    def benchmark_vanilla(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> ThroughputResult:
        """
        Benchmark vanilla model forward pass (no uncertainty).
        """
        # Warmup
        for i in range(min(self.warmup_samples, len(prompts))):
            input_ids = self.tokenizer.encode(
                prompts[i] + responses[i], return_tensors='pt'
            ).to(self.device)
            with torch.inference_mode():
                _ = self.model(input_ids)

        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        for prompt, response in zip(prompts, responses):
            input_ids = self.tokenizer.encode(
                prompt + response, return_tensors='pt'
            ).to(self.device)

            with torch.inference_mode():
                _ = self.model(input_ids)

            total_tokens += input_ids.size(1)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        return ThroughputResult(
            method="vanilla_gpt2",
            tokens_per_second=total_tokens / total_time,
            samples_per_second=len(prompts) / total_time,
            total_tokens=total_tokens,
            total_samples=len(prompts),
            total_time_seconds=total_time,
            avg_tokens_per_sample=total_tokens / len(prompts)
        )

    def benchmark_ag_sar(
        self,
        ag_sar,
        prompts: List[str],
        responses: List[str]
    ) -> ThroughputResult:
        """
        Benchmark AG-SAR pipeline.
        """
        # Warmup
        for i in range(min(self.warmup_samples, len(prompts))):
            ag_sar.compute_uncertainty(prompts[i], responses[i])

        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        for prompt, response in zip(prompts, responses):
            ag_sar.compute_uncertainty(prompt, response)
            total_tokens += self._count_tokens(prompt, response)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        return ThroughputResult(
            method="ag_sar",
            tokens_per_second=total_tokens / total_time,
            samples_per_second=len(prompts) / total_time,
            total_tokens=total_tokens,
            total_samples=len(prompts),
            total_time_seconds=total_time,
            avg_tokens_per_sample=total_tokens / len(prompts)
        )

    def benchmark_original_sar(
        self,
        original_sar,
        prompts: List[str],
        responses: List[str]
    ) -> ThroughputResult:
        """
        Benchmark Original SAR (O(N) RoBERTa passes per sample).
        """
        # Warmup
        for i in range(min(self.warmup_samples, len(prompts))):
            original_sar.compute_uncertainty(prompts[i], responses[i])

        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        for prompt, response in zip(prompts, responses):
            original_sar.compute_uncertainty(prompt, response)
            total_tokens += self._count_tokens(prompt, response)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        return ThroughputResult(
            method="original_sar",
            tokens_per_second=total_tokens / total_time,
            samples_per_second=len(prompts) / total_time,
            total_tokens=total_tokens,
            total_samples=len(prompts),
            total_time_seconds=total_time,
            avg_tokens_per_sample=total_tokens / len(prompts)
        )

    def benchmark_predictive_entropy(
        self,
        pe_baseline,
        prompts: List[str],
        responses: List[str]
    ) -> ThroughputResult:
        """
        Benchmark Predictive Entropy baseline.
        """
        # Warmup
        for i in range(min(self.warmup_samples, len(prompts))):
            pe_baseline.compute_uncertainty(prompts[i], responses[i])

        torch.cuda.synchronize()

        # Benchmark
        total_tokens = 0
        start = time.perf_counter()

        for prompt, response in zip(prompts, responses):
            pe_baseline.compute_uncertainty(prompt, response)
            total_tokens += self._count_tokens(prompt, response)

        torch.cuda.synchronize()
        total_time = time.perf_counter() - start

        return ThroughputResult(
            method="predictive_entropy",
            tokens_per_second=total_tokens / total_time,
            samples_per_second=len(prompts) / total_time,
            total_tokens=total_tokens,
            total_samples=len(prompts),
            total_time_seconds=total_time,
            avg_tokens_per_sample=total_tokens / len(prompts)
        )

    def run_all(
        self,
        ag_sar,
        original_sar,
        pe_baseline,
        prompts: List[str],
        responses: List[str]
    ) -> Dict[str, ThroughputResult]:
        """
        Run all throughput benchmarks.

        Returns:
            Dict mapping method name to ThroughputResult
        """
        results = {}

        print("Benchmarking Vanilla GPT-2...")
        results['vanilla'] = self.benchmark_vanilla(prompts, responses)

        print("Benchmarking AG-SAR...")
        results['ag_sar'] = self.benchmark_ag_sar(ag_sar, prompts, responses)

        print("Benchmarking Predictive Entropy...")
        results['predictive_entropy'] = self.benchmark_predictive_entropy(
            pe_baseline, prompts, responses
        )

        print("Benchmarking Original SAR (O(N) RoBERTa passes)...")
        results['original_sar'] = self.benchmark_original_sar(
            original_sar, prompts, responses
        )

        return results


def compute_speedup(results: Dict[str, ThroughputResult]) -> Dict[str, float]:
    """
    Compute speedup ratios relative to Original SAR.

    Returns:
        Dict mapping method to speedup factor
    """
    baseline_tps = results['original_sar'].tokens_per_second
    return {
        method: result.tokens_per_second / baseline_tps
        for method, result in results.items()
    }
