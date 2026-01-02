"""
Latency profiling for AG-SAR components.

Uses torch.cuda.Event for microsecond precision timing on H100.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from ag_sar.centrality import matrix_free_power_iteration
from ag_sar.uncertainty import compute_graph_shifted_entropy


@dataclass
class LatencyResult:
    """Result of latency measurement."""
    component: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    num_runs: int

    def __repr__(self):
        return f"{self.component}: {self.mean_ms:.3f} +/- {self.std_ms:.3f} ms"


class LatencyProfiler:
    """
    GPU-native latency profiler using CUDA events.

    Example:
        >>> profiler = LatencyProfiler()
        >>> with profiler.measure("forward_pass"):
        ...     output = model(input_ids)
        >>> print(profiler.get_results())
    """

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}
        self._start_events: Dict[str, torch.cuda.Event] = {}
        self._end_events: Dict[str, torch.cuda.Event] = {}

    @contextmanager
    def measure(self, name: str):
        """Context manager for timing a code block."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()

            elapsed_ms = start.elapsed_time(end)
            if name not in self._timings:
                self._timings[name] = []
            self._timings[name].append(elapsed_ms)

    def reset(self):
        """Clear all timing data."""
        self._timings.clear()

    def get_results(self) -> Dict[str, LatencyResult]:
        """Get timing statistics for all measured components."""
        results = {}
        for name, times in self._timings.items():
            times_tensor = torch.tensor(times)
            results[name] = LatencyResult(
                component=name,
                mean_ms=times_tensor.mean().item(),
                std_ms=times_tensor.std().item() if len(times) > 1 else 0.0,
                min_ms=times_tensor.min().item(),
                max_ms=times_tensor.max().item(),
                num_runs=len(times)
            )
        return results

    def get_breakdown(self) -> Dict[str, float]:
        """Get percentage breakdown of total time."""
        results = self.get_results()
        total = sum(r.mean_ms for r in results.values())
        if total == 0:
            return {}
        return {name: (r.mean_ms / total) * 100 for name, r in results.items()}


def profile_ag_sar_components(
    ag_sar,
    prompt: str,
    response: str,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, LatencyResult]:
    """
    Profile AG-SAR pipeline decomposed into components.

    Components measured:
    1. forward_pass: Model forward with hook execution
    2. triton_centrality: Matrix-free centrality via Triton kernel
    3. gse_computation: Graph-Shifted Entropy calculation

    Args:
        ag_sar: AGSAR instance
        prompt: Test prompt
        response: Test response
        num_runs: Number of timing runs
        warmup_runs: Warmup iterations

    Returns:
        Dict mapping component name to LatencyResult
    """
    profiler = LatencyProfiler()
    device = ag_sar.device
    dtype = ag_sar.dtype

    # Tokenize once
    input_ids, attention_mask, response_start = ag_sar._tokenize(prompt, response)

    # Warmup
    for _ in range(warmup_runs):
        ag_sar.compute_uncertainty(prompt, response)

    # Profile each component
    for _ in range(num_runs):
        ag_sar.extractor.clear_cache()

        # 1. Forward pass with hooks (captures Q, K, V norms)
        with profiler.measure("1_forward_pass"):
            with torch.inference_mode():
                output = ag_sar.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    return_dict=True
                )

        # 2. Matrix-free centrality via Triton kernel
        # Stack Q/K from semantic layers
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        num_heads = ag_sar.extractor.num_heads
        head_dim = ag_sar.extractor.head_dim
        num_layers = len(ag_sar._semantic_layer_indices)

        # Create dummy Q/K stacks for profiling
        Q_stack = torch.randn(
            batch_size, num_layers * num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        K_stack = torch.randn(
            batch_size, num_layers * num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )

        with profiler.measure("2_triton_centrality"):
            _ = matrix_free_power_iteration(Q_stack, K_stack, num_iterations=3)

        # 3. GSE computation
        dummy_entropy = torch.rand(batch_size, seq_len, device=device, dtype=dtype)
        dummy_relevance = torch.rand(batch_size, seq_len, device=device, dtype=dtype)

        with profiler.measure("3_gse_computation"):
            _ = compute_graph_shifted_entropy(dummy_entropy, dummy_relevance)

    return profiler.get_results()


def profile_full_pipeline(
    ag_sar,
    prompts: List[str],
    responses: List[str],
    warmup_runs: int = 10
) -> LatencyResult:
    """
    Profile complete AG-SAR pipeline end-to-end.

    Args:
        ag_sar: AGSAR instance
        prompts: List of test prompts
        responses: List of test responses
        warmup_runs: Warmup iterations

    Returns:
        LatencyResult for full pipeline
    """
    profiler = LatencyProfiler()

    # Warmup
    for _ in range(min(warmup_runs, len(prompts))):
        ag_sar.compute_uncertainty(prompts[0], responses[0])

    # Profile
    for prompt, response in zip(prompts, responses):
        with profiler.measure("full_pipeline"):
            ag_sar.compute_uncertainty(prompt, response)

    return profiler.get_results()["full_pipeline"]
