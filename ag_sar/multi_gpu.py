"""
Multi-GPU support for AG-SAR.

Provides utilities for distributing uncertainty computation across multiple GPUs.

Strategies:
1. Data Parallelism: Process different samples on different GPUs
2. Round-Robin: Distribute samples across GPUs in round-robin fashion
3. Batch Sharding: Split a batch across GPUs

Key Classes:
    - MultiGPUAGSAR: Wrapper for multi-GPU inference
    - GPUPool: Manages a pool of AGSAR instances across GPUs
"""

from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import torch
import torch.nn as nn

from .config import AGSARConfig


class GPUPool:
    """
    Manages a pool of models across multiple GPUs.

    Each GPU gets its own model replica for parallel inference.
    Uses ThreadPoolExecutor for concurrent GPU operations.

    Example:
        >>> pool = GPUPool(model_cls, model_name='gpt2', num_gpus=2)
        >>> results = pool.map(compute_fn, samples)
    """

    def __init__(
        self,
        model_class: type,
        model_name: str = 'gpt2',
        num_gpus: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize GPU pool.

        Args:
            model_class: Model class to instantiate (e.g., GPT2LMHeadModel)
            model_name: Pretrained model name
            num_gpus: Number of GPUs to use (default: all available)
            dtype: Model dtype
        """
        self.num_gpus = num_gpus or torch.cuda.device_count()

        if self.num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")

        self.models: List[nn.Module] = []
        self.devices: List[torch.device] = []

        # Create model replica on each GPU
        for i in range(self.num_gpus):
            device = torch.device(f'cuda:{i}')
            self.devices.append(device)

            model = model_class.from_pretrained(model_name)
            model = model.to(device=device, dtype=dtype)
            model.eval()
            self.models.append(model)

        # Thread pool for concurrent execution
        self._executor = ThreadPoolExecutor(max_workers=self.num_gpus)
        self._lock = threading.Lock()

    def get_model(self, gpu_idx: int) -> nn.Module:
        """Get model for a specific GPU."""
        return self.models[gpu_idx]

    def get_device(self, gpu_idx: int) -> torch.device:
        """Get device for a specific GPU."""
        return self.devices[gpu_idx]

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        if hasattr(self, '_executor'):
            self.shutdown()


class MultiGPUAGSAR:
    """
    Multi-GPU wrapper for AG-SAR uncertainty computation.

    Distributes samples across multiple GPUs for parallel processing.
    Each GPU has its own AGSAR instance with dedicated model replica.

    Example:
        >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
        >>> multi_ag_sar = MultiGPUAGSAR.from_pretrained('gpt2', num_gpus=2)
        >>> results = multi_ag_sar.batch_compute_uncertainty(prompts, responses)
    """

    def __init__(
        self,
        models: List[nn.Module],
        tokenizers: List[Any],
        config: Optional[AGSARConfig] = None,
    ):
        """
        Initialize MultiGPUAGSAR with pre-loaded models.

        Args:
            models: List of models, one per GPU
            tokenizers: List of tokenizers (can share same instance)
            config: AG-SAR configuration
        """
        from .ag_sar import AGSAR

        self.num_gpus = len(models)
        self.config = config or AGSARConfig()

        # Create AGSAR instance for each GPU
        self.ag_sar_instances: List[AGSAR] = []
        for model, tokenizer in zip(models, tokenizers):
            ag_sar = AGSAR(model, tokenizer, config)
            self.ag_sar_instances.append(ag_sar)

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=self.num_gpus)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = 'gpt2',
        num_gpus: Optional[int] = None,
        config: Optional[AGSARConfig] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> 'MultiGPUAGSAR':
        """
        Create MultiGPUAGSAR from pretrained model.

        Automatically loads model replicas onto each GPU.

        Args:
            model_name: HuggingFace model name
            num_gpus: Number of GPUs (default: all available)
            config: AG-SAR configuration
            dtype: Model precision

        Returns:
            MultiGPUAGSAR instance
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        num_gpus = num_gpus or torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA GPUs available")

        print(f"Initializing MultiGPUAGSAR with {num_gpus} GPUs...")

        models = []
        tokenizers = []

        # Load tokenizer once (can be shared)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        for i in range(num_gpus):
            device = torch.device(f'cuda:{i}')
            print(f"  Loading model on GPU {i}...")

            model = AutoModelForCausalLM.from_pretrained(model_name)
            model = model.to(device=device, dtype=dtype)
            model.eval()

            models.append(model)
            tokenizers.append(tokenizer)

        return cls(models, tokenizers, config)

    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        gpu_idx: int = 0,
        return_details: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute uncertainty on a specific GPU.

        Args:
            prompt: Input prompt
            response: Generated response
            gpu_idx: Which GPU to use (default: 0)
            return_details: Whether to return detailed results

        Returns:
            GSE score or detailed results dict
        """
        ag_sar = self.ag_sar_instances[gpu_idx % self.num_gpus]
        return ag_sar.compute_uncertainty(prompt, response, return_details)

    def batch_compute_uncertainty(
        self,
        prompts: List[str],
        responses: List[str],
        strategy: str = 'round_robin',
    ) -> List[float]:
        """
        Compute uncertainty for multiple samples across GPUs.

        Args:
            prompts: List of prompts
            responses: List of responses
            strategy: Distribution strategy ('round_robin' or 'chunk')

        Returns:
            List of GSE scores in input order
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")

        n_samples = len(prompts)

        if strategy == 'round_robin':
            return self._batch_round_robin(prompts, responses)
        elif strategy == 'chunk':
            return self._batch_chunk(prompts, responses)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _batch_round_robin(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """Distribute samples round-robin across GPUs."""
        n_samples = len(prompts)
        results = [None] * n_samples
        futures = {}

        def process_sample(idx: int, gpu_idx: int) -> tuple:
            ag_sar = self.ag_sar_instances[gpu_idx]
            result = ag_sar.compute_uncertainty(prompts[idx], responses[idx])
            return idx, result

        # Submit all tasks
        for i in range(n_samples):
            gpu_idx = i % self.num_gpus
            future = self._executor.submit(process_sample, i, gpu_idx)
            futures[future] = i

        # Collect results
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

        return results

    def _batch_chunk(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[float]:
        """Split batch into contiguous chunks per GPU."""
        n_samples = len(prompts)
        chunk_size = (n_samples + self.num_gpus - 1) // self.num_gpus
        results = [None] * n_samples
        futures = {}

        def process_chunk(gpu_idx: int, start: int, end: int) -> List[tuple]:
            ag_sar = self.ag_sar_instances[gpu_idx]
            chunk_results = []
            for i in range(start, min(end, n_samples)):
                result = ag_sar.compute_uncertainty(prompts[i], responses[i])
                chunk_results.append((i, result))
            return chunk_results

        # Submit chunk tasks
        for gpu_idx in range(self.num_gpus):
            start = gpu_idx * chunk_size
            end = start + chunk_size
            if start < n_samples:
                future = self._executor.submit(process_chunk, gpu_idx, start, end)
                futures[future] = gpu_idx

        # Collect results
        for future in as_completed(futures):
            chunk_results = future.result()
            for idx, result in chunk_results:
                results[idx] = result

        return results

    def batch_detect_hallucination(
        self,
        prompts: List[str],
        responses: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect hallucinations in batch across GPUs.

        Args:
            prompts: List of prompts
            responses: List of responses
            threshold: Detection threshold (default: config value)

        Returns:
            List of dicts with 'is_hallucination', 'confidence', 'gse'
        """
        threshold = threshold or self.config.hallucination_threshold
        gse_scores = self.batch_compute_uncertainty(prompts, responses)

        results = []
        for gse in gse_scores:
            is_hall = gse > threshold
            # Confidence: sigmoid distance from threshold
            confidence = 1 / (1 + torch.exp(torch.tensor(-(gse - threshold)))).item()
            results.append({
                'is_hallucination': is_hall,
                'confidence': confidence,
                'gse': gse,
            })

        return results

    def cleanup(self) -> None:
        """Clean up all AGSAR instances."""
        for ag_sar in self.ag_sar_instances:
            ag_sar.cleanup()
        self._executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return (
            f"MultiGPUAGSAR(num_gpus={self.num_gpus}, "
            f"threshold={self.config.hallucination_threshold})"
        )


def get_optimal_gpu_count() -> int:
    """
    Get optimal number of GPUs for current workload.

    Returns the number of available GPUs, or 1 if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def distribute_samples(
    samples: List[Any],
    num_gpus: int,
    strategy: str = 'round_robin'
) -> List[List[tuple]]:
    """
    Distribute samples across GPUs.

    Args:
        samples: List of samples to distribute
        num_gpus: Number of GPUs
        strategy: 'round_robin' or 'chunk'

    Returns:
        List of (gpu_idx, sample_idx, sample) assignments per GPU
    """
    assignments = [[] for _ in range(num_gpus)]

    if strategy == 'round_robin':
        for i, sample in enumerate(samples):
            gpu_idx = i % num_gpus
            assignments[gpu_idx].append((i, sample))
    elif strategy == 'chunk':
        chunk_size = (len(samples) + num_gpus - 1) // num_gpus
        for gpu_idx in range(num_gpus):
            start = gpu_idx * chunk_size
            end = min(start + chunk_size, len(samples))
            for i in range(start, end):
                assignments[gpu_idx].append((i, samples[i]))

    return assignments
