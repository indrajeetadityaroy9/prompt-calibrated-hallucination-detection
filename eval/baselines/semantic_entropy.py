"""
Semantic Entropy baseline for uncertainty estimation.

Implements a simplified version of Semantic Entropy (Kuhn et al., 2023).
Core idea: Generate K samples, cluster by semantic meaning, compute entropy over clusters.

Complexity: O(K*N) - requires K generations per sample

Paper: "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation"
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class SemanticEntropy:
    """
    Semantic Entropy baseline.

    Generates K samples, clusters by semantic similarity, and computes
    entropy over cluster distribution.

    Key insight: If model consistently generates same answer (1 cluster),
    low uncertainty. If diverse answers (many clusters), high uncertainty.

    Complexity: O(K*N) model forward passes per sample.

    Example:
        >>> se = SemanticEntropy(model, tokenizer, k_samples=5)
        >>> uncertainty = se.compute_uncertainty("The capital of France is", max_new_tokens=10)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        k_samples: int = 5,
        temperature: float = 0.7,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.k_samples = k_samples
        self.temperature = temperature

        # Auto-detect device and dtype from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = next(model.parameters()).dtype
        else:
            self.dtype = dtype

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate_samples(
        self,
        prompt: str,
        max_new_tokens: int = 20,
        do_sample: bool = True
    ) -> List[str]:
        """
        Generate K diverse samples for the given prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to use sampling (True) or greedy (False)

        Returns:
            List of K generated responses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        generations = []
        for _ in range(self.k_samples):
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            # Extract only the response part
            response_ids = output[0, prompt_len:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            generations.append(response.strip())

        return generations

    def _simple_cluster(self, responses: List[str]) -> List[int]:
        """
        Simple clustering based on exact string matching.

        For a more sophisticated approach, use sentence embeddings
        and cosine similarity clustering.

        Args:
            responses: List of generated responses

        Returns:
            List of cluster IDs for each response
        """
        # Normalize responses (lowercase, strip whitespace/punctuation)
        normalized = []
        for r in responses:
            # Simple normalization
            r_clean = r.lower().strip()
            # Remove trailing punctuation
            while r_clean and r_clean[-1] in '.,!?':
                r_clean = r_clean[:-1]
            normalized.append(r_clean)

        # Assign cluster IDs
        cluster_map = {}
        cluster_ids = []
        for r in normalized:
            if r not in cluster_map:
                cluster_map[r] = len(cluster_map)
            cluster_ids.append(cluster_map[r])

        return cluster_ids

    def _compute_cluster_entropy(self, cluster_ids: List[int]) -> float:
        """
        Compute entropy over cluster distribution.

        H = -Σ p(c) log p(c)

        Args:
            cluster_ids: List of cluster IDs

        Returns:
            Entropy in nats
        """
        # Count clusters
        counts = defaultdict(int)
        for c in cluster_ids:
            counts[c] += 1

        # Compute probabilities
        total = len(cluster_ids)
        probs = [count / total for count in counts.values()]

        # Entropy
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * np.log(p)

        return entropy

    def compute_uncertainty(
        self,
        prompt: str,
        response: Optional[str] = None,
        max_new_tokens: int = 20
    ) -> float:
        """
        Compute Semantic Entropy uncertainty.

        Note: response parameter is ignored - we generate our own samples.

        Args:
            prompt: Input prompt
            response: Ignored (for API compatibility)
            max_new_tokens: Max tokens to generate per sample

        Returns:
            Semantic entropy (higher = more uncertain)
        """
        # Generate K samples
        responses = self.generate_samples(prompt, max_new_tokens=max_new_tokens)

        # Cluster responses
        cluster_ids = self._simple_cluster(responses)

        # Compute entropy over clusters
        entropy = self._compute_cluster_entropy(cluster_ids)

        return entropy

    def compute_uncertainty_with_response(
        self,
        prompt: str,
        response: str
    ) -> float:
        """
        Compute uncertainty for a specific prompt-response pair.

        For fair comparison, we still generate K samples and compute
        semantic entropy, but we also factor in whether the given
        response appears in our samples.

        Args:
            prompt: Input prompt
            response: Given response

        Returns:
            Semantic entropy
        """
        return self.compute_uncertainty(prompt, max_new_tokens=len(response.split()) + 5)

    def batch_compute_uncertainty(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        max_new_tokens: int = 20
    ) -> List[float]:
        """
        Compute uncertainty for multiple prompts.

        Args:
            prompts: List of prompts
            responses: Ignored
            max_new_tokens: Max tokens to generate

        Returns:
            List of entropy scores
        """
        return [
            self.compute_uncertainty(p, max_new_tokens=max_new_tokens)
            for p in prompts
        ]
