"""
Semantic Entropy: Uncertainty via Semantic Clustering.

Official implementation based on:
"Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in NLG"
Kuhn et al., ICLR 2023
https://arxiv.org/abs/2302.09664

Key insight: Multiple samples may express the same semantic meaning.
Clustering by semantic similarity before computing entropy gives a
better measure of true uncertainty than raw sample entropy.

Example:
    "Paris is the capital" and "The capital is Paris" are semantically
    equivalent but lexically different. Naive entropy would overcout this
    as two distinct meanings.
"""

import time
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn

from experiments.methods.base import UncertaintyMethod, MethodResult


class SemanticEntropyMethod(UncertaintyMethod):
    """
    Semantic Entropy uncertainty method (Kuhn et al., ICLR 2023).

    Generates multiple stochastic responses, clusters them by semantic
    similarity using sentence embeddings, and computes entropy over
    the cluster distribution.

    High entropy = Many distinct semantic meanings = High uncertainty.
    Low entropy = All samples express similar meaning = Low uncertainty.

    Warning: This method requires ~5 generations per sample, making it
    significantly slower than single-pass methods like AG-SAR.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_samples: int = 5,
        similarity_threshold: float = 0.8,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Semantic Entropy method.

        Args:
            model: Language model for generation
            tokenizer: Tokenizer for the model
            num_samples: Number of stochastic samples to generate (default: 5)
            similarity_threshold: Cosine similarity threshold for clustering (default: 0.8)
            embedding_model: Sentence-transformer model for embeddings
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature for diversity
            device: Compute device
        """
        super().__init__(model, tokenizer, device)

        self.num_samples = num_samples
        self.similarity_threshold = similarity_threshold
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Ensure pad token is set (required for batch generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load sentence embedding model (lightweight, ~80MB)
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(embedding_model, device=str(self.device))

    @property
    def name(self) -> str:
        return "SemanticEntropy"

    @property
    def requires_sampling(self) -> bool:
        return True  # Requires multiple generations

    def _generate_samples(self, prompt: str) -> List[str]:
        """
        Generate multiple stochastic samples from the model.

        Args:
            prompt: Input prompt

        Returns:
            List of generated response strings
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - self.max_new_tokens,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.9,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].size(1)

        # Decode and strip prompt
        samples = []
        for i in range(self.num_samples):
            response = self.tokenizer.decode(
                outputs[i, prompt_len:], skip_special_tokens=True
            ).strip()
            samples.append(response)

        return samples

    def _cluster_by_similarity(self, samples: List[str]) -> List[List[int]]:
        """
        Cluster samples by semantic similarity using greedy agglomerative clustering.

        Args:
            samples: List of text samples

        Returns:
            List of clusters, where each cluster is a list of sample indices
        """
        if not samples:
            return []

        # Embed all samples
        embeddings = self._embedder.encode(samples, convert_to_tensor=True)
        embeddings = embeddings.cpu().numpy()

        # Greedy agglomerative clustering
        # Start with first sample, greedily add to existing cluster or create new
        clusters: List[List[int]] = []
        assigned = set()

        for i in range(len(samples)):
            if i in assigned:
                continue

            # Start new cluster with sample i
            current_cluster = [i]
            assigned.add(i)

            # Find all unassigned samples similar to i
            for j in range(i + 1, len(samples)):
                if j in assigned:
                    continue

                # Cosine similarity
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )

                if sim >= self.similarity_threshold:
                    current_cluster.append(j)
                    assigned.add(j)

            clusters.append(current_cluster)

        return clusters

    def _compute_semantic_entropy(self, clusters: List[List[int]], n_samples: int) -> float:
        """
        Compute entropy over cluster distribution.

        Args:
            clusters: List of clusters (each is list of sample indices)
            n_samples: Total number of samples

        Returns:
            Entropy value (higher = more semantic diversity = more uncertain)
        """
        if not clusters or n_samples == 0:
            return 0.0

        # Cluster probabilities = cluster_size / n_samples
        cluster_sizes = np.array([len(c) for c in clusters])
        cluster_probs = cluster_sizes / n_samples

        # Shannon entropy: H = -Σ p(c) * log(p(c))
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))

        return float(entropy)

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute Semantic Entropy uncertainty score.

        Note: The provided `response` is included as one of the samples.
        Additional samples are generated stochastically.

        Args:
            prompt: Input prompt
            response: Model's response (included in clustering)

        Returns:
            MethodResult with entropy-based uncertainty score
        """
        t0 = time.perf_counter()

        # Handle empty response
        if not response.strip():
            latency = (time.perf_counter() - t0) * 1000
            return MethodResult(
                score=float("nan"),
                confidence=0.0,
                latency_ms=latency,
                extra={"status": "DROP", "reason": "empty_response"},
            )

        try:
            # Generate additional samples
            generated_samples = self._generate_samples(prompt)

            # Include the original response as first sample
            all_samples = [response] + generated_samples

            # Cluster by semantic similarity
            clusters = self._cluster_by_similarity(all_samples)

            # Compute entropy over clusters
            entropy = self._compute_semantic_entropy(clusters, len(all_samples))

            # Normalize entropy to [0, 1] range
            # Max entropy = log(n_samples) when each sample is its own cluster
            max_entropy = np.log(len(all_samples))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            # Uncertainty = normalized entropy
            uncertainty = float(normalized_entropy)

        except Exception as e:
            # Fallback on error
            uncertainty = 0.5
            clusters = []
            entropy = 0.0

        latency = (time.perf_counter() - t0) * 1000

        # MethodResult.extra must be JSON-serializable scalars only
        num_clusters = len(clusters) if clusters else 0
        max_cluster_size = max(len(c) for c in clusters) if clusters else 0
        min_cluster_size = min(len(c) for c in clusters) if clusters else 0

        return MethodResult(
            score=uncertainty,
            confidence=1.0 - uncertainty,
            latency_ms=latency,
            extra={
                "num_clusters": num_clusters,
                "max_cluster_size": max_cluster_size,
                "min_cluster_size": min_cluster_size,
                "raw_entropy": float(entropy),
                "num_samples": len(all_samples) if "all_samples" in dir() else 0,
            },
        )

    def cleanup(self) -> None:
        """Release resources."""
        # Sentence transformer doesn't require explicit cleanup
        pass
