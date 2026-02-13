"""
Context Grounding Score - Mechanistic Hallucination Detection.

Core Insight:
    If the model is grounded in context, its output representations should be
    "explainable" by the context representations. Hallucinations produce
    representations that diverge from the context subspace.

Mathematical Foundation:
    For each output token hidden state h_t:
    1. Project h_t onto the context subspace (spanned by context hidden states)
    2. Measure the residual (component orthogonal to context)
    3. High residual = output is not well-explained by context = hallucination

    Grounding Score = ||proj_context(h_t)|| / ||h_t||
    Hallucination Risk = 1 - Grounding Score

Key Properties:
    1. Zero-shot: No training required
    2. Single-pass: Works with forced decoding
    3. No external knowledge: Derived purely from model internals
    4. Task-agnostic: Based on fundamental principle of grounding

This is inspired by:
    - Transformer-circuits research on information flow
    - Linear representation hypothesis
    - Causal tracing methods
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


EPS = 1e-10


@dataclass
class GroundingResult:
    """Result of context grounding analysis."""
    # Per-token grounding scores [0, 1] - higher = more grounded
    grounding_scores: np.ndarray

    # Per-token hallucination risk [0, 1] - higher = riskier
    hallucination_risk: np.ndarray

    # Response-level aggregated risk
    response_risk: float

    # Diagnostics
    context_rank: int  # Effective rank of context subspace
    avg_projection_ratio: float  # Average ||proj|| / ||h||
    avg_residual_ratio: float  # Average ||residual|| / ||h||


class ContextGroundingSignal:
    """
    Compute context grounding scores from hidden states.

    The key idea: Output tokens that are grounded in context should have
    hidden representations that can be explained by the context subspace.
    Hallucinations produce "out-of-distribution" representations.
    """

    def __init__(
        self,
        use_layers: str = "middle",  # "middle", "last", "all"
        projection_method: str = "svd",  # "svd", "direct"
        min_context_tokens: int = 5,
    ):
        """
        Initialize context grounding signal.

        Args:
            use_layers: Which layers to analyze
            projection_method: How to compute projection onto context subspace
            min_context_tokens: Minimum context tokens for valid analysis
        """
        self.use_layers = use_layers
        self.projection_method = projection_method
        self.min_context_tokens = min_context_tokens

    def _get_layer_indices(self, n_layers: int) -> List[int]:
        """Get layer indices to use based on configuration."""
        if self.use_layers == "middle":
            # Middle third of layers (most informative per RAGLens)
            start = n_layers // 3
            end = 2 * n_layers // 3
            return list(range(start, end))
        elif self.use_layers == "last":
            return [n_layers - 1]
        else:  # "all"
            return list(range(n_layers))

    def _compute_context_basis(
        self,
        context_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Compute orthonormal basis for context subspace using SVD.

        Args:
            context_hidden: [n_context, hidden_dim] context hidden states

        Returns:
            Tuple of (basis vectors [k, hidden_dim], center [1, hidden_dim], effective rank k)
        """
        # Center the context representations
        context_mean = context_hidden.mean(dim=0, keepdim=True)
        context_centered = context_hidden - context_mean

        # SVD to get principal components
        try:
            U, S, Vh = torch.linalg.svd(context_centered, full_matrices=False)
        except RuntimeError:
            # Fallback for numerical issues
            return context_hidden.T, context_hidden.mean(dim=0, keepdim=True), context_hidden.shape[0]

        # Determine effective rank (singular values > threshold)
        total_var = (S ** 2).sum()
        cumvar = (S ** 2).cumsum(dim=0) / (total_var + EPS)

        # Keep components explaining 95% of variance
        k = int((cumvar < 0.95).sum()) + 1
        k = min(k, len(S), context_hidden.shape[0])

        # Return top-k right singular vectors (basis for context subspace)
        basis = Vh[:k]  # [k, hidden_dim]

        return basis, context_mean, k

    def _project_onto_subspace(
        self,
        vectors: torch.Tensor,
        basis: torch.Tensor,
        center: torch.Tensor,
        short_context: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project vectors onto subspace defined by basis.

        Args:
            vectors: [n, hidden_dim] vectors to project
            basis: [k, hidden_dim] orthonormal basis vectors (or normalized context vectors for short contexts)
            center: [1, hidden_dim] center of subspace
            short_context: If True, use max cosine similarity instead of orthogonal projection

        Returns:
            Tuple of (projections [n, hidden_dim], residuals [n, hidden_dim])
        """
        if short_context:
            # For short contexts, basis contains normalized context vectors
            # Compute max cosine similarity to any context token
            vectors_normalized = vectors / (torch.norm(vectors, dim=-1, keepdim=True) + EPS)
            # similarity: [n_vectors, n_context]
            similarity = vectors_normalized @ basis.T
            # Max similarity for each output token
            max_sim, _ = similarity.max(dim=-1, keepdim=True)  # [n, 1]
            # Create "projection" that represents grounded component
            # Scale by max similarity - if similar to context, projection is large
            projection = max_sim * vectors_normalized
            residual = vectors_normalized - projection
            return projection, residual

        # Center vectors
        vectors_centered = vectors - center

        # Project: proj = V @ V^T @ x (where V is basis)
        # coefficients = vectors @ basis^T  -> [n, k]
        # projection = coefficients @ basis -> [n, hidden_dim]
        coefficients = vectors_centered @ basis.T
        projection = coefficients @ basis

        # Residual is the orthogonal component
        residual = vectors_centered - projection

        return projection, residual

    def compute_grounding_scores(
        self,
        hidden_states: List[torch.Tensor],
        prompt_len: int,
    ) -> GroundingResult:
        """
        Compute per-token grounding scores.

        Args:
            hidden_states: List of hidden states per layer [batch, seq, dim]
            prompt_len: Number of prompt/context tokens

        Returns:
            GroundingResult with grounding scores and diagnostics
        """
        n_layers = len(hidden_states)
        layer_indices = self._get_layer_indices(n_layers)

        seq_len = hidden_states[0].shape[1]
        response_len = seq_len - prompt_len

        if response_len < 1 or prompt_len < self.min_context_tokens:
            return GroundingResult(
                grounding_scores=np.array([0.5]),
                hallucination_risk=np.array([0.5]),
                response_risk=0.5,
                context_rank=0,
                avg_projection_ratio=0.5,
                avg_residual_ratio=0.5,
            )

        all_grounding_scores = []
        all_projection_ratios = []
        all_residual_ratios = []
        context_ranks = []

        for layer_idx in layer_indices:
            hidden = hidden_states[layer_idx][0].float()  # [seq, dim]

            context_hidden = hidden[:prompt_len]
            response_hidden = hidden[prompt_len:]

            # Compute context subspace basis via SVD
            basis, center, rank = self._compute_context_basis(context_hidden)
            context_ranks.append(rank)

            # For low-rank contexts (like QA), SVD projection loses information
            # Use direct max cosine similarity instead
            low_rank = (rank < 10)

            if low_rank:
                # Direct similarity approach: check if output is similar to ANY context token
                context_normalized = context_hidden / (torch.norm(context_hidden, dim=-1, keepdim=True) + EPS)
                response_normalized = response_hidden / (torch.norm(response_hidden, dim=-1, keepdim=True) + EPS)
                # similarity: [n_response, n_context]
                similarity = response_normalized @ context_normalized.T
                # Max similarity for each output token = grounding score
                max_sim, _ = similarity.max(dim=-1)  # [n_response]
                grounding = max_sim.clamp(0, 1)
                # For diagnostics
                projection_norms = max_sim
                residual_norms = 1 - max_sim
                response_norms = torch.ones_like(max_sim)
            else:
                # SVD projection approach for high-rank contexts
                projection, residual = self._project_onto_subspace(
                    response_hidden, basis, center, short_context=False
                )
                response_norms = torch.norm(response_hidden - center, dim=-1)
                projection_norms = torch.norm(projection, dim=-1)
                residual_norms = torch.norm(residual, dim=-1)
                grounding = projection_norms / (response_norms + EPS)
                grounding = torch.clamp(grounding, 0, 1)

            all_grounding_scores.append(grounding.cpu().numpy())
            all_projection_ratios.append(
                (projection_norms / (response_norms + EPS)).mean().item()
            )
            all_residual_ratios.append(
                (residual_norms / (response_norms + EPS)).mean().item()
            )

        # Average across layers
        grounding_scores = np.mean(all_grounding_scores, axis=0)

        # Hallucination risk = 1 - grounding
        hallucination_risk = 1 - grounding_scores

        # Response-level aggregation: use 90th percentile of risk
        response_risk = float(np.percentile(hallucination_risk, 90))

        return GroundingResult(
            grounding_scores=grounding_scores,
            hallucination_risk=hallucination_risk,
            response_risk=response_risk,
            context_rank=int(np.mean(context_ranks)),
            avg_projection_ratio=float(np.mean(all_projection_ratios)),
            avg_residual_ratio=float(np.mean(all_residual_ratios)),
        )
