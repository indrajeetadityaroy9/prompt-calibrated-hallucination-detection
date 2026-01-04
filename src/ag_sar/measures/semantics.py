"""
Semantic Dispersion Measures for Hallucination Detection (AG-SAR v8.0).

Implements Logit Semantic Dispersion (LSD) - measuring the semantic
consistency of the model's top-k predictions.

Key Insight: "Confidently Wrong" vs "Semantically Confused"
- Low dispersion: Top-k tokens are synonyms (US, USA, America) → Grounded
- High dispersion: Top-k tokens are unrelated (Paris, London, Rome) → Hallucination

This replaces raw confidence (max probability) with semantic consistency,
which is more robust to "confident lies" in natural hallucinations.

References:
- "Semantic Uncertainty" (Kuhn et al., 2023)
- "Detecting Hallucinations with Semantic Entropy" (Farquhar et al., 2024)
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_semantic_dispersion(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Semantic Dispersion of Top-K predictions.

    Measures how semantically diverse the model's top predictions are.
    High dispersion = model is "confused" between unrelated options = risky.
    Low dispersion = model's alternatives are synonyms = grounded.

    Algorithm:
    1. Identify Top-K candidate tokens by probability
    2. Retrieve their embedding vectors from the output matrix
    3. Compute weighted average cosine distance from Top-1

    Args:
        logits: (B, S, V) or (B, V) logits from model output
        embed_matrix: (V, D) output embedding matrix (unembedding)
        k: Number of top tokens to consider (default 5)
        temperature: Softmax temperature (default 1.0)

    Returns:
        dispersion: (B, S) or (B,) semantic dispersion in [0, 1]
                    0.0 = All candidates mean the same thing (Safe)
                    1.0 = Candidates are totally different (Risky)

    Example:
        >>> # "The capital of France is ___"
        >>> # Top-5: ["Paris", "the", "located", "a", "called"]
        >>> # "Paris" vs others = high dispersion (but Paris is correct!)
        >>>
        >>> # "The US is also known as ___"
        >>> # Top-5: ["USA", "America", "the", "United", "States"]
        >>> # All related to USA = low dispersion = grounded
    """
    original_shape = logits.shape

    # Handle 3D input (B, S, V) by flattening
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits_flat = logits.view(B * S, V)
    else:
        B = logits.shape[0]
        S = 1
        logits_flat = logits

    # 1. Get Top-K Probabilities
    probs = F.softmax(logits_flat / temperature, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k, dim=-1)  # (N, K)

    # Re-normalize probs to sum to 1 within the cluster
    top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # 2. Get Semantic Vectors from embedding matrix
    # embed_matrix: (V, D), top_indices: (N, K)
    # top_vectors: (N, K, D)
    top_vectors = F.embedding(top_indices, embed_matrix)

    # 3. Calculate Cosine Similarity to Top-1
    # Top-1 is at index 0. Shape: (N, 1, D)
    top1_vector = top_vectors[:, 0:1, :]

    # Normalize for Cosine Similarity
    vec_norm = F.normalize(top_vectors, p=2, dim=-1)
    top1_norm = F.normalize(top1_vector, p=2, dim=-1)

    # Cosine Similarity: (N, K)
    similarities = torch.sum(vec_norm * top1_norm, dim=-1)

    # 4. Compute Weighted Dispersion
    # Distance = 1 - Similarity (Range [0, 2], typically [0, 1])
    distances = 1.0 - similarities

    # Weighted sum by probability (excluding top-1 which has distance 0)
    # This gives average distance from top-1 to other candidates
    dispersion = torch.sum(top_probs * distances, dim=-1)  # (N,)

    # Clamp to [0, 1] for numerical stability
    dispersion = dispersion.clamp(0.0, 1.0)

    # Reshape back to original
    if len(original_shape) == 3:
        dispersion = dispersion.view(B, S)

    return dispersion


def compute_semantic_trust(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    sensitivity: float = 5.0,
) -> torch.Tensor:
    """
    Compute Semantic Trust score (inverse of dispersion).

    Convenience function that converts dispersion to trust:
    Trust = 1 - (Dispersion × Sensitivity)

    Args:
        logits: (B, S, V) or (B, V) logits from model output
        embed_matrix: (V, D) output embedding matrix
        k: Number of top tokens to consider
        sensitivity: Scale factor (higher = more aggressive penalty)

    Returns:
        trust: (B, S) or (B,) semantic trust in [0, 1]
               1.0 = High consistency (trustworthy)
               0.0 = High dispersion (untrustworthy)
    """
    dispersion = compute_semantic_dispersion(logits, embed_matrix, k)

    # Invert: High dispersion = Low trust
    # Sensitivity scales the impact (dispersion is usually small, e.g., 0.05)
    trust = 1.0 - (dispersion * sensitivity)

    # Clamp to [0, 1]
    trust = trust.clamp(0.0, 1.0)

    return trust


def compute_entropy_weighted_dispersion(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Compute dispersion weighted by prediction entropy.

    High entropy + High dispersion = Maximum uncertainty
    Low entropy + Low dispersion = Maximum confidence

    This combines two orthogonal signals:
    - Entropy: How uncertain is the model about the top choice?
    - Dispersion: How semantically different are the alternatives?

    Args:
        logits: (B, S, V) logits
        embed_matrix: (V, D) embedding matrix
        k: Number of top tokens

    Returns:
        weighted_dispersion: (B, S) entropy-weighted dispersion
    """
    # Compute base dispersion
    dispersion = compute_semantic_dispersion(logits, embed_matrix, k)

    # Compute entropy
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (B, S)

    # Normalize entropy by max possible (log vocab size)
    vocab_size = logits.size(-1)
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=logits.dtype, device=logits.device))
    normalized_entropy = entropy / max_entropy  # [0, 1]

    # Weight dispersion by entropy
    # High entropy amplifies dispersion concern
    weighted = dispersion * (1.0 + normalized_entropy)

    return weighted.clamp(0.0, 1.0)
