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


def compute_top1_projection(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-1 Projection Semantic Dispersion (Legacy/QA).

    Measures distance from the top-1 prediction to alternatives.
    Best for factual QA where "1990" vs "1991" should be distinguished.

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


def compute_centroid_variance(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    JEPA-Style Latent Variance (Summarization).

    Measures geometric spread of top-k predictions around their weighted centroid.
    Robust to synonyms (e.g., "huge" vs "giant" cluster tightly → low variance).

    WARNING: Antonyms also cluster tightly in embedding space. This method
    should be combined with Authority Flow for factual verification.
    The "Antonym Safety Valve" relies on parametric_weight < threshold.

    Args:
        logits: (B, S, V) or (B, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        k: Top-k tokens to consider (default 10 for broader semantic capture)
        temperature: Softmax temperature

    Returns:
        variance: (B, S) or (B,) in [0, 1] where 0=tight cluster, 1=dispersed
    """
    original_shape = logits.shape

    # Handle 3D input
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits_flat = logits.view(B * S, V)
    else:
        B = logits.shape[0]
        S = 1
        logits_flat = logits

    # 1. Get probability distribution
    probs = F.softmax(logits_flat / temperature, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    # Normalize to sum to 1 within top-k
    top_probs_norm = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)

    # 2. Retrieve embeddings [N, K, D]
    top_embeds = F.embedding(top_ids, embed_matrix)

    # 3. Compute weighted centroid (the "mean meaning")
    # Shape: [N, 1, D]
    centroid = (top_embeds * top_probs_norm.unsqueeze(-1)).sum(dim=1, keepdim=True)

    # 4. Compute angular distance from centroid
    # CRITICAL: Normalize centroid to prevent norm-collapse issues
    top_embeds_n = F.normalize(top_embeds, p=2, dim=-1)
    centroid_n = F.normalize(centroid, p=2, dim=-1)

    # Cosine similarity: [N, K]
    cosine_sims = (top_embeds_n * centroid_n).sum(dim=-1)

    # Distance = 1 - Similarity
    distances = 1.0 - cosine_sims

    # 5. Weighted variance (expected distance to centroid)
    variance = (distances * top_probs_norm).sum(dim=-1)
    variance = variance.clamp(0.0, 1.0)

    # Reshape back
    if len(original_shape) == 3:
        variance = variance.view(B, S)

    return variance


def compute_nucleus_centroid_variance(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    top_p: float = 0.95,
    temperature: float = 1.0,
    min_k: int = 1,
    max_k: int = 50,
) -> torch.Tensor:
    """
    SOTA Nucleus (Top-P) Centroid Variance.

    Dynamic Semantic Receptive Field: Instead of fixed Top-K, uses the smallest
    set of tokens whose cumulative probability exceeds top_p.

    Key Insight:
    - Confident token ("Paris"): Nucleus might include only 1 token → Variance = 0
    - Confused token ("1984/1985"): Nucleus includes many → High variance if diverse

    This adapts the cluster size to the model's confidence, avoiding:
    - Signal dilution on confident tokens (fixed k=10 captures irrelevant tokens)
    - Signal truncation on uncertain tokens (fixed k=10 misses important alternatives)

    Args:
        logits: (B, S, V) or (B, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        top_p: Cumulative probability threshold (default 0.95)
        temperature: Softmax temperature
        min_k: Minimum tokens in nucleus (default 1)
        max_k: Maximum tokens to consider (default 50, for efficiency)

    Returns:
        variance: (B, S) or (B,) in [0, 1] where 0=tight cluster, 1=dispersed
    """
    original_shape = logits.shape

    # Handle 3D input
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits_flat = logits.view(B * S, V)
    else:
        B = logits.shape[0]
        S = 1
        logits_flat = logits

    N = logits_flat.size(0)
    device = logits_flat.device
    dtype = logits_flat.dtype

    # 1. Get probability distribution
    probs = F.softmax(logits_flat / temperature, dim=-1)

    # 2. Sort probabilities descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # 3. Compute cumulative probabilities
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # 4. Create nucleus mask: tokens within top_p (always keep at least min_k)
    # Shift cumsum to include the token that crosses threshold
    cumsum_shifted = torch.cat([
        torch.zeros(N, 1, device=device, dtype=dtype),
        cumsum_probs[:, :-1]
    ], dim=1)
    nucleus_mask = cumsum_shifted < top_p

    # Enforce min_k and max_k
    nucleus_mask[:, :min_k] = True
    nucleus_mask[:, max_k:] = False

    # 5. Limit to max_k tokens for efficiency
    sorted_probs = sorted_probs[:, :max_k]
    sorted_indices = sorted_indices[:, :max_k]
    nucleus_mask = nucleus_mask[:, :max_k]

    # 6. Get embeddings for nucleus tokens
    top_embeds = F.embedding(sorted_indices, embed_matrix)  # [N, max_k, D]

    # 7. Apply mask to probabilities (zero out non-nucleus tokens)
    masked_probs = sorted_probs * nucleus_mask.float()

    # Re-normalize within nucleus
    masked_probs_sum = masked_probs.sum(dim=-1, keepdim=True) + 1e-10
    masked_probs_norm = masked_probs / masked_probs_sum

    # 8. Compute weighted centroid
    centroid = (top_embeds * masked_probs_norm.unsqueeze(-1)).sum(dim=1, keepdim=True)

    # 9. Compute angular distance from centroid
    top_embeds_n = F.normalize(top_embeds, p=2, dim=-1)
    centroid_n = F.normalize(centroid, p=2, dim=-1)

    # Cosine similarity: [N, max_k]
    cosine_sims = (top_embeds_n * centroid_n).sum(dim=-1)

    # Distance = 1 - Similarity
    distances = 1.0 - cosine_sims

    # 10. Weighted variance (only over nucleus tokens)
    variance = (distances * masked_probs_norm).sum(dim=-1)
    variance = variance.clamp(0.0, 1.0)

    # Reshape back
    if len(original_shape) == 3:
        variance = variance.view(B, S)

    return variance


def compute_semantic_dispersion(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    temperature: float = 1.0,
    method: str = "top1_projection",
    top_p: float = 0.95,
) -> torch.Tensor:
    """
    Dispatcher for semantic dispersion calculation.

    Selects between algorithms based on task type:
    - "top1_projection": Legacy/QA - measures distance from top-1 prediction
    - "centroid_variance": JEPA/Summ - measures spread around weighted centroid (Top-K)
    - "nucleus_variance": SOTA - adaptive Top-P clustering (dynamic k)

    Args:
        logits: (B, S, V) or (B, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        k: Number of top tokens to consider (for top1_projection, centroid_variance)
        temperature: Softmax temperature
        method: "top1_projection", "centroid_variance", or "nucleus_variance"
        top_p: Cumulative probability threshold (for nucleus_variance)

    Returns:
        dispersion: (B, S) or (B,) semantic dispersion in [0, 1]
    """
    if method == "nucleus_variance":
        return compute_nucleus_centroid_variance(
            logits, embed_matrix, top_p=top_p, temperature=temperature
        )
    elif method == "centroid_variance":
        return compute_centroid_variance(logits, embed_matrix, k=k, temperature=temperature)
    else:
        return compute_top1_projection(logits, embed_matrix, k=k, temperature=temperature)


def compute_semantic_trust(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    sensitivity: float = 5.0,
    method: str = "top1_projection",
    top_p: float = 0.95,
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
        method: "top1_projection", "centroid_variance", or "nucleus_variance"
        top_p: Cumulative probability threshold for nucleus_variance (default 0.95)

    Returns:
        trust: (B, S) or (B,) semantic trust in [0, 1]
               1.0 = High consistency (trustworthy)
               0.0 = High dispersion (untrustworthy)
    """
    dispersion = compute_semantic_dispersion(logits, embed_matrix, k, method=method, top_p=top_p)

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
