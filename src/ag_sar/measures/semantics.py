"""
Semantic Dispersion - measures consistency of top-k predictions.

Low dispersion: Top-k tokens are synonyms (US, USA, America) -> Grounded
High dispersion: Top-k tokens are unrelated (Paris, London, Rome) -> Hallucination
"""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_semantic_dispersion(
    logits: torch.Tensor,
    embed_matrix: torch.Tensor,
    k: int = 5,
    method: str = "nucleus_variance",
    top_p: float = 0.95,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute semantic dispersion of top predictions.

    Args:
        logits: (B, S, V) or (B, V) model output logits
        embed_matrix: (V, D) output embedding matrix
        k: Number of top tokens (for centroid_variance)
        method: "nucleus_variance" (SOTA) or "centroid_variance"
        top_p: Cumulative probability threshold for nucleus
        temperature: Softmax temperature

    Returns:
        dispersion: (B, S) or (B,) in [0, 1]
    """
    original_shape = logits.shape
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits_flat = logits.view(B * S, V)
    else:
        B, S = logits.shape[0], 1
        logits_flat = logits

    N = logits_flat.size(0)
    device = logits_flat.device
    dtype = logits_flat.dtype

    probs = F.softmax(logits_flat / temperature, dim=-1)

    if method == "nucleus_variance":
        # Adaptive Top-P clustering
        max_k = 50
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        cumsum_shifted = torch.cat([
            torch.zeros(N, 1, device=device, dtype=dtype),
            cumsum[:, :-1]
        ], dim=1)

        mask = cumsum_shifted < top_p
        mask[:, 0] = True  # Keep at least one token
        mask[:, max_k:] = False

        sorted_probs = sorted_probs[:, :max_k]
        sorted_indices = sorted_indices[:, :max_k]
        mask = mask[:, :max_k]

        embeds = F.embedding(sorted_indices, embed_matrix)
        masked_probs = sorted_probs * mask.float()
        masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-10)

        centroid = (embeds * masked_probs.unsqueeze(-1)).sum(dim=1, keepdim=True)
        embeds_n = F.normalize(embeds, p=2, dim=-1)
        centroid_n = F.normalize(centroid, p=2, dim=-1)
        distances = 1.0 - (embeds_n * centroid_n).sum(dim=-1)
        dispersion = (distances * masked_probs).sum(dim=-1)
    else:
        # Fixed Top-K centroid variance
        top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)

        embeds = F.embedding(top_ids, embed_matrix)
        centroid = (embeds * top_probs.unsqueeze(-1)).sum(dim=1, keepdim=True)
        embeds_n = F.normalize(embeds, p=2, dim=-1)
        centroid_n = F.normalize(centroid, p=2, dim=-1)
        distances = 1.0 - (embeds_n * centroid_n).sum(dim=-1)
        dispersion = (distances * top_probs).sum(dim=-1)

    dispersion = dispersion.clamp(0.0, 1.0)
    if len(original_shape) == 3:
        dispersion = dispersion.view(B, S)
    return dispersion
