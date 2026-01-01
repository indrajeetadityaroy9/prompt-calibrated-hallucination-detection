"""
Sink-aware eigenvector centrality computation.

CRITICAL: Uses pure PyTorch power iteration to avoid NetworkX CPU bottleneck.
NetworkX would require tensor-to-list conversions causing 100ms+ latency.

Key formula: R(t_i) = C_eigen(t_i) × ||v_i||_2

This naturally filters attention sinks which have high centrality
but low value norms (mechanistically passive tokens).
"""

from typing import Dict, Optional, Tuple
import torch

from .utils import apply_attention_mask


def power_iteration(
    adj_matrix: torch.Tensor,
    num_iterations: int = 3,
    tol: float = 1e-4,
    normalize_adjacency: bool = True
) -> torch.Tensor:
    """
    Compute eigenvector centrality via power iteration (GPU-native).

    Implements the dominant eigenvector computation for attention graphs.
    Pure PyTorch implementation avoids NetworkX CPU bottleneck (100ms+ latency).

    For attention matrices, 3 iterations is typically sufficient for convergence.

    Args:
        adj_matrix: (batch, n, n) adjacency matrix
        num_iterations: Maximum iterations for convergence (default: 3)
        tol: Convergence tolerance (stop when change < tol)
        normalize_adjacency: Whether to row-normalize the matrix first

    Returns:
        centrality: (batch, n) eigenvector centrality scores, normalized to sum to 1
    """
    batch_size, n, _ = adj_matrix.shape

    # Row-normalize to create stochastic matrix
    if normalize_adjacency:
        row_sum = adj_matrix.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        adj_matrix = adj_matrix / row_sum

    # Initialize with uniform vector
    v = torch.ones(batch_size, n, 1, device=adj_matrix.device,
                   dtype=adj_matrix.dtype) / n

    # Power iteration loop
    for _ in range(num_iterations):
        v_next = torch.bmm(adj_matrix, v)
        # L1-normalize (mathematically equivalent to L2 for stochastic matrix)
        v_next = v_next / (v_next.sum(dim=1, keepdim=True) + 1e-10)

        # Early stopping on convergence
        if torch.norm(v_next - v, dim=1).max() < tol:
            v = v_next
            break
        v = v_next

    # Return as (batch, n) with positive values, normalized to sum to 1
    centrality = v.squeeze(-1).abs()
    return centrality / centrality.sum(dim=-1, keepdim=True).clamp(min=1e-10)


def compute_sink_aware_centrality(
    attention_graph: torch.Tensor,
    value_norms: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_iterations: int = 100,
    tol: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute sink-aware relevance: R(t_i) = C_eigen(t_i) × ||v_i||_2

    The value norm weighting naturally filters attention sinks which have
    high centrality but low value norms (mechanistically passive tokens
    like <s>, punctuation, etc.).

    Args:
        attention_graph: (batch, seq, seq) global attention matrix
        value_norms: (batch, seq) L2 norm of value vectors per token
        attention_mask: (batch, seq) valid token mask (1=valid, 0=padding)
        num_iterations: Power iteration steps
        tol: Convergence tolerance

    Returns:
        relevance: (batch, seq) sink-aware relevance scores
        centrality: (batch, seq) raw eigenvector centrality (before sink correction)
    """
    # Apply mask to attention graph first (with normalization)
    if attention_mask is not None:
        attention_graph = apply_attention_mask(
            attention_graph, attention_mask, normalize=True
        )

    # Compute eigenvector centrality
    centrality = power_iteration(
        attention_graph,
        num_iterations=num_iterations,
        tol=tol,
        normalize_adjacency=False  # Already normalized above
    )

    # Sink-aware relevance: multiply centrality by value norms
    # This naturally down-weights attention sinks (high C, low ||v||)
    relevance = centrality * value_norms

    # Apply mask to final relevance
    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    return relevance, centrality


def aggregate_value_norms(
    value_norms_dict: Dict[int, torch.Tensor],
    semantic_layers: int = 4,
    aggregation: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate value norms across layers and heads.

    Args:
        value_norms_dict: Dict[layer_idx, (batch, heads, seq)]
        semantic_layers: Use last N layers (0 = use all)
        aggregation: 'mean' or 'max'

    Returns:
        value_norms: (batch, seq) aggregated norms per token
    """
    if not value_norms_dict:
        raise ValueError("value_norms_dict is empty")

    layer_indices = sorted(value_norms_dict.keys())

    # Take last N layers if specified
    if semantic_layers > 0 and semantic_layers < len(layer_indices):
        layer_indices = layer_indices[-semantic_layers:]

    norms_list = []
    for layer_idx in layer_indices:
        layer_norms = value_norms_dict[layer_idx]  # (batch, heads, seq)

        # Aggregate across heads first
        if aggregation == 'mean':
            head_agg = layer_norms.mean(dim=1)  # (batch, seq)
        else:  # max
            head_agg = layer_norms.max(dim=1).values

        norms_list.append(head_agg)

    # Stack and aggregate across layers
    stacked = torch.stack(norms_list, dim=0)  # (layers, batch, seq)

    if aggregation == 'mean':
        return stacked.mean(dim=0)
    else:  # max
        return stacked.max(dim=0).values
