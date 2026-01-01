"""
Sink-aware eigenvector centrality computation.

Supports two computation modes:
1. Matrix-based (legacy): Uses explicit O(N^2) attention graph
2. Matrix-free (Triton): Uses O(N) memory via Flash-style kernel

Key formula: R(t_i) = C_eigen(t_i) × ||v_i||_2

This naturally filters attention sinks which have high centrality
but low value norms (mechanistically passive tokens).
"""

from typing import Dict, Optional, Tuple
import torch

from .utils import apply_attention_mask
from .kernels.centrality_flash import centrality_flash_fwd


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


def matrix_free_power_iteration(
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    num_iterations: int = 3,
    residual_weight: float = 0.5,
    tol: float = 1e-4,
    attention_mask: Optional[torch.Tensor] = None,
    return_raw: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvector centrality via matrix-free power iteration.

    Uses Triton kernel to avoid O(N^2) attention matrix materialization.
    This is the core of the "Flash Centrality" approach.

    Algorithm per iteration:
        1. per_head_out = triton_kernel(Q, K, v)  # Attention-weighted sum
        2. attn_contrib = mean(per_head_out, dim=heads)
        3. v_next = (1 - residual_weight) * attn_contrib + residual_weight * v
        4. v_next = normalize(v_next)

    The residual connection (step 3) prevents centrality collapse on early
    tokens in causal attention, equivalent to adding identity matrix
    in the matrix-based approach.

    Args:
        Q_stack: Stacked query vectors from semantic layers (B, L*H, S, D)
        K_stack: Stacked key vectors from semantic layers (B, L*H, S, D)
        num_iterations: Power iteration steps (3 typically sufficient)
        residual_weight: Weight for self-attention residual (0.5 = equal mix)
        tol: Convergence tolerance for early stopping
        attention_mask: (B, S) padding mask (1=valid, 0=padding)
        return_raw: If True, return per-head contributions for head specialization analysis

    Returns:
        centrality: (B, S) normalized eigenvector centrality
        per_head_contrib: (B, L*H, S) per-head contributions if return_raw=True, else None
    """
    B, total_heads, S, D = Q_stack.shape
    device = Q_stack.device
    dtype = Q_stack.dtype

    # Initialize uniform centrality
    v = torch.ones(B, S, device=device, dtype=dtype) / S

    # Apply mask to initial vector
    if attention_mask is not None:
        v = v * attention_mask.float()
        v = v / v.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Track per-head contributions from final iteration
    final_per_head_out = None

    for _ in range(num_iterations):
        # Triton kernel: attention-weighted sum per head
        # per_head_out[b, h, i] = sum_j softmax(Q@K.T)[i,j] * v[j]
        per_head_out = centrality_flash_fwd(Q_stack, K_stack, v)  # (B, L*H, S)

        # Save for head specialization analysis
        if return_raw:
            final_per_head_out = per_head_out.clone()

        # Aggregate across all heads (mean pooling)
        attn_contrib = per_head_out.mean(dim=1)  # (B, S)

        # Apply residual connection: v_next = (1-w) * Av + w * v
        # This prevents centrality collapse on early tokens
        v_next = (1 - residual_weight) * attn_contrib + residual_weight * v

        # Apply mask
        if attention_mask is not None:
            v_next = v_next * attention_mask.float()

        # Normalize to sum=1
        v_next = v_next / v_next.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Early stopping check
        if (v_next - v).abs().max() < tol:
            v = v_next
            break

        v = v_next

    return v, final_per_head_out


def compute_sink_aware_centrality(
    value_norms: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_iterations: int = 3,
    tol: float = 1e-4,
    # Matrix-free path (preferred)
    Q_stack: Optional[torch.Tensor] = None,
    K_stack: Optional[torch.Tensor] = None,
    residual_weight: float = 0.5,
    # Legacy matrix-based path
    attention_graph: Optional[torch.Tensor] = None,
    # Head specialization analysis
    return_raw: bool = False,
    # Sink token masking (StreamingLLM-style)
    sink_token_count: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute sink-aware relevance: R(t_i) = C_eigen(t_i) × ||v_i||_2

    The value norm weighting naturally filters attention sinks which have
    high centrality but low value norms (mechanistically passive tokens
    like <s>, punctuation, etc.).

    Supports two computation modes:
    1. Matrix-free (Triton): Pass Q_stack, K_stack for O(N) memory
    2. Matrix-based (legacy): Pass attention_graph for O(N^2) memory

    Args:
        value_norms: (batch, seq) L2 norm of value vectors per token
        attention_mask: (batch, seq) valid token mask (1=valid, 0=padding)
        num_iterations: Power iteration steps
        tol: Convergence tolerance
        Q_stack: (B, L*H, S, D) stacked queries for matrix-free path
        K_stack: (B, L*H, S, D) stacked keys for matrix-free path
        residual_weight: Weight for self-attention in matrix-free path
        attention_graph: (batch, seq, seq) for legacy matrix-based path
        return_raw: If True, return per-head contributions for head specialization
        sink_token_count: Number of initial tokens to zero out (BOS/sink tokens)
            These have high centrality but are structural, not semantic.

    Returns:
        relevance: (batch, seq) sink-aware relevance scores
        centrality: (batch, seq) raw eigenvector centrality (before sink correction)
        per_head_contrib: (B, L*H, S) per-head contributions if return_raw=True, else None
    """
    # Determine which path to use
    use_matrix_free = Q_stack is not None and K_stack is not None
    per_head_contrib = None

    if use_matrix_free:
        # Matrix-free path using Triton kernel
        centrality, per_head_contrib = matrix_free_power_iteration(
            Q_stack=Q_stack,
            K_stack=K_stack,
            num_iterations=num_iterations,
            residual_weight=residual_weight,
            tol=tol,
            attention_mask=attention_mask,
            return_raw=return_raw,
        )
    else:
        # Legacy matrix-based path
        if attention_graph is None:
            raise ValueError(
                "Must provide either (Q_stack, K_stack) for matrix-free path "
                "or attention_graph for legacy matrix-based path"
            )

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

    # Zero out sink tokens (BOS, structural attention sinks)
    # These have high centrality but are not semantically meaningful
    if sink_token_count > 0:
        relevance[:, :sink_token_count] = 0.0

    # Apply mask to final relevance
    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    return relevance, centrality, per_head_contrib


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
