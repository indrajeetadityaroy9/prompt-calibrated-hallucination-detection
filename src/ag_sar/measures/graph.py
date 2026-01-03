"""
Graph-Based Centrality Measures.

Matrix-free O(N) eigenvector centrality computation using power iteration.
The key innovation: no O(N^2) attention matrix materialization.

Core formula: R(t_i) = C_eigen(t_i) * ||v_i||_2
This naturally filters attention sinks (high centrality, low value norms).
"""

from typing import Dict, Optional, Tuple
import torch

from ..ops import centrality_kernel


def compute_centrality(
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    num_iterations: int = 3,
    residual_weight: float = 0.5,
    tol: float = 1e-4,
    attention_mask: Optional[torch.Tensor] = None,
    value_norms: Optional[torch.Tensor] = None,
    return_per_head: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvector centrality via matrix-free power iteration.

    Algorithm per iteration:
        1. v_in = v * signal_filter (inject value norms)
        2. per_head_out = triton_kernel(Q, K, v_in)
        3. attn_contrib = mean(per_head_out, dim=heads)
        4. v_next = (1 - residual_weight) * attn_contrib + residual_weight * v
        5. v_next = normalize(v_next)

    The residual connection prevents centrality collapse on early tokens.

    Args:
        Q_stack: (B, L*H, S, D) stacked query vectors
        K_stack: (B, L*H, S, D) stacked key vectors
        num_iterations: Power iteration steps (3 typically sufficient)
        residual_weight: Weight for self-attention residual (0.5 = equal mix)
        tol: Convergence tolerance
        attention_mask: (B, S) padding mask
        value_norms: (B, S) value vector norms for signal injection
        return_per_head: Return per-head contributions

    Returns:
        centrality: (B, S) normalized eigenvector centrality
        per_head_contrib: (B, L*H, S) if return_per_head, else None
    """
    B, total_heads, S, D = Q_stack.shape
    device = Q_stack.device
    dtype = Q_stack.dtype

    # Ensure attention_mask is on correct device
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=device)

    # Initialize uniform centrality
    v = torch.ones(B, S, device=device, dtype=dtype) / S

    if attention_mask is not None:
        v = v * attention_mask.float()
        v = v / v.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Pre-compute signal filter from value norms (ensure device and dtype consistency)
    signal_filter = None
    if value_norms is not None:
        value_norms = value_norms.to(device=device, dtype=dtype)
        signal_filter = value_norms / (value_norms.max(dim=-1, keepdim=True)[0] + 1e-6)
        signal_filter = signal_filter.to(device=device, dtype=dtype)

    final_per_head_out = None

    for _ in range(num_iterations):
        # Inject signal strength
        if signal_filter is not None:
            v_in = v * signal_filter
            v_in = v_in / v_in.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            v_in = v

        # Triton/PyTorch kernel: attention-weighted sum per head
        per_head_out = centrality_kernel(Q_stack, K_stack, v_in)

        if return_per_head:
            final_per_head_out = per_head_out.clone()

        # Aggregate across heads
        attn_contrib = per_head_out.mean(dim=1)

        # Residual connection
        v_next = (1 - residual_weight) * attn_contrib + residual_weight * v

        if attention_mask is not None:
            v_next = v_next * attention_mask.float()

        # Normalize
        v_next = v_next / v_next.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Early stopping
        if (v_next - v).abs().max() < tol:
            v = v_next
            break

        v = v_next

    return v, final_per_head_out


def compute_sink_aware_centrality(
    value_norms: torch.Tensor,
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_iterations: int = 3,
    tol: float = 1e-4,
    residual_weight: float = 0.5,
    return_raw: bool = False,
    sink_token_count: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute sink-aware relevance: R(t_i) = C_eigen(t_i) * ||v_i||_2

    The value norm weighting naturally filters attention sinks which have
    high centrality but low value norms.

    Args:
        value_norms: (B, S) L2 norm of value vectors per token
        Q_stack: (B, L*H, S, D) stacked queries
        K_stack: (B, L*H, S, D) stacked keys
        attention_mask: (B, S) valid token mask
        num_iterations: Power iteration steps
        tol: Convergence tolerance
        residual_weight: Residual weight
        return_raw: Return per-head contributions
        sink_token_count: First N tokens to zero out

    Returns:
        relevance: (B, S) sink-aware relevance scores
        centrality: (B, S) raw eigenvector centrality
        per_head_contrib: (B, L*H, S) if return_raw, else None
    """
    if Q_stack is None or K_stack is None:
        raise ValueError("Q_stack and K_stack required for Flash Centrality")

    centrality, per_head_contrib = compute_centrality(
        Q_stack=Q_stack,
        K_stack=K_stack,
        num_iterations=num_iterations,
        residual_weight=residual_weight,
        tol=tol,
        attention_mask=attention_mask,
        value_norms=value_norms,
        return_per_head=return_raw,
    )

    # Sink-aware relevance
    relevance = centrality * value_norms

    # Zero out sink tokens
    if sink_token_count > 0:
        relevance[:, :sink_token_count] = 0.0

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
        value_norms_dict: Dict[layer_idx, (B, heads, S)]
        semantic_layers: Use last N layers (0 = use all)
        aggregation: 'mean' or 'max'

    Returns:
        value_norms: (B, S) aggregated norms
    """
    if not value_norms_dict:
        raise ValueError("value_norms_dict is empty")

    layer_indices = sorted(value_norms_dict.keys())

    if semantic_layers > 0 and semantic_layers < len(layer_indices):
        layer_indices = layer_indices[-semantic_layers:]

    norms_list = []
    for layer_idx in layer_indices:
        layer_norms = value_norms_dict[layer_idx]

        if aggregation == 'mean':
            head_agg = layer_norms.mean(dim=1)
        else:
            head_agg = layer_norms.max(dim=1).values

        norms_list.append(head_agg)

    stacked = torch.stack(norms_list, dim=0)

    if aggregation == 'mean':
        return stacked.mean(dim=0)
    else:
        return stacked.max(dim=0).values
