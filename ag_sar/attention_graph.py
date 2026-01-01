"""
Attention graph construction with Attention Rollout.

CRITICAL: Must add identity matrix (0.5*A + 0.5*I) BEFORE rollout
to prevent lower-triangular decay in causal attention.
"""

from typing import Dict, List, Optional
import torch

from .utils import apply_attention_mask


def add_residual_connection(
    attention: torch.Tensor,
    residual_weight: float = 0.5
) -> torch.Tensor:
    """
    Add residual connection to attention matrix.

    CRITICAL: Must add identity BEFORE rollout to prevent
    lower-triangular decay in causal attention. Without this,
    early tokens lose information through recursive multiplication.

    Formula: A = (1 - residual_weight) * W_att + residual_weight * I

    Args:
        attention: (batch, seq, seq) attention matrix
        residual_weight: Weight for identity matrix (default: 0.5)

    Returns:
        Attention with residual: (batch, seq, seq)
    """
    batch_size, seq_len, _ = attention.shape
    device = attention.device
    dtype = attention.dtype

    # Create identity matrix and expand for batch
    identity = torch.eye(seq_len, device=device, dtype=dtype)
    identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

    # Combine: A = (1 - w) * W_att + w * I
    result = (1 - residual_weight) * attention + residual_weight * identity

    return result


def compute_attention_rollout(
    attention_layers: List[torch.Tensor],
    residual_weight: float = 0.5,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute Attention Rollout across layers.

    Recursively multiplies attention matrices from first to last layer:
    A_rollout = A_L @ A_{L-1} @ ... @ A_1

    Each layer's attention is first augmented with residual connection.

    Args:
        attention_layers: List of attention matrices per layer.
            Each: (batch, seq, seq)
        residual_weight: Weight for residual connection
        normalize: Whether to normalize rows after rollout

    Returns:
        rollout: (batch, seq, seq) global attention flow matrix
    """
    if not attention_layers:
        raise ValueError("Empty attention_layers list")

    # Start with first layer (add residual)
    rollout = add_residual_connection(attention_layers[0], residual_weight)

    # Multiply through remaining layers
    for layer_attn in attention_layers[1:]:
        layer_with_residual = add_residual_connection(layer_attn, residual_weight)
        # Matrix multiplication: rollout = layer @ rollout
        rollout = torch.bmm(layer_with_residual, rollout)

    # Normalize rows to ensure valid probability distribution
    if normalize:
        row_sum = rollout.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        rollout = rollout / row_sum

    return rollout


def aggregate_heads(
    attention: torch.Tensor,
    method: str = 'mean'
) -> torch.Tensor:
    """
    Aggregate attention across heads.

    Args:
        attention: (batch, heads, seq, seq) multi-head attention
        method: 'mean', 'max', or 'sum'

    Returns:
        aggregated: (batch, seq, seq) single-head attention
    """
    if method == 'mean':
        return attention.mean(dim=1)
    elif method == 'max':
        return attention.max(dim=1).values
    elif method == 'sum':
        return attention.sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def build_global_attention_graph(
    attention_weights: Dict[int, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    semantic_layers: int = 4,
    residual_weight: float = 0.5,
    head_aggregation: str = 'mean',
    use_rollout: bool = True
) -> torch.Tensor:
    """
    Build global attention graph from layer-wise attention.

    This is the main entry point for constructing the attention graph
    used in AG-SAR uncertainty quantification.

    Args:
        attention_weights: Dict[layer_idx, (batch, heads, seq, seq)]
        attention_mask: (batch, seq) padding mask (1=valid, 0=padding)
        semantic_layers: Use last N layers (where semantic info consolidates)
        residual_weight: Weight for residual connection in rollout
        head_aggregation: How to aggregate heads ('mean', 'max', 'sum')
        use_rollout: Whether to use attention rollout (True) or simple average (False)

    Returns:
        global_attention: (batch, seq, seq) adjacency matrix for graph analysis
    """
    if not attention_weights:
        raise ValueError("attention_weights dict is empty")

    # Sort layers and take last N for semantic analysis
    layer_indices = sorted(attention_weights.keys())
    if semantic_layers > 0 and semantic_layers < len(layer_indices):
        layer_indices = layer_indices[-semantic_layers:]

    # Aggregate heads within each layer
    aggregated_layers = []
    for layer_idx in layer_indices:
        attn = attention_weights[layer_idx]  # (batch, heads, seq, seq)
        layer_attn = aggregate_heads(attn, method=head_aggregation)
        aggregated_layers.append(layer_attn)

    # Apply padding mask BEFORE rollout (CRITICAL for preventing sink artifacts)
    if attention_mask is not None:
        masked_layers = []
        for layer_attn in aggregated_layers:
            masked = apply_attention_mask(layer_attn, attention_mask, normalize=True)
            masked_layers.append(masked)
        aggregated_layers = masked_layers

    if use_rollout:
        # Compute attention rollout across layers
        global_attention = compute_attention_rollout(
            aggregated_layers,
            residual_weight=residual_weight,
            normalize=True
        )
    else:
        # Simple average across layers (faster but less accurate)
        stacked = torch.stack(aggregated_layers, dim=0)
        global_attention = stacked.mean(dim=0)

    return global_attention
