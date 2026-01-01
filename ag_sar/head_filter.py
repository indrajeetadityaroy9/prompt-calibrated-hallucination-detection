"""
Entropy-based filtering of attention heads.

Filters out heads that are:
- Too focused (entropy < threshold_low): Attend only to specific tokens
- Too diffuse (entropy > threshold_high): Attend uniformly to everything

Keeps "sparse" semantic heads that show meaningful attention patterns.
"""

from typing import Dict, Optional, Tuple
import torch


def compute_attention_entropy(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10
) -> torch.Tensor:
    """
    Compute entropy of attention distributions.

    H = -sum(p * log(p)) normalized by max possible entropy.

    Args:
        attention: (batch, heads, seq, seq) attention weights
        attention_mask: (batch, seq) valid token mask (1=valid, 0=padding)
        eps: Small constant for numerical stability

    Returns:
        entropy: (batch, heads, seq) normalized entropy per query position
    """
    # Clamp for numerical stability in log
    attention = attention.clamp(min=eps)

    # Compute entropy: H = -sum(p * log(p)) along last dim (key positions)
    entropy = -torch.sum(attention * torch.log(attention), dim=-1)

    # Normalize by max possible entropy
    if attention_mask is not None:
        # Max entropy is log(num_valid_tokens) for each sequence
        valid_count = attention_mask.sum(dim=-1, keepdim=True).float().clamp(min=1)
        max_entropy = torch.log(valid_count)
        # Expand for heads dimension: (batch, 1, 1)
        max_entropy = max_entropy.unsqueeze(1)
    else:
        seq_len = attention.size(-1)
        max_entropy = torch.log(
            torch.tensor(seq_len, dtype=attention.dtype, device=attention.device)
        )

    # Normalized entropy in [0, 1]
    normalized_entropy = entropy / max_entropy.clamp(min=eps)

    return normalized_entropy


def compute_head_entropy(
    attention: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute mean entropy per attention head.

    Args:
        attention: (batch, heads, seq, seq) attention weights
        attention_mask: (batch, seq) valid token mask

    Returns:
        head_entropy: (batch, heads) mean normalized entropy per head
    """
    # Get per-position entropy
    token_entropy = compute_attention_entropy(attention, attention_mask)

    if attention_mask is not None:
        # Average only over valid tokens
        mask = attention_mask.unsqueeze(1).float()  # (batch, 1, seq)
        masked_entropy = token_entropy * mask
        valid_count = mask.sum(dim=-1).clamp(min=1)  # (batch, 1)
        head_entropy = masked_entropy.sum(dim=-1) / valid_count
    else:
        head_entropy = token_entropy.mean(dim=-1)

    return head_entropy


def create_head_mask(
    head_entropy: torch.Tensor,
    entropy_low: float = 0.3,
    entropy_high: float = 0.95
) -> torch.Tensor:
    """
    Create boolean mask for heads within entropy thresholds.

    Args:
        head_entropy: (batch, heads) or (num_layers, batch, heads) entropy values
        entropy_low: Minimum entropy threshold (heads below are too focused)
        entropy_high: Maximum entropy threshold (heads above are too diffuse)

    Returns:
        mask: Boolean tensor, True for heads to keep
    """
    return (head_entropy >= entropy_low) & (head_entropy <= entropy_high)


def filter_heads_by_entropy(
    attention_weights: Dict[int, torch.Tensor],
    entropy_low: float = 0.3,
    entropy_high: float = 0.95,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Filter attention heads by entropy threshold.

    Removes heads that are too focused (< entropy_low) or
    too diffuse (> entropy_high).

    Args:
        attention_weights: Dict[layer_idx, (batch, heads, seq, seq)]
        entropy_low: Minimum entropy threshold
        entropy_high: Maximum entropy threshold
        attention_mask: (batch, seq) valid token mask

    Returns:
        filtered_weights: Dict with same structure, masked heads zeroed
        head_mask: (num_layers, heads) boolean mask of included heads
    """
    if not attention_weights:
        raise ValueError("attention_weights dict is empty")

    layer_indices = sorted(attention_weights.keys())
    num_layers = len(layer_indices)

    # Compute entropy for each layer
    all_entropies = []
    num_heads = None

    for layer_idx in layer_indices:
        attn = attention_weights[layer_idx]
        if num_heads is None:
            num_heads = attn.size(1)

        head_entropy = compute_head_entropy(attn, attention_mask)
        # Average across batch for consistent filtering
        mean_entropy = head_entropy.mean(dim=0)  # (heads,)
        all_entropies.append(mean_entropy)

    # Stack: (num_layers, heads)
    all_entropies = torch.stack(all_entropies, dim=0)

    # Create mask: True if entropy in valid range
    head_mask = create_head_mask(all_entropies, entropy_low, entropy_high)

    # Apply mask to attention weights
    filtered_weights = {}
    for i, layer_idx in enumerate(layer_indices):
        attn = attention_weights[layer_idx]  # (batch, heads, seq, seq)
        layer_mask = head_mask[i]  # (heads,)

        # Expand mask for broadcasting: (1, heads, 1, 1)
        mask_expanded = layer_mask.view(1, -1, 1, 1).float()

        # Zero out filtered heads
        filtered_attn = attn * mask_expanded
        filtered_weights[layer_idx] = filtered_attn

    return filtered_weights, head_mask
