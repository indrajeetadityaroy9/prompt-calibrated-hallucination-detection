"""
Graph-Based Centrality Measures.

Matrix-free O(N) eigenvector centrality computation using power iteration.
The key innovation: no O(N^2) attention matrix materialization.

Core formula: R(t_i) = C_eigen(t_i) * ||v_i||_2
This naturally filters attention sinks (high centrality, low value norms).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

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
    # SGSS parameters
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
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
        surprisal: (B, S) per-token surprisal for SGSS
        head_scores: (L*H,) Z-scored head calibration scores for SGSS
        steering_alpha: SGSS steering strength
        steering_beta: SGSS gate sensitivity
        response_start: Start index for response tokens

    Returns:
        centrality: (B, S) normalized eigenvector centrality
        per_head_contrib: (B, L*H, S) if return_per_head, else None
    """
    B, total_heads, S, D = Q_stack.shape
    device = Q_stack.device
    dtype = Q_stack.dtype

    # Initialize uniform centrality
    v = torch.ones(B, S, device=device, dtype=dtype) / S

    if attention_mask is not None:
        v = v * attention_mask.float()
        v = v / v.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    # Pre-compute signal filter from value norms
    signal_filter = None
    if value_norms is not None:
        signal_filter = value_norms / (value_norms.max(dim=-1, keepdim=True)[0] + 1e-6)
        signal_filter = signal_filter.to(dtype)

    final_per_head_out = None

    for _ in range(num_iterations):
        # Inject signal strength
        if signal_filter is not None:
            v_in = v * signal_filter
            v_in = v_in / v_in.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            v_in = v

        # Triton/PyTorch kernel: attention-weighted sum per head
        per_head_out = centrality_kernel(
            Q_stack, K_stack, v_in,
            surprisal=surprisal,
            head_scores=head_scores,
            steering_alpha=steering_alpha,
            steering_beta=steering_beta,
            response_start=response_start,
        )

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
    hebbian_weights: Optional[torch.Tensor] = None,
    use_hebbian: bool = False,
    # SGSS parameters
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
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
        hebbian_weights: (B, S) Hebbian weights for MC-SS
        use_hebbian: Use Hebbian-filtered iteration
        surprisal: (B, S) per-token surprisal for SGSS
        head_scores: (L*H,) head calibration scores for SGSS
        steering_alpha: SGSS steering strength
        steering_beta: SGSS gate sensitivity
        response_start: Start index for response tokens

    Returns:
        relevance: (B, S) sink-aware relevance scores
        centrality: (B, S) raw eigenvector centrality
        per_head_contrib: (B, L*H, S) if return_raw, else None
    """
    if Q_stack is None or K_stack is None:
        raise ValueError("Q_stack and K_stack required for Flash Centrality")

    # Optionally modulate value norms with Hebbian weights
    effective_value_norms = value_norms
    if use_hebbian and hebbian_weights is not None:
        effective_value_norms = value_norms * hebbian_weights

    centrality, per_head_contrib = compute_centrality(
        Q_stack=Q_stack,
        K_stack=K_stack,
        num_iterations=num_iterations,
        residual_weight=residual_weight,
        tol=tol,
        attention_mask=attention_mask,
        value_norms=effective_value_norms,
        return_per_head=return_raw,
        surprisal=surprisal,
        head_scores=head_scores,
        steering_alpha=steering_alpha,
        steering_beta=steering_beta,
        response_start=response_start,
    )

    # Sink-aware relevance
    relevance = centrality * value_norms

    # Zero out sink tokens
    if sink_token_count > 0:
        relevance[:, :sink_token_count] = 0.0

    if attention_mask is not None:
        relevance = relevance * attention_mask.float()

    return relevance, centrality, per_head_contrib


def compute_hebbian_weights(
    K_stack: torch.Tensor,
    prompt_end_idx: int,
    tau: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Hebbian weights using consensus embedding.

    Uses "Anchored Centroid" approach: compute centroid from PROMPT tokens
    only, then measure similarity of all tokens to this anchor.

    Args:
        K_stack: (B, H_total, S, D) stacked key vectors
        prompt_end_idx: Index where prompt ends
        tau: Similarity threshold for Hebbian prior
        attention_mask: (B, S) optional padding mask

    Returns:
        hebbian_weights: (B, S) in [0, 1]
    """
    B, total_heads, S, D = K_stack.shape

    # Consensus embedding: mean over heads
    K_consensus = K_stack.mean(dim=1)

    # Handle edge cases
    prompt_end_idx = max(1, min(prompt_end_idx, S))

    # Prompt centroid
    K_prompt = K_consensus[:, :prompt_end_idx, :]

    if attention_mask is not None:
        prompt_mask = attention_mask[:, :prompt_end_idx].unsqueeze(-1).float()
        K_prompt_masked = K_prompt * prompt_mask
        valid_count = prompt_mask.sum(dim=1).clamp(min=1)
        centroid = K_prompt_masked.sum(dim=1) / valid_count
    else:
        centroid = K_prompt.mean(dim=1)

    # Cosine similarity
    orig_dtype = K_consensus.dtype
    K_norm = F.normalize(K_consensus.float(), p=2, dim=-1)
    C_norm = F.normalize(centroid.unsqueeze(1).float(), p=2, dim=-1)

    sim = torch.matmul(K_norm, C_norm.transpose(1, 2)).squeeze(-1)
    sim = sim.to(orig_dtype)

    # ReLU threshold + max-normalize
    weights = torch.relu(sim - tau)
    weights = weights / (weights.max(dim=-1, keepdim=True).values + 1e-6)

    if attention_mask is not None:
        weights = weights * attention_mask.float()

    return weights


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
