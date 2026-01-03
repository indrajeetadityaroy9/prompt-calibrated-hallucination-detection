"""
Sink-aware eigenvector centrality computation.

Uses matrix-free O(N) memory computation via Flash-style Triton kernel.

Key formula: R(t_i) = C_eigen(t_i) × ||v_i||_2

This naturally filters attention sinks which have high centrality
but low value norms (mechanistically passive tokens).
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F

from .kernels import centrality_flash_fwd


def matrix_free_power_iteration(
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    num_iterations: int = 3,
    residual_weight: float = 0.5,
    tol: float = 1e-4,
    attention_mask: Optional[torch.Tensor] = None,
    return_raw: bool = False,
    value_norms: Optional[torch.Tensor] = None,
    # SGSS parameters
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute eigenvector centrality via matrix-free power iteration.

    Uses Triton kernel to avoid O(N^2) attention matrix materialization.
    This is the core of the "Flash Centrality" approach.

    Algorithm per iteration:
        1. v_in = v * signal_filter  # Inject value norms for non-uniform signal
        2. per_head_out = triton_kernel(Q, K, v_in)  # Attention-weighted sum
        3. attn_contrib = mean(per_head_out, dim=heads)
        4. v_next = (1 - residual_weight) * attn_contrib + residual_weight * v
        5. v_next = normalize(v_next)

    CRITICAL: Injecting value_norms inside the loop (not just at the end) breaks
    the uniform fixed point. With spiky v_in:
    - Different heads route semantic spikes to different destinations
    - SGSS dynamically adjusts head contributions based on surprisal

    The residual connection prevents centrality collapse on early tokens in causal
    attention, equivalent to adding identity matrix in the matrix-based approach.

    Args:
        Q_stack: Stacked query vectors from semantic layers (B, L*H, S, D)
        K_stack: Stacked key vectors from semantic layers (B, L*H, S, D)
        num_iterations: Power iteration steps (3 typically sufficient)
        residual_weight: Weight for self-attention residual (0.5 = equal mix)
        tol: Convergence tolerance for early stopping
        attention_mask: (B, S) padding mask (1=valid, 0=padding)
        return_raw: If True, return per-head contributions for head specialization analysis
        value_norms: (B, S) value vector norms for signal injection (breaks uniform fixed point)
        surprisal: (B, S) optional per-token surprisal for SGSS gating
        head_scores: (L*H,) optional Z-scored head calibration scores for SGSS
        steering_alpha: SGSS steering strength (default 2.0)
        steering_beta: SGSS gate sensitivity (default 5.0)
        response_start: Start index for response tokens (SGSS only applied to response)

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

    # Pre-compute signal filter from value norms
    # This creates "spiky" non-uniform input that breaks the uniform fixed point
    # High value norms = semantically important tokens (entities, content words)
    # Low value norms = attention sinks (BOS, punctuation, stop words)
    if value_norms is not None:
        # Normalize to [0, 1] range to avoid numerical issues
        signal_filter = value_norms / (value_norms.max(dim=-1, keepdim=True)[0] + 1e-6)
        # Ensure it's the right dtype
        signal_filter = signal_filter.to(dtype)
    else:
        signal_filter = None

    # Track per-head contributions from final iteration
    final_per_head_out = None

    for _ in range(num_iterations):
        # INJECT SIGNAL STRENGTH: Propagate weighted centrality, not just centrality
        # With spiky v_in, different heads will produce different outputs
        # This makes head_weights meaningful for differentiating Truth vs Induction Heads
        if signal_filter is not None:
            v_in = v * signal_filter
            # Re-normalize to maintain probability distribution
            v_in = v_in / v_in.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        else:
            v_in = v

        # Triton kernel: attention-weighted sum per head (with optional SGSS)
        # per_head_out[b, h, i] = sum_j softmax(Q@K.T)[i,j] * v_in[j] * w_sgss
        per_head_out = centrality_flash_fwd(
            Q_stack, K_stack, v_in,
            surprisal=surprisal,
            head_scores=head_scores,
            steering_alpha=steering_alpha,
            steering_beta=steering_beta,
            response_start=response_start,
        )  # (B, L*H, S)

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


def compute_hebbian_weights(
    K_stack: torch.Tensor,
    prompt_end_idx: int,
    tau: float = 0.1,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Hebbian weights using CONSENSUS EMBEDDING (head-averaged).

    Uses the "Anchored Centroid" approach: compute centroid from PROMPT tokens
    only, then measure similarity of all tokens to this anchor. This prevents
    hallucinated tokens from polluting the centroid.

    The key insight is that tokens grounded in the prompt (factual) will have
    high similarity to the prompt centroid, while hallucinated tokens that
    drift semantically will have low similarity.

    Algorithm:
        1. Collapse heads: K_consensus = K_stack.mean(dim=1)  # (B, S, D)
        2. Compute prompt centroid: μ = K_consensus[:, :prompt_end].mean(dim=1)
        3. Compute similarity: h_t = cos_sim(K_t, μ)
        4. Apply threshold: h_t = ReLU(h_t - τ)
        5. Max-normalize to [0, 1]

    Args:
        K_stack: (B, H_total, S, D) stacked key vectors from semantic layers
        prompt_end_idx: Index where prompt ends (exclusive)
        tau: Similarity threshold for Hebbian prior (default 0.1)
        attention_mask: (B, S) optional padding mask

    Returns:
        hebbian_weights: (B, S) Hebbian similarity weights in [0, 1]
    """
    B, total_heads, S, D = K_stack.shape

    # 1. Collapse Heads FIRST (Consensus Embedding)
    # This extracts the "average semantic representation" across all heads
    # avoiding fragmentation from specialized heads (syntax, translation, etc.)
    K_consensus = K_stack.mean(dim=1)  # (B, S, D)

    # Handle edge case: if prompt_end_idx >= S, use entire sequence
    if prompt_end_idx >= S:
        prompt_end_idx = S

    # Handle edge case: if prompt_end_idx <= 0, use first token
    if prompt_end_idx <= 0:
        prompt_end_idx = 1

    # 2. Extract Prompt Anchor (CRITICAL: only prompt tokens, not generated)
    # This anchors the Hebbian prior to ground truth, preventing hallucination
    # from polluting the centroid
    K_prompt = K_consensus[:, :prompt_end_idx, :]  # (B, P, D)

    # Apply mask to prompt if provided (for padded batches)
    if attention_mask is not None:
        prompt_mask = attention_mask[:, :prompt_end_idx]  # (B, P)
        # Weight prompt keys by mask
        prompt_mask_expanded = prompt_mask.unsqueeze(-1).float()  # (B, P, 1)
        K_prompt_masked = K_prompt * prompt_mask_expanded
        # Mean over valid tokens
        valid_count = prompt_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        centroid = K_prompt_masked.sum(dim=1) / valid_count  # (B, D)
    else:
        centroid = K_prompt.mean(dim=1)  # (B, D)

    # 3. Normalize for Cosine Similarity
    # Cast to float32 for numerical stability in normalize/matmul
    orig_dtype = K_consensus.dtype
    K_norm = F.normalize(K_consensus.float(), p=2, dim=-1)  # (B, S, D)
    C_norm = F.normalize(centroid.unsqueeze(1).float(), p=2, dim=-1)  # (B, 1, D)

    # 4. Compute Similarity O(N)
    # (B, S, D) @ (B, D, 1) -> (B, S)
    sim = torch.matmul(K_norm, C_norm.transpose(1, 2)).squeeze(-1)
    sim = sim.to(orig_dtype)  # Cast back to original dtype

    # 5. ReLU Threshold (Hebbian Prior)
    # "Is this token semantically closer to the prompt than threshold tau?"
    weights = torch.relu(sim - tau)

    # 6. Max-normalize to [0, 1] for stability
    # Avoid division by zero
    weights_max = weights.max(dim=-1, keepdim=True).values + 1e-6
    weights = weights / weights_max

    # Apply attention mask
    if attention_mask is not None:
        weights = weights * attention_mask.float()

    return weights


def matrix_free_power_iteration_hebbian(
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    hebbian_weights: torch.Tensor,
    num_iterations: int = 3,
    residual_weight: float = 0.5,
    tol: float = 1e-4,
    attention_mask: Optional[torch.Tensor] = None,
    value_norms: Optional[torch.Tensor] = None,
    # SGSS parameters
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
) -> torch.Tensor:
    """
    Compute Hebbian-filtered eigenvector centrality via matrix-free power iteration.

    Modified power iteration that incorporates Hebbian prior AND value norms at each step:
        v^(k+1) = A · (v^(k) ⊙ h_hebbian ⊙ ||V||)

    The Hebbian weights bias toward tokens semantically grounded in the prompt.
    Value norms inject signal strength to break the uniform fixed point.

    Args:
        Q_stack: (B, L*H, S, D) stacked query vectors
        K_stack: (B, L*H, S, D) stacked key vectors
        hebbian_weights: (B, S) Hebbian similarity weights from compute_hebbian_weights()
        num_iterations: Power iteration steps (default 3)
        residual_weight: Weight for self-attention residual (default 0.5)
        tol: Convergence tolerance
        attention_mask: (B, S) padding mask
        value_norms: (B, S) value vector norms for signal injection
        surprisal: (B, S) optional per-token surprisal for SGSS gating
        head_scores: (L*H,) optional Z-scored head calibration scores for SGSS
        steering_alpha: SGSS steering strength (default 2.0)
        steering_beta: SGSS gate sensitivity (default 5.0)
        response_start: Start index for response tokens (SGSS only applied to response)

    Returns:
        centrality: (B, S) Hebbian-filtered eigenvector centrality (L1 normalized)
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

    # Pre-compute combined signal filter (Hebbian + Value Norms)
    # This creates spiky input that makes head_weights meaningful
    signal_filter = hebbian_weights.to(dtype)
    if value_norms is not None:
        # Normalize value norms to [0, 1] and combine with Hebbian
        vn_normalized = value_norms / (value_norms.max(dim=-1, keepdim=True)[0] + 1e-6)
        signal_filter = signal_filter * vn_normalized.to(dtype)

    for _ in range(num_iterations):
        # Apply combined modulation: v_modulated = v * signal_filter
        # This biases iteration toward semantically grounded, high-value tokens
        v_modulated = v * signal_filter

        # Re-normalize after modulation to maintain probability distribution
        v_modulated = v_modulated / v_modulated.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Triton kernel: attention-weighted sum per head (with optional SGSS)
        per_head_out = centrality_flash_fwd(
            Q_stack, K_stack, v_modulated,
            surprisal=surprisal,
            head_scores=head_scores,
            steering_alpha=steering_alpha,
            steering_beta=steering_beta,
            response_start=response_start,
        )  # (B, L*H, S)

        # Aggregate across all heads (mean pooling)
        attn_contrib = per_head_out.mean(dim=1)  # (B, S)

        # Apply residual connection: v_next = (1-w) * Av + w * v
        v_next = (1 - residual_weight) * attn_contrib + residual_weight * v

        # Apply mask
        if attention_mask is not None:
            v_next = v_next * attention_mask.float()

        # Normalize to sum=1 (L1 normalization for iteration stability)
        v_next = v_next / v_next.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        # Early stopping check
        if (v_next - v).abs().max() < tol:
            v = v_next
            break

        v = v_next

    return v


def compute_sink_aware_centrality(
    value_norms: torch.Tensor,
    Q_stack: torch.Tensor,
    K_stack: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    num_iterations: int = 3,
    tol: float = 1e-4,
    residual_weight: float = 0.5,
    # Head specialization analysis
    return_raw: bool = False,
    # Sink token masking (StreamingLLM-style)
    sink_token_count: int = 0,
    # MC-SS Hebbian filtering
    hebbian_weights: Optional[torch.Tensor] = None,
    use_hebbian: bool = False,
    # SGSS: Surprisal-Gated Spectral Steering
    surprisal: Optional[torch.Tensor] = None,
    head_scores: Optional[torch.Tensor] = None,
    steering_alpha: float = 2.0,
    steering_beta: float = 5.0,
    response_start: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Compute sink-aware relevance: R(t_i) = C_eigen(t_i) × ||v_i||_2

    The value norm weighting naturally filters attention sinks which have
    high centrality but low value norms (mechanistically passive tokens
    like <s>, punctuation, etc.).

    Uses matrix-free O(N) computation via Triton kernel.

    Optionally supports Hebbian filtering for MC-SS:
    - Pass hebbian_weights and use_hebbian=True
    - Uses matrix_free_power_iteration_hebbian instead of standard

    Args:
        value_norms: (batch, seq) L2 norm of value vectors per token
        Q_stack: (B, L*H, S, D) stacked queries from semantic layers
        K_stack: (B, L*H, S, D) stacked keys from semantic layers
        attention_mask: (batch, seq) valid token mask (1=valid, 0=padding)
        num_iterations: Power iteration steps
        tol: Convergence tolerance
        residual_weight: Weight for self-attention residual
        return_raw: If True, return per-head contributions for head specialization
        sink_token_count: Number of initial tokens to zero out (BOS/sink tokens)
            These have high centrality but are structural, not semantic.
        hebbian_weights: (B, S) pre-computed Hebbian similarity weights for MC-SS
        use_hebbian: If True, use Hebbian-filtered power iteration
        surprisal: (B, S) optional per-token surprisal for SGSS gating
        head_scores: (L*H,) optional Z-scored head calibration scores for SGSS
        steering_alpha: SGSS steering strength (default 2.0)
        steering_beta: SGSS gate sensitivity (default 5.0)
        response_start: Start index for response tokens (SGSS only applied to response)

    Returns:
        relevance: (batch, seq) sink-aware relevance scores
        centrality: (batch, seq) raw eigenvector centrality (before sink correction)
        per_head_contrib: (B, L*H, S) per-head contributions if return_raw=True, else None
    """
    # AG-SAR v2 requires Q/K stacks for matrix-free Flash Centrality
    if Q_stack is None or K_stack is None:
        raise ValueError(
            "AG-SAR v2 requires Q_stack and K_stack for Flash Centrality. "
            "Ensure AttentionExtractor is returning semantic Q/K tensors."
        )

    per_head_contrib = None

    if use_hebbian and hebbian_weights is not None:
        # MC-SS path: Hebbian-filtered power iteration
        # Also inject value_norms to break uniform fixed point
        centrality = matrix_free_power_iteration_hebbian(
            Q_stack=Q_stack,
            K_stack=K_stack,
            hebbian_weights=hebbian_weights,
            num_iterations=num_iterations,
            residual_weight=residual_weight,
            tol=tol,
            attention_mask=attention_mask,
            value_norms=value_norms,
            # SGSS parameters
            surprisal=surprisal,
            head_scores=head_scores,
            steering_alpha=steering_alpha,
            steering_beta=steering_beta,
            response_start=response_start,
        )
        # Hebbian version doesn't return per-head contributions
        per_head_contrib = None
    else:
        # Standard matrix-free path using Triton kernel
        # CRITICAL: Pass value_norms to inject signal strength inside iteration loop
        # This breaks the uniform fixed point and makes head_weights effective
        centrality, per_head_contrib = matrix_free_power_iteration(
            Q_stack=Q_stack,
            K_stack=K_stack,
            num_iterations=num_iterations,
            residual_weight=residual_weight,
            tol=tol,
            attention_mask=attention_mask,
            return_raw=return_raw,
            value_norms=value_norms,
            # SGSS parameters
            surprisal=surprisal,
            head_scores=head_scores,
            steering_alpha=steering_alpha,
            steering_beta=steering_beta,
            response_start=response_start,
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
