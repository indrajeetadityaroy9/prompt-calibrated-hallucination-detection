"""
AG-SAR v3.1 Pure PyTorch Functional Operations.

Implements the mathematical kernels from the literature synthesis:
- Paper 1 (ViT Registers): Kurtosis-based register detection
- Paper 2 (StreamingLLM): Sink token handling
- Paper 6 (Lookback Lens): Authority-weighted context ratio
- Paper 9 (GSP): Spectral roughness via causal residual

All operations are O(N) and designed for streaming inference.

H100 Optimizations:
- torch.compile decorators on hot paths for Inductor backend
- mode="reduce-overhead" for iterative computations
- fullgraph=True for maximum optimization
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


# =============================================================================
# torch.compile Configuration for H100/Hopper
# =============================================================================

def _should_compile() -> bool:
    """Check if torch.compile should be used."""
    return (
        hasattr(torch, 'compile') and
        torch.cuda.is_available() and
        torch.cuda.get_device_capability()[0] >= 8  # Ampere+
    )


# Decorator for compiled hot paths
def _compile_if_available(mode: str = "default"):
    """Conditionally apply torch.compile based on hardware.

    Uses mode="default" to avoid CUDAGraph issues with dynamic shapes.
    """
    def decorator(func):
        if _should_compile():
            # Use default mode (no CUDAGraphs) to handle dynamic sequence lengths
            return torch.compile(func, mode="default", dynamic=True)
        return func
    return decorator


@dataclass
class EMAState:
    """
    State container for Welford's Online EMA Statistics.

    Used for streaming Z-score normalization in the Register Filter.
    Initialize with pre-computed constants, then adapt online.

    Attributes:
        mean: Running mean per layer (L,) or scalar
        var: Running variance per layer (L,) or scalar
        count: Number of updates (for warmup handling)
    """
    mean: torch.Tensor
    var: torch.Tensor
    count: int = 0

    @classmethod
    def initialize(
        cls,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        init_mean: float = 0.0,
        init_var: float = 1.0,
    ) -> "EMAState":
        """Initialize EMA state with default statistics."""
        return cls(
            mean=torch.full((num_layers,), init_mean, device=device, dtype=dtype),
            var=torch.full((num_layers,), init_var, device=device, dtype=dtype),
            count=0,
        )


# =============================================================================
# GQA (Grouped Query Attention) Support - Llama-3.1 Compatibility
# =============================================================================

def align_gqa_heads(
    v_states: torch.Tensor,
    n_q_heads: int,
) -> torch.Tensor:
    """
    Align GQA Value states to match Query head count.

    Llama-3.1 uses Grouped Query Attention (GQA):
    - Llama-3.1-8B: 32 Query heads, 8 KV heads (4x repetition)
    - Llama-3.1-70B: 64 Query heads, 8 KV heads (8x repetition)

    This function expands KV heads to match Query heads via repeat-interleave,
    enabling the spectral roughness calculation: ||h_attn - Σ A·v||

    Args:
        v_states: (B, n_kv_heads, S, head_dim) Value states from KV cache
        n_q_heads: Number of query heads (e.g., 32 for Llama-3.1-8B)

    Returns:
        v_aligned: (B, n_q_heads, S, head_dim) Value states expanded to match queries

    Example:
        >>> v = torch.randn(1, 8, 128, 128)  # 8 KV heads
        >>> v_aligned = align_gqa_heads(v, n_q_heads=32)  # -> (1, 32, 128, 128)
    """
    if v_states.dim() == 3:
        # (B, S, D) format - no head dimension, return as-is
        return v_states

    batch, n_kv_heads, seq_len, head_dim = v_states.shape

    if n_kv_heads == n_q_heads:
        # MHA: No expansion needed
        return v_states

    if n_q_heads % n_kv_heads != 0:
        raise ValueError(
            f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

    n_rep = n_q_heads // n_kv_heads

    # Expand: [B, KV, S, D] -> [B, KV, 1, S, D] -> [B, KV, Rep, S, D]
    v_expanded = v_states.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, head_dim)

    # Flatten: [B, KV, Rep, S, D] -> [B, KV*Rep, S, D] = [B, Q, S, D]
    v_aligned = v_expanded.reshape(batch, n_q_heads, seq_len, head_dim)

    return v_aligned


def get_gqa_config(model_config) -> Tuple[int, int, int]:
    """
    Extract GQA configuration from model config.

    Args:
        model_config: HuggingFace model config object

    Returns:
        Tuple of (n_q_heads, n_kv_heads, n_rep)
    """
    n_q_heads = getattr(model_config, 'num_attention_heads', 32)
    n_kv_heads = getattr(model_config, 'num_key_value_heads', n_q_heads)
    n_rep = n_q_heads // n_kv_heads
    return n_q_heads, n_kv_heads, n_rep


# =============================================================================
# Register Filter (Mechanism 1)
# =============================================================================

def fisher_kurtosis(
    x: torch.Tensor,
    dim: int = -1,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """
    Compute Fisher Kurtosis (excess kurtosis) in O(N).

    Fisher Kurtosis = E[(x-μ)^4] / σ^4 - 3

    For a normal distribution, Fisher kurtosis = 0.
    - Positive kurtosis (leptokurtic): Heavy tails, sharp peak
    - Negative kurtosis (platykurtic): Light tails, flat peak

    Used for Register Filter (Mechanism 1):
    - Semantic tokens: High kurtosis (spiky, concentrated features)
    - Register/Sink tokens: Low kurtosis (uniform, diffuse features)

    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute kurtosis
        epsilon: Numerical stability for std clamping

    Returns:
        Kurtosis values with `dim` reduced

    Example:
        >>> v = torch.randn(32, 128)  # (batch, hidden)
        >>> kurt = fisher_kurtosis(v, dim=-1)  # (batch,)
    """
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp(min=epsilon)

    # Z-score normalization
    z = (x - mean) / std

    # Fourth moment minus 3 (Fisher's excess kurtosis)
    kurt = (z ** 4).mean(dim=dim) - 3.0

    return kurt


def welford_update(
    curr_val: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    decay: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Welford's Algorithm for Online EMA Statistics.

    Implements exponentially-weighted moving average/variance
    for streaming Z-score normalization.

    Update equations:
        μ_t = (1-α)μ_{t-1} + α × x_t
        σ²_t = (1-α)σ²_{t-1} + α × (x_t - μ_t)(x_t - μ_{t-1})

    where α = 1 - decay.

    Args:
        curr_val: Current observation (any shape, must broadcast with running_*)
        running_mean: Previous running mean
        running_var: Previous running variance
        decay: EMA decay factor (0.99 = 1% update from new sample)

    Returns:
        Tuple of (new_mean, new_var)

    Example:
        >>> ema = EMAState.initialize(4, device, dtype)
        >>> new_mean, new_var = welford_update(kurt, ema.mean, ema.var, 0.99)
    """
    alpha = 1.0 - decay

    # Update mean: μ_t = μ_{t-1} + α(x_t - μ_{t-1})
    delta = curr_val - running_mean
    new_mean = running_mean + alpha * delta

    # Update variance: σ²_t = (1-α)σ²_{t-1} + α(x_t - μ_new)(x_t - μ_old)
    # This is the incremental variance update (Welford's method)
    new_var = running_var * decay + alpha * delta * (curr_val - new_mean)

    return new_mean, new_var


def compute_register_mask(
    value_vectors: torch.Tensor,
    ema_state: Optional[EMAState] = None,
    kurtosis_threshold: float = 2.0,
    sink_token_count: int = 4,
    ema_decay: float = 0.995,
    update_ema: bool = True,
) -> Tuple[torch.Tensor, Optional[EMAState]]:
    """
    Compute Register Mask M(t) for filtering sinks and registers.

    Implements the v3.1 Filter (Papers 1 & 2):
        M(t) = (t > 4) × Sigmoid(-Z(t) + τ)

    where Z(t) = (Kurt(v_t) - μ_EMA) / σ_EMA

    Low-kurtosis tokens (registers/sinks) → low mask value
    High-kurtosis tokens (semantic) → high mask value

    Args:
        value_vectors: (B, S, D) value vectors per token
        ema_state: Previous EMA state (for online adaptation)
        kurtosis_threshold: τ threshold for sigmoid gate
        sink_token_count: First N tokens to mask as sinks (StreamingLLM)
        ema_decay: Decay factor for EMA update
        update_ema: Whether to update EMA state

    Returns:
        Tuple of:
        - mask: (B, S) register mask in [0, 1]
        - updated_ema_state: Updated EMA state (or None if not updating)

    Example:
        >>> mask, ema = compute_register_mask(v, ema_state, tau=2.0)
        >>> authority = authority * mask  # Filter sinks
    """
    B, S, D = value_vectors.shape
    device = value_vectors.device
    dtype = value_vectors.dtype

    # Compute kurtosis per token
    kurt = fisher_kurtosis(value_vectors, dim=-1)  # (B, S)

    # Initialize or update EMA state
    if ema_state is None:
        # First call: initialize with reasonable defaults from calibration
        # Use init_var=4.0 based on empirical kurtosis variance across models
        # This allows meaningful z-scores on first batch instead of disabling the filter
        ema_state = EMAState(
            mean=torch.tensor([0.0], device=device, dtype=dtype),
            var=torch.tensor([4.0], device=device, dtype=dtype),
            count=1,
        )
        # Compute z-score with initial defaults (not zeros)
        sigma = (ema_state.var + 1e-6).sqrt()
        z_score = (kurt - ema_state.mean) / sigma
    else:
        # Compute Z-score with current EMA statistics
        sigma = (ema_state.var + 1e-6).sqrt()
        z_score = (kurt - ema_state.mean) / sigma

        # Update EMA if requested
        if update_ema:
            batch_mean = kurt.mean()
            batch_var = kurt.var().clamp(min=1e-6)
            new_mean, new_var = welford_update(
                batch_mean, ema_state.mean, ema_state.var, ema_decay
            )
            ema_state = EMAState(
                mean=new_mean,
                var=new_var.clamp(min=1e-6),
                count=ema_state.count + 1,
            )

    # Apply sigmoid gate: high kurtosis → high mask
    # Using -Z so that low Z (low kurtosis) → low mask
    mask = torch.sigmoid(-z_score + kurtosis_threshold)

    # Hard-code sink tokens (positions 0..sink_token_count-1)
    if sink_token_count > 0:
        mask[:, :sink_token_count] = 0.0

    return mask, ema_state


@_compile_if_available(mode="reduce-overhead")
def compute_spectral_roughness(
    h_attn: torch.Tensor,
    value_vectors: torch.Tensor,
    attention_weights: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Spectral Roughness using Dirichlet Energy approximation.

    Implements Dirichlet Energy from Paper 9 (GSP Framework):
        δ(t) = Σ_{j<t} A_{t,j} × ||v_t - v_j||^2

    This measures how much the current token's representation differs
    from its attended tokens - a "rough" signal on the attention graph
    indicates inconsistency that may signal hallucination.

    Unlike the naive ||h_attn - Σ A×v|| which is always ~0 for correct
    attention, this Dirichlet formulation captures actual signal roughness.

    Args:
        h_attn: (B, S, D) attention output (unused in this formulation,
                kept for API compatibility)
        value_vectors: (B, S, D) value vectors per token
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        attention_mask: (B, S) optional padding mask

    Returns:
        roughness: (B, S) spectral roughness per token (normalized)

    Example:
        >>> delta = compute_spectral_roughness(h_attn, v, attn_weights)
        >>> authority_penalized = authority / (1 + lambda * delta)
    """
    B, S, D = value_vectors.shape

    # Mean-pool attention over heads if needed
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=1)  # (B, S, S)
    else:
        attn = attention_weights

    # Compute pairwise squared distances: ||v_i - v_j||^2
    # Efficient: ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 * v_i · v_j
    v_norm_sq = (value_vectors ** 2).sum(dim=-1, keepdim=True)  # (B, S, 1)
    v_dot = torch.bmm(value_vectors, value_vectors.transpose(-1, -2))  # (B, S, S)
    pairwise_dist_sq = v_norm_sq + v_norm_sq.transpose(-1, -2) - 2 * v_dot  # (B, S, S)
    pairwise_dist_sq = pairwise_dist_sq.clamp(min=0)  # Numerical stability

    # Dirichlet Energy: δ(t) = Σ_j A_{t,j} × ||v_t - v_j||^2
    # (B, S, S) * (B, S, S) -> sum over j -> (B, S)
    roughness = (attn * pairwise_dist_sq).sum(dim=-1)  # (B, S)

    # Normalize by hidden dimension for scale invariance
    roughness = roughness / D

    # Apply mask if provided
    if attention_mask is not None:
        roughness = roughness * attention_mask.float()

    return roughness


@_compile_if_available(mode="reduce-overhead")
def compute_mlp_divergence(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MLP Divergence for Llama-3 hallucination detection.

    v3.2 Hypothesis: When a model hallucinates, the MLP layer overrides
    the attention layer's signal with parametric memory.
    - Grounded: Attention says "Berlin" (context) → MLP refines → Vectors align
    - Hallucination: Attention sees "Berlin" → MLP overrules with "Paris" → Divergence

    The metric: δ(t) = 1 - CosineSim(h_attn, h_block)

    Args:
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        attention_mask: (B, S) optional padding mask

    Returns:
        divergence: (B, S) MLP divergence per token [0, 2] where:
            - 0 = perfect alignment (attention and MLP agree)
            - 1 = orthogonal
            - 2 = opposite directions (maximum divergence)

    Example:
        >>> div = compute_mlp_divergence(h_attn, h_block)
        >>> authority_penalized = authority / (1 + lambda * div)
    """
    # Normalize for cosine similarity
    h_attn_norm = F.normalize(h_attn, p=2, dim=-1)
    h_block_norm = F.normalize(h_block, p=2, dim=-1)

    # Cosine similarity: (B, S)
    cos_sim = (h_attn_norm * h_block_norm).sum(dim=-1)

    # Divergence = 1 - cosine_similarity
    # Range: [0, 2] where 0 = aligned, 2 = opposite
    divergence = 1.0 - cos_sim

    # Apply mask if provided
    if attention_mask is not None:
        divergence = divergence * attention_mask.float()

    return divergence


def compute_spectral_roughness_gqa(
    h_attn: torch.Tensor,
    v_states: torch.Tensor,
    attention_weights: torch.Tensor,
    n_q_heads: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Spectral Roughness with GQA head alignment for Llama-3.1.

    Handles the shape mismatch between Query heads and KV heads in GQA:
    - Llama-3.1-8B: 32 Query heads, 8 KV heads
    - Attention weights: (B, 32, S, S)
    - Value states: (B, 8, S, head_dim) -> expanded to (B, 32, S, head_dim)

    Formula: δ(t) = ||h_attn(t) - Σ A_{t,j} × v_j||_2

    Args:
        h_attn: Attention output. Can be:
            - (B, S, D): Merged hidden states (post-o_proj)
            - (B, H, S, head_dim): Per-head attention output (pre-o_proj)
        v_states: Value states. Can be:
            - (B, S, D): Merged value states
            - (B, n_kv_heads, S, head_dim): Per-head value states (GQA)
        attention_weights: (B, n_q_heads, S, S) attention weights
        n_q_heads: Number of query heads (for GQA alignment). If None, inferred.
        attention_mask: (B, S) optional padding mask

    Returns:
        roughness: (B, S) spectral roughness per token

    Example:
        >>> # Llama-3.1-8B: 32 Q heads, 8 KV heads
        >>> h = torch.randn(1, 32, 128, 128)  # (B, H, S, head_dim)
        >>> v = torch.randn(1, 8, 128, 128)   # (B, KV, S, head_dim)
        >>> A = torch.randn(1, 32, 128, 128)  # (B, H, S, S)
        >>> delta = compute_spectral_roughness_gqa(h, v, A, n_q_heads=32)
    """
    # Handle different input formats
    if h_attn.dim() == 3:
        # (B, S, D) format
        B, S, D = h_attn.shape

        if v_states.dim() == 4:
            # v_states: (B, n_kv_heads, S, head_dim)
            # Need to expand KV heads to match Q heads, then flatten
            n_kv = v_states.shape[1]
            head_dim = v_states.shape[-1]

            # Infer n_q_heads from D and head_dim
            if n_q_heads is None:
                n_q_heads = D // head_dim

            # Expand KV heads to Q heads via repeat-interleave
            v_expanded = align_gqa_heads(v_states, n_q_heads)  # (B, n_q_heads, S, head_dim)
            v_flat = v_expanded.permute(0, 2, 1, 3).reshape(B, S, n_q_heads * head_dim)
        else:
            v_flat = v_states

        return compute_spectral_roughness(h_attn, v_flat, attention_weights, attention_mask)

    # Per-head format: (B, H, S, head_dim)
    B, n_h, S, head_dim = h_attn.shape

    # Infer n_q_heads from attention weights if not provided
    if n_q_heads is None:
        if attention_weights.dim() == 4:
            n_q_heads = attention_weights.shape[1]
        else:
            n_q_heads = n_h

    # Align value states to query head count (GQA expansion)
    if v_states.dim() == 4:
        v_aligned = align_gqa_heads(v_states, n_q_heads)  # (B, n_q_heads, S, head_dim)
    else:
        # (B, S, D) - reshape to per-head
        D = v_states.shape[-1]
        v_aligned = v_states.view(B, S, n_q_heads, head_dim).permute(0, 2, 1, 3)

    # Ensure attention weights have head dimension
    if attention_weights.dim() == 3:
        attn = attention_weights.unsqueeze(1).expand(B, n_q_heads, S, S)
    else:
        attn = attention_weights  # (B, H, S, S)

    # Compute expected value per head: v_expected = A @ v
    # (B, H, S, S) @ (B, H, S, head_dim) -> (B, H, S, head_dim)
    v_expected = torch.matmul(attn, v_aligned)

    # Compute L2 deviation per head, then mean across heads
    # ||h_attn - v_expected||_2 per position
    delta_per_head = (h_attn - v_expected).norm(dim=-1, p=2)  # (B, H, S)
    roughness = delta_per_head.mean(dim=1)  # (B, S) - mean over heads

    # Apply mask if provided
    if attention_mask is not None:
        roughness = roughness * attention_mask.float()

    return roughness


def compute_authority_flow(
    attention_weights: torch.Tensor,
    prompt_length: int,
    register_mask: Optional[torch.Tensor] = None,
    previous_authority: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Recursive Authority with Prompt Recharge in O(N).

    Implements the v3.1 Authority Flow (Paper 6 corrected):
        𝒜(t) = [Σ_{j ∈ Prompt} A_{t,j}] + [Σ_{j ∈ Gen} A_{t,j} × 𝒜(j)] × M(t)

    Key insight: Prompt tokens always contribute 1.0 (source of truth),
    preventing the "Vanishing Authority" problem in long sequences.

    For the first forward pass (no previous_authority), prompt tokens
    get authority=1.0 and generated tokens get authority based on
    how much they attend to the prompt.

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends (exclusive)
        register_mask: (B, S) register mask M(t) in [0, 1]
        previous_authority: (B, S) authority from previous computation
        attention_mask: (B, S) padding mask

    Returns:
        authority: (B, S) authority scores in [0, 1]

    Example:
        >>> auth = compute_authority_flow(attn, prompt_len=50, mask=M)
        >>> hallucination_score = 1 - auth[response_tokens]
    """
    # If attention has head dimension, mean-pool over heads
    if attention_weights.dim() == 4:
        B, H, S, _ = attention_weights.shape
        attn = attention_weights.mean(dim=1)  # (B, S, S)
    else:
        B, S, _ = attention_weights.shape
        attn = attention_weights

    device = attn.device
    dtype = attn.dtype

    # Initialize authority: prompt tokens = 1.0, generated = 0.0
    if previous_authority is None:
        authority = torch.zeros(B, S, device=device, dtype=dtype)
        authority[:, :prompt_length] = 1.0
    else:
        authority = previous_authority.clone()

    # For each generated token, compute authority flow
    for t in range(prompt_length, S):
        # Attention to prompt tokens (recharge)
        prompt_attn = attn[:, t, :prompt_length].sum(dim=-1)  # (B,)

        # Attention to generated tokens (flow)
        if t > prompt_length:
            gen_attn = attn[:, t, prompt_length:t]  # (B, t-prompt_length)
            gen_auth = authority[:, prompt_length:t]  # (B, t-prompt_length)
            gen_flow = (gen_attn * gen_auth).sum(dim=-1)  # (B,)
        else:
            gen_flow = torch.zeros(B, device=device, dtype=dtype)

        # Combined authority: recharge + flow
        raw_authority = prompt_attn + gen_flow

        # Apply register mask if provided
        if register_mask is not None:
            raw_authority = raw_authority * register_mask[:, t]

        authority[:, t] = raw_authority

    # Apply attention mask
    if attention_mask is not None:
        authority = authority * attention_mask.float()

    # Clamp to [0, 1] for numerical stability
    authority = authority.clamp(0.0, 1.0)

    return authority


def compute_snapkv_eviction(
    attention_weights: torch.Tensor,
    authority: torch.Tensor,
    budget: int,
    observation_window: int = 32,
    kernel_size: int = 5,
    sink_token_count: int = 4,
) -> torch.Tensor:
    """
    Compute authority-weighted SnapKV eviction indices in O(N).

    Implements the v3.1 KV Cache Compression (Paper 5 + Authority):
        vote(j) = Σ_{queries in window} A_{t,j} × 𝒜(j)
        keep_indices = TopK(vote, budget)

    Key insight: Standard SnapKV uses raw attention votes, but this
    can evict semantically important tokens that happen to have low
    attention. Authority-weighting ensures we keep tokens that are
    grounded in the prompt even if not heavily attended.

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        authority: (B, S) authority scores in [0, 1]
        budget: Number of tokens to keep (excluding observation window)
        observation_window: Last N tokens to use as query window
        kernel_size: Pooling kernel for clustering adjacent positions
        sink_token_count: First N tokens always kept (attention sinks)

    Returns:
        keep_indices: (B, H, budget) or (B, budget) indices to keep

    Example:
        >>> indices = compute_snapkv_eviction(attn, authority, budget=128)
        >>> k_compressed = k_states.gather(dim=2, index=indices.unsqueeze(-1).expand(...))
    """
    # Handle head dimension
    if attention_weights.dim() == 4:
        B, H, S, _ = attention_weights.shape
        has_heads = True
    else:
        B, S, _ = attention_weights.shape
        H = 1
        attention_weights = attention_weights.unsqueeze(1)
        has_heads = False

    device = attention_weights.device
    dtype = attention_weights.dtype

    # Compute voting window: last observation_window queries
    # Looking at attention to prefix (excluding observation window)
    window_start = max(0, S - observation_window)
    prefix_end = window_start

    if prefix_end <= sink_token_count:
        # Not enough prefix tokens to compress
        # Return indices for all non-window tokens
        indices = torch.arange(prefix_end, device=device).unsqueeze(0).expand(B, -1)
        if has_heads:
            indices = indices.unsqueeze(1).expand(-1, H, -1)
        return indices

    # Extract attention from window queries to prefix keys
    # (B, H, window_size, prefix_length)
    window_attn = attention_weights[:, :, window_start:, :prefix_end]

    # Expand authority to match heads: (B, S) -> (B, H, S)
    authority_expanded = authority.unsqueeze(1).expand(-1, H, -1)

    # Sum attention across queries (voting)
    # (B, H, prefix_length)
    raw_votes = window_attn.sum(dim=2)

    # Authority-weighted votes: vote(j) = Σ A_{t,j} × 𝒜(j)
    # (B, H, prefix_length)
    authority_prefix = authority_expanded[:, :, :prefix_end]
    weighted_votes = raw_votes * authority_prefix

    # Apply 1D average pooling for clustering adjacent positions
    if kernel_size > 1 and prefix_end > kernel_size:
        # (B*H, 1, prefix_length)
        votes_flat = weighted_votes.view(B * H, 1, prefix_end)
        pooled = F.avg_pool1d(
            votes_flat,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        # Trim to original length if padding added extra
        pooled = pooled[:, :, :prefix_end]
        weighted_votes = pooled.view(B, H, prefix_end)

    # Always keep sink tokens (set their votes to max)
    if sink_token_count > 0:
        max_vote = weighted_votes.max() + 1.0
        weighted_votes[:, :, :sink_token_count] = max_vote

    # Select top-k indices
    actual_budget = min(budget, prefix_end)
    _, keep_indices = weighted_votes.topk(actual_budget, dim=-1)

    # Sort indices for contiguous memory access
    keep_indices, _ = keep_indices.sort(dim=-1)

    if not has_heads:
        keep_indices = keep_indices.squeeze(1)

    return keep_indices


def compress_kv_cache(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    keep_indices: torch.Tensor,
    observation_window: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress KV cache using eviction indices.

    Args:
        key_states: (B, H, S, D) key states
        value_states: (B, H, S, D) value states
        keep_indices: (B, H, budget) indices from compute_snapkv_eviction
        observation_window: Last N tokens to preserve (not compressed)

    Returns:
        Tuple of compressed (key_states, value_states)
    """
    B, H, S, D = key_states.shape
    budget = keep_indices.size(-1)

    # Split into prefix and observation window
    window_start = max(0, S - observation_window)

    # Gather compressed prefix
    # Expand indices for gathering: (B, H, budget) -> (B, H, budget, D)
    indices_expanded = keep_indices.unsqueeze(-1).expand(-1, -1, -1, D)

    k_prefix = key_states[:, :, :window_start, :]
    v_prefix = value_states[:, :, :window_start, :]

    k_compressed = k_prefix.gather(dim=2, index=indices_expanded)
    v_compressed = v_prefix.gather(dim=2, index=indices_expanded)

    # Append observation window (uncompressed)
    k_window = key_states[:, :, window_start:, :]
    v_window = value_states[:, :, window_start:, :]

    k_final = torch.cat([k_compressed, k_window], dim=2)
    v_final = torch.cat([v_compressed, v_window], dim=2)

    return k_final, v_final


@_compile_if_available(mode="reduce-overhead")
def compute_authority_flow_vectorized(
    attention_weights: torch.Tensor,
    prompt_length: int,
    register_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vectorized Authority Flow computation (non-recursive approximation).

    For streaming inference, use compute_authority_flow with previous_authority.
    This version computes a single-pass approximation suitable for
    batch evaluation where the full sequence is available.

    Approximation:
        𝒜(t) ≈ Σ_{j ∈ Prompt} A_{t,j} + decay^(t-prompt_length) × Σ_{j ∈ Gen} A_{t,j}

    This captures the intuition that prompt attention provides grounding
    and generated-token attention decays with distance.

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        register_mask: (B, S) register mask M(t)
        attention_mask: (B, S) padding mask

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    # Mean-pool over heads if needed
    if attention_weights.dim() == 4:
        B, H, S, _ = attention_weights.shape
        attn = attention_weights.mean(dim=1)  # (B, S, S)
    else:
        B, S, _ = attention_weights.shape
        attn = attention_weights

    device = attn.device
    dtype = attn.dtype

    # Prompt attention (recharge): sum of attention to prompt tokens
    prompt_attn = attn[:, :, :prompt_length].sum(dim=-1)  # (B, S)

    # Initialize authority
    authority = torch.zeros(B, S, device=device, dtype=dtype)

    # Prompt tokens have full authority
    authority[:, :prompt_length] = 1.0

    # Generated tokens: authority = prompt_attention + decayed_gen_attention
    # Simple heuristic: authority ≈ prompt_attn for generated tokens
    authority[:, prompt_length:] = prompt_attn[:, prompt_length:]

    # Apply register mask
    if register_mask is not None:
        authority = authority * register_mask

    # Apply attention mask
    if attention_mask is not None:
        authority = authority * attention_mask.float()

    # Clamp to [0, 1]
    authority = authority.clamp(0.0, 1.0)

    return authority


# =============================================================================
# Centrality Kernel Fallback (Pure PyTorch - for non-Linux platforms)
# =============================================================================

@_compile_if_available(mode="reduce-overhead")
def centrality_kernel_fallback(
    Q: torch.Tensor,
    K: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Pure PyTorch fallback for centrality_flash_fwd.

    Computes Out[b,h,i] = sum_j softmax(Q[b,h,i,:] @ K[b,h,j,:].T / sqrt(D)) * v[b,j]
    with causal masking (j <= i).

    This is functionally identical to the Triton kernel but runs on CPU or
    any GPU platform (including macOS MPS).

    Args:
        Q: (B, H, S, D) Query vectors
        K: (B, H, S, D) Key vectors
        v: (B, S) Value signal (centrality input)

    Returns:
        out: (B, H, S) Attention-weighted centrality per head
    """
    B, H, S, D = Q.shape
    device = Q.device

    # Compute attention scores: (B, H, S, S)
    scale = 1.0 / (D ** 0.5)
    attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

    # Apply causal mask: j <= i
    # Use large negative (-1e9) instead of -inf to avoid NaN in softmax edge cases
    causal_mask = torch.triu(
        torch.ones(S, S, device=device, dtype=torch.bool),
        diagonal=1
    )
    attn_scores = attn_scores.masked_fill(causal_mask, -1e9)

    # Softmax over keys
    attn_probs = F.softmax(attn_scores, dim=-1)

    # Expand v for broadcasting: (B, S) -> (B, 1, 1, S)
    v_expanded = v.unsqueeze(1).unsqueeze(2)

    # Weighted sum: (B, H, S, S) * (B, 1, 1, S) -> sum over S -> (B, H, S)
    out = (attn_probs * v_expanded).sum(dim=-1)

    return out


# Alias for backward compatibility
centrality_kernel = centrality_kernel_fallback
