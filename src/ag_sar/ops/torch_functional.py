"""
AG-SAR v8.0 Pure PyTorch Functional Operations.

Core operations for Authority Flow hallucination detection:
- Authority Flow: Prompt provenance tracking
- Stability Gate: MLP divergence-based gating
- GQA Support: Grouped Query Attention for Llama-3.1

All operations are O(N) and designed for streaming inference.

H100 Optimizations:
- torch.compile decorators on hot paths for Inductor backend
- mode="reduce-overhead" for iterative computations
"""

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


@_compile_if_available(mode="default")
def compute_stability_gate(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    sensitivity: float = 10.0,
) -> torch.Tensor:
    """
    Compute Stability Gate (Conductivity) for v7.0 Context-Dependent Gating.

    The gate measures how much the MLP layer "agreed" with attention.
    High agreement = stable conductivity = trust Authority Flow
    Low agreement = MLP override = parametric injection needed

    Gate = exp(-sensitivity × divergence)

    where divergence = 1 - cosine_similarity(h_attn, h_block)

    Physical Interpretation:
    - RAG scenario: MLP validates context → high gate → trust Flow
    - Free gen: MLP injects parametric knowledge → low gate → inject Confidence

    Args:
        h_attn: (B, S, D) attention output BEFORE MLP
        h_block: (B, S, D) block output AFTER MLP + residuals
        sensitivity: Controls sharpness of gate (default 10.0)
                     Higher = sharper transition, more binary behavior

    Returns:
        gate: (B, S) stability gate in [0, 1]
            1.0 = MLP agrees with attention (stable, trust Flow)
            0.0 = MLP strongly overrides (unstable, inject Confidence)

    Example:
        >>> gate = compute_stability_gate(h_attn, h_block, sensitivity=10.0)
        >>> # In RAG: gate ≈ 1.0 (attention already has the answer)
        >>> # In Free Gen: gate ≈ 0.0 (MLP provides parametric memory)
    """
    # 1. Normalize vectors for cosine similarity
    h_a_norm = F.normalize(h_attn, p=2, dim=-1)
    h_b_norm = F.normalize(h_block, p=2, dim=-1)

    # 2. Cosine Similarity: (B, S)
    similarity = torch.sum(h_a_norm * h_b_norm, dim=-1)

    # 3. Divergence = 1 - similarity (range [0, 2])
    divergence = 1.0 - similarity

    # 4. Exponential Gate: high divergence → low gate
    gate = torch.exp(-sensitivity * divergence)

    return gate


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


@_compile_if_available(mode="reduce-overhead")
def compute_authority_flow_vectorized(
    attention_weights: torch.Tensor,
    prompt_length: int,
    register_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    subject_boost: float = 0.0,
    subject_token_count: int = 5,
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

    Subject Anchor Enhancement (for context-free generation):
        When prompt_length is small, the first N tokens serve as the "subject anchor".
        Authority is boosted for attention to these subject tokens:
        𝒜(t) ≈ Σ_{j ∈ Subject} A_{t,j} × subject_boost + Σ_{j ∈ Other} A_{t,j}

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        prompt_length: Index where prompt ends
        register_mask: (B, S) register mask M(t) - deprecated, ignored
        attention_mask: (B, S) padding mask
        subject_boost: Multiplier for attention to subject tokens (0.0 = disabled)
        subject_token_count: Number of tokens to treat as subject

    Returns:
        authority: (B, S) authority scores in [0, 1]
    """
    # Mean pool over heads if needed
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

    # v4.0 Subject Anchor: Boost attention to subject tokens (first N response tokens)
    # This helps in WikiBio-style generation where the subject establishes grounding
    # Key insight: Even with a short prompt, the first few response tokens establish
    # the subject entity. Valid facts maintain high attention to these tokens.
    if subject_boost > 0.0 and subject_token_count > 0:
        # Subject region: first N tokens AFTER prompt (the response subject)
        subject_start = prompt_length
        subject_end = min(prompt_length + subject_token_count, S)

        if subject_end > subject_start:
            # Subject tokens themselves have high authority
            authority[:, subject_start:subject_end] = 1.0

            # For tokens after the subject region, boost attention to subject
            if subject_end < S:
                # Attention to subject tokens (boosted)
                subject_attn = attn[:, subject_end:, subject_start:subject_end].sum(dim=-1)
                # Attention to prompt (if any)
                prompt_attn_post = attn[:, subject_end:, :prompt_length].sum(dim=-1) if prompt_length > 0 else 0
                # Attention to non-subject tokens
                non_subject_attn = attn[:, subject_end:, subject_end:].sum(dim=-1)

                # Boosted authority formula:
                # auth = (prompt_attn + subject_attn * boost + non_subject_attn) / normalization
                raw_auth = prompt_attn_post + subject_attn * subject_boost + non_subject_attn

                # Normalize: max possible when all attention goes to subject
                max_possible = 1.0 + subject_boost  # Prompt can contribute 1, subject*boost
                authority[:, subject_end:] = raw_auth / max_possible

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
