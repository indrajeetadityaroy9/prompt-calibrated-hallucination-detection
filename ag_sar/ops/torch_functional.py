"""
AG-SAR Core Operations - Pure PyTorch.

Authority Flow and Agreement Gate computations with H100 optimizations.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import os

# Disable torch.compile via environment variable
_DISABLE_COMPILE = os.environ.get("AG_SAR_DISABLE_COMPILE", "0") == "1"
_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])


def _should_compile() -> bool:
    if _DISABLE_COMPILE or _TORCH_VERSION == (2, 4):
        return False
    return (
        hasattr(torch, 'compile') and
        torch.cuda.is_available() and
        torch.cuda.get_device_capability()[0] >= 8
    )


def _compile_if_available(mode: str = "default"):
    def decorator(func):
        if _should_compile():
            try:
                return torch.compile(func, mode="default", dynamic=True)
            except Exception:
                return func
        return func
    return decorator


def align_gqa_heads(v_states: torch.Tensor, n_q_heads: int) -> torch.Tensor:
    """Expand GQA KV heads to match query head count."""
    if v_states.dim() == 3:
        return v_states
    batch, n_kv_heads, seq_len, head_dim = v_states.shape
    if n_kv_heads == n_q_heads:
        return v_states
    n_rep = n_q_heads // n_kv_heads
    v_expanded = v_states.unsqueeze(2).expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    return v_expanded.reshape(batch, n_q_heads, seq_len, head_dim)


@_compile_if_available(mode="reduce-overhead")
def compute_mlp_divergence(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute MLP divergence: δ(t) = 1 - CosineSim(h_attn, h_block).

    High divergence = MLP overriding attention signal.
    """
    h_attn_norm = F.normalize(h_attn, p=2, dim=-1)
    h_block_norm = F.normalize(h_block, p=2, dim=-1)
    cos_sim = (h_attn_norm * h_block_norm).sum(dim=-1)
    divergence = 1.0 - cos_sim
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
    Compute agreement gate: Gate = exp(-sensitivity × divergence).

    High gate = MLP agrees with attention (trust context).
    Low gate = MLP overrides (use parametric memory).
    """
    h_a_norm = F.normalize(h_attn, p=2, dim=-1)
    h_b_norm = F.normalize(h_block, p=2, dim=-1)
    similarity = torch.sum(h_a_norm * h_b_norm, dim=-1)
    divergence = 1.0 - similarity
    return torch.exp(-sensitivity * divergence)


def compute_authority_flow(
    attention_weights: torch.Tensor,
    prompt_length: int,
    previous_authority: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute recursive authority flow in O(N).

    Authority(t) = Σ_{prompt} A_{t,j} + Σ_{gen} A_{t,j} × Authority(j)
    """
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=1)
    else:
        attn = attention_weights

    B, S, _ = attn.shape
    device, dtype = attn.device, attn.dtype

    if previous_authority is None:
        authority = torch.zeros(B, S, device=device, dtype=dtype)
        authority[:, :prompt_length] = 1.0
    else:
        authority = previous_authority.clone()

    for t in range(prompt_length, S):
        prompt_attn = attn[:, t, :prompt_length].sum(dim=-1)
        if t > prompt_length:
            gen_attn = attn[:, t, prompt_length:t]
            gen_auth = authority[:, prompt_length:t]
            gen_flow = (gen_attn * gen_auth).sum(dim=-1)
        else:
            gen_flow = torch.zeros(B, device=device, dtype=dtype)
        authority[:, t] = prompt_attn + gen_flow

    if attention_mask is not None:
        authority = authority * attention_mask.float()
    return authority.clamp(0.0, 1.0)


@_compile_if_available(mode="reduce-overhead")
def compute_authority_flow_vectorized(
    attention_weights: torch.Tensor,
    prompt_length: int,
    attention_mask: Optional[torch.Tensor] = None,
    subject_boost: float = 0.0,
    subject_token_count: int = 5,
) -> torch.Tensor:
    """
    Vectorized authority flow (single-pass approximation).

    Authority ≈ attention to prompt tokens.
    """
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=1)
    else:
        attn = attention_weights

    B, S, _ = attn.shape
    device, dtype = attn.device, attn.dtype

    prompt_attn = attn[:, :, :prompt_length].sum(dim=-1)
    authority = torch.zeros(B, S, device=device, dtype=dtype)
    authority[:, :prompt_length] = 1.0
    authority[:, prompt_length:] = prompt_attn[:, prompt_length:]

    if subject_boost > 0.0 and subject_token_count > 0:
        subject_start = prompt_length
        subject_end = min(prompt_length + subject_token_count, S)
        if subject_end > subject_start:
            authority[:, subject_start:subject_end] = 1.0
            if subject_end < S:
                subject_attn = attn[:, subject_end:, subject_start:subject_end].sum(dim=-1)
                prompt_attn_post = attn[:, subject_end:, :prompt_length].sum(dim=-1) if prompt_length > 0 else 0
                non_subject_attn = attn[:, subject_end:, subject_end:].sum(dim=-1)
                raw_auth = prompt_attn_post + subject_attn * subject_boost + non_subject_attn
                authority[:, subject_end:] = raw_auth / (1.0 + subject_boost)

    if attention_mask is not None:
        authority = authority * attention_mask.float()
    return authority.clamp(0.0, 1.0)


@_compile_if_available(mode="reduce-overhead")
def compute_authority_flow_streaming(
    q_curr: torch.Tensor,
    k_cache: torch.Tensor,
    previous_authority: torch.Tensor,
    prompt_length: int,
    current_position: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming authority for single token (O(N) memory).

    Returns (current_authority, updated_authority_state).
    """
    import math
    device, dtype = q_curr.device, q_curr.dtype
    scale = 1.0 / math.sqrt(head_dim)

    attn_logits = torch.matmul(q_curr, k_cache.transpose(-2, -1)) * scale
    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
    attn = attn_probs.squeeze(-2).mean(dim=1).to(dtype)

    B, S = attn.shape
    prompt_attn = attn[:, :prompt_length].sum(dim=-1)

    if current_position > prompt_length:
        gen_attn = attn[:, prompt_length:current_position]
        gen_auth = previous_authority[:, prompt_length:current_position]
        gen_flow = (gen_attn * gen_auth).sum(dim=-1)
    else:
        gen_flow = torch.zeros(B, device=device, dtype=dtype)

    current_authority = (prompt_attn + gen_flow).clamp(0.0, 1.0)
    updated_authority = torch.cat([previous_authority, current_authority.unsqueeze(-1)], dim=-1)

    return current_authority, updated_authority


def compute_structural_gain(
    epiplexity: torch.Tensor,
    lambda_struct: float = 1.0,
) -> torch.Tensor:
    """
    Compute structural gain from epiplexity.

    γ_t = 1 + λ_struct × E_t

    Structural gain amplifies authority propagation for valid reasoning chains.
    Higher epiplexity (more "structural work") leads to higher gain.

    Args:
        epiplexity: Per-token epiplexity proxy [B, S], values in [0, 1]
        lambda_struct: Gain strength coefficient (default 1.0)

    Returns:
        Structural gain [B, S], values in [1, 1 + λ_struct]
    """
    return 1.0 + lambda_struct * epiplexity


@_compile_if_available(mode="reduce-overhead")
def compute_authority_flow_recursive(
    attention_weights: torch.Tensor,
    prompt_length: int,
    structural_gain: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Pure recursive authority flow with structural gain (v3).

    A(t) = Σ_context α_tj + γ_t × Σ_gen α_tj × A(j)

    This is the v3 authority flow that:
    1. Treats prompt tokens as sources of authority (A=1.0)
    2. Propagates authority through generated tokens recursively
    3. Amplifies propagation by structural gain γ_t for valid reasoning

    CRITICAL: NO residual mixing here. The Gate (G_t) handles the trade-off
    between contextual and parametric reliance in the master equation.
    Mixing here would dilute the pure provenance signal.

    Args:
        attention_weights: Attention weights [B, H, S, S] or [B, S, S]
        prompt_length: Number of prompt tokens (authority sources)
        structural_gain: Per-token γ_t values [B, S], defaults to 1.0
        attention_mask: Optional mask for padding

    Returns:
        Authority values [B, S], clamped to [0, 1]
    """
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=1)
    else:
        attn = attention_weights

    B, S, _ = attn.shape
    device, dtype = attn.device, attn.dtype

    authority = torch.zeros(B, S, device=device, dtype=dtype)
    authority[:, :prompt_length] = 1.0

    if structural_gain is None:
        gamma = torch.ones(B, S, device=device, dtype=dtype)
    else:
        gamma = structural_gain

    for t in range(prompt_length, S):
        prompt_attn = attn[:, t, :prompt_length].sum(dim=-1)

        if t > prompt_length:
            gen_attn = attn[:, t, prompt_length:t]
            gen_auth = authority[:, prompt_length:t]
            gen_flow = (gen_attn * gen_auth).sum(dim=-1)
        else:
            gen_flow = torch.zeros(B, device=device, dtype=dtype)

        authority[:, t] = prompt_attn + gamma[:, t] * gen_flow

    if attention_mask is not None:
        authority = authority * attention_mask.float()

    return authority.clamp(0.0, 1.0)


def compute_authority_from_qk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    structural_gain: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    """
    Compute recursive authority flow from Q, K states (Python reference).

    This is the reference implementation for testing the Triton FlashAuthority kernel.
    It computes attention on-the-fly from Q, K rather than using precomputed weights.

    Args:
        query_states: [B, H_q, S, D] query vectors
        key_states: [B, H_k, S, D] key vectors (H_k may differ from H_q for GQA)
        structural_gain: [B, S] per-token γ_t values
        prompt_length: Number of prompt tokens

    Returns:
        authority: [B, S] averaged over heads
    """
    import math

    B, H_q, S, D = query_states.shape
    H_k = key_states.shape[1]
    device = query_states.device

    # GQA Expansion: For models with Grouped Query Attention (Llama-3, Mistral, Qwen),
    # K has fewer heads than Q. Expand K to match Q.
    if H_q != H_k:
        n_rep = H_q // H_k
        # [B, H_k, S, D] -> [B, H_k, n_rep, S, D] -> [B, H_q, S, D]
        key_states = key_states[:, :, None, :, :].expand(
            B, H_k, n_rep, S, D
        ).reshape(B, H_q, S, D)

    H = H_q  # After expansion, both have H_q heads
    scale = 1.0 / math.sqrt(D)

    # Output per head
    authority_per_head = torch.zeros(B, H, S, device=device, dtype=torch.float32)
    authority_per_head[:, :, :prompt_length] = 1.0

    for t in range(prompt_length, S):
        # Compute attention scores for position t: Q[t] @ K[:t+1].T
        q_t = query_states[:, :, t:t+1, :]  # [B, H, 1, D]
        k_past = key_states[:, :, :t+1, :]  # [B, H, t+1, D]

        # [B, H, 1, t+1]
        attn_scores = torch.matmul(q_t, k_past.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1).squeeze(2)  # [B, H, t+1]

        # Split into context and generation
        context_probs = attn_probs[:, :, :prompt_length]  # [B, H, prompt_len]
        flow_context = context_probs.sum(dim=-1)  # [B, H]

        if t > prompt_length:
            gen_probs = attn_probs[:, :, prompt_length:t]  # [B, H, t-prompt_len]
            gen_auth = authority_per_head[:, :, prompt_length:t]  # [B, H, t-prompt_len]
            flow_gen = (gen_probs * gen_auth).sum(dim=-1)  # [B, H]
        else:
            flow_gen = torch.zeros(B, H, device=device, dtype=torch.float32)

        # Apply structural gain (broadcast from [B, S] to [B, H])
        gamma_t = structural_gain[:, t:t+1]  # [B, 1]
        authority_per_head[:, :, t] = flow_context + gamma_t * flow_gen

    # Average over heads and clamp
    authority = authority_per_head.mean(dim=1)
    return authority.clamp(0.0, 1.0)


@_compile_if_available(mode="reduce-overhead")
def compute_logit_divergence_jsd(
    h_attn: torch.Tensor,
    h_block: torch.Tensor,
    embed_matrix: torch.Tensor,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between attention and final block logits.

    This is the core signal for v8 "Residual Stream Contrast": measuring whether
    the FFN overrode the context signal (attention output).

    JSD = (KL(P||M) + KL(Q||M)) / 2, where M = (P+Q)/2

    Args:
        h_attn: Pre-MLP hidden states [B, S, D] - "what context says"
        h_block: Post-MLP hidden states [B, S, D] - "what model outputs"
        embed_matrix: Unembedding matrix [V, D]
        top_k: Use top-k approximation for efficiency (O(k) vs O(V))

    Returns:
        JSD values [B, S] in [0, 1], where:
        - 0 = P_attn == P_final (FFN preserved context)
        - 1 = P_attn and P_final are completely different (FFN overrode context)
    """
    B, S, D = h_attn.shape
    device = h_attn.device
    dtype = h_attn.dtype

    # Project to logit space: [B, S, D] @ [D, V] -> [B, S, V]
    # This is O(S × D × V) - the critical path for performance
    logits_attn = torch.matmul(h_attn.float(), embed_matrix.T.float())
    logits_block = torch.matmul(h_block.float(), embed_matrix.T.float())

    # Top-k selection for efficiency (O(k) vs O(V))
    # Get top-k indices from the final distribution (what model actually predicts)
    _, top_indices = logits_block.topk(top_k, dim=-1)  # [B, S, k]

    # Gather top-k logits for both distributions
    logits_attn_topk = logits_attn.gather(-1, top_indices)  # [B, S, k]
    logits_block_topk = logits_block.gather(-1, top_indices)  # [B, S, k]

    # Convert to probabilities with numerical stability
    # Using log_softmax for stability, then exp
    log_p_attn = F.log_softmax(logits_attn_topk, dim=-1)
    log_p_block = F.log_softmax(logits_block_topk, dim=-1)

    p_attn = log_p_attn.exp()
    p_block = log_p_block.exp()

    # Compute midpoint distribution M = (P + Q) / 2
    p_mean = (p_attn + p_block) / 2

    # Compute log of midpoint (with epsilon for stability)
    eps = 1e-10
    log_p_mean = (p_mean + eps).log()

    # KL(P||M) = Σ P * (log P - log M)
    kl_attn = (p_attn * (log_p_attn - log_p_mean)).sum(dim=-1)
    kl_block = (p_block * (log_p_block - log_p_mean)).sum(dim=-1)

    # JSD = (KL(P||M) + KL(Q||M)) / 2
    jsd = (kl_attn + kl_block) / 2

    # JSD is bounded [0, log(2)] for binary case, but with softmax
    # over k classes, max is log(2) ≈ 0.693. Normalize to [0, 1].
    jsd_normalized = jsd / 0.693

    return jsd_normalized.clamp(0.0, 1.0).to(dtype)
