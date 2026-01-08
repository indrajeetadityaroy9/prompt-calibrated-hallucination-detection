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
