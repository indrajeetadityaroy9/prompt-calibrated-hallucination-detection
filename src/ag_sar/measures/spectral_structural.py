"""
Spectral-Structural Measures for Hallucination Detection (AG-SAR v6.0).

Implements cutting-edge (2024-2025) zero-latency hallucination detection:

1. Laplacian Spectral Entropy (Horizontal - Graph Topology)
   - Based on: "Hallucination Detection Using Spectral Features of Attention Maps" (arXiv 2025)
   - Star/Clique graphs (facts) have concentrated eigenvalues
   - Chain/Random graphs (hallucinations) have diffuse eigenvalues

2. Layer-Contrastive Divergence (Vertical - Depth Dynamics)
   - Based on: "DoLa: Decoding by Contrasting Layers" (ICLR 2024)
   - Factual knowledge emerges in late layers
   - High divergence = fact retrieval; Low divergence = autocomplete/hallucination

These methods detect hallucinations without external context, making them
ideal for free generation tasks like WikiBio.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def compute_laplacian_entropy(
    attention_weights: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute Spectral Entropy of the Attention Graph Laplacian.

    The Graph Laplacian L = D - A encodes connectivity structure.
    Its eigenvalue distribution reveals the graph topology:
    - Concentrated eigenvalues → Star/Clique → Structured reasoning (Fact)
    - Diffuse eigenvalues → Chain/Random → Unstructured drift (Hallucination)

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        normalize: Whether to normalize entropy by log(S) for scale invariance
        eps: Numerical stability constant

    Returns:
        entropy: (B,) Spectral entropy per batch item
                 Lower = more structured (factual)
                 Higher = more diffuse (hallucination)
    """
    # Average over heads if needed
    if attention_weights.dim() == 4:
        A = attention_weights.mean(dim=1)  # (B, S, S)
    else:
        A = attention_weights

    B, S, _ = A.shape
    device = A.device

    # Cast to float32 for eigenvalue computation
    A = A.float()

    # Symmetrize for real eigenvalues: A_sym = 0.5 * (A + A^T)
    A_sym = 0.5 * (A + A.transpose(-1, -2))

    # Compute Degree Matrix D[i,i] = sum(A[i,:])
    degrees = A_sym.sum(dim=-1)  # (B, S)
    D = torch.diag_embed(degrees)  # (B, S, S)

    # Compute Laplacian L = D - A
    L = D - A_sym

    # Compute eigenvalues (Laplacian is symmetric positive semi-definite)
    # eigvalsh is more stable than eigvals for symmetric matrices
    try:
        eigvals = torch.linalg.eigvalsh(L)  # (B, S), real eigenvalues
    except RuntimeError:
        # Fallback for numerical issues
        return torch.zeros(B, device=device)

    # Shift eigenvalues to be positive (Laplacian has non-negative eigenvalues)
    eigvals = eigvals.clamp(min=eps)

    # Normalize to probability distribution
    probs = eigvals / eigvals.sum(dim=-1, keepdim=True)

    # Compute entropy: H = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)  # (B,)

    # Optionally normalize by maximum entropy log(S)
    if normalize:
        max_entropy = torch.log(torch.tensor(S, dtype=torch.float32, device=device))
        entropy = entropy / max_entropy

    return entropy


def compute_laplacian_entropy_per_token(
    attention_weights: torch.Tensor,
    window_size: int = 20,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute per-token Laplacian Entropy using sliding windows.

    For each token position, computes spectral entropy of the local
    attention subgraph (tokens in the causal window).

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        window_size: Size of local window for entropy computation
        normalize: Whether to normalize entropy
        eps: Numerical stability

    Returns:
        entropy: (B, S) per-token spectral entropy
    """
    if attention_weights.dim() == 4:
        A = attention_weights.mean(dim=1)
    else:
        A = attention_weights

    B, S, _ = A.shape
    device = A.device
    dtype = A.dtype

    entropy = torch.zeros(B, S, device=device, dtype=torch.float32)

    # Compute entropy for each position using its causal window
    for t in range(window_size, S):
        start = t - window_size
        # Extract local attention subgraph
        A_local = A[:, start:t, start:t].float()

        # Symmetrize
        A_sym = 0.5 * (A_local + A_local.transpose(-1, -2))

        # Degree and Laplacian
        degrees = A_sym.sum(dim=-1)
        D = torch.diag_embed(degrees)
        L = D - A_sym

        try:
            eigvals = torch.linalg.eigvalsh(L)
            eigvals = eigvals.clamp(min=eps)
            probs = eigvals / eigvals.sum(dim=-1, keepdim=True)
            ent = -torch.sum(probs * torch.log(probs + eps), dim=-1)

            if normalize:
                max_ent = torch.log(torch.tensor(window_size, dtype=torch.float32, device=device))
                ent = ent / max_ent

            entropy[:, t] = ent
        except RuntimeError:
            continue

    return entropy.to(dtype)


def compute_layer_divergence(
    logits_early: torch.Tensor,
    logits_late: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon Divergence between early and late layer outputs.

    Based on DoLa (ICLR 2024): Factual knowledge emerges in late layers.
    High divergence = late layer "corrected" early layer = Fact retrieval
    Low divergence = late layer followed early layer = Autocomplete/Hallucination

    Args:
        logits_early: (B, S, V) logits from early/premature layer
        logits_late: (B, S, V) logits from late/mature layer
        temperature: Softmax temperature for probability distributions

    Returns:
        divergence: (B, S) JS divergence per token
                    Higher = more factual (late layer correction)
                    Lower = more hallucination (autocomplete drift)
    """
    # Apply temperature scaling
    p = F.softmax(logits_early / temperature, dim=-1)
    q = F.softmax(logits_late / temperature, dim=-1)

    # Mixture distribution
    m = 0.5 * (p + q)

    # KL divergences (use log_softmax for numerical stability)
    log_p = F.log_softmax(logits_early / temperature, dim=-1)
    log_q = F.log_softmax(logits_late / temperature, dim=-1)
    log_m = torch.log(m + 1e-10)

    # JS = 0.5 * (KL(p||m) + KL(q||m))
    kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
    kl_qm = torch.sum(q * (log_q - log_m), dim=-1)

    js_div = 0.5 * (kl_pm + kl_qm)

    return js_div


def compute_layer_divergence_from_hidden(
    hidden_early: torch.Tensor,
    hidden_late: torch.Tensor,
    lm_head: torch.nn.Module,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute layer divergence from hidden states using the LM head.

    Convenience function that projects hidden states to logits before
    computing divergence.

    Args:
        hidden_early: (B, S, D) hidden states from early layer
        hidden_late: (B, S, D) hidden states from late layer
        lm_head: The language model's output projection layer
        temperature: Softmax temperature

    Returns:
        divergence: (B, S) JS divergence per token
    """
    # Project to vocabulary space
    with torch.no_grad():
        logits_early = lm_head(hidden_early)
        logits_late = lm_head(hidden_late)

    return compute_layer_divergence(logits_early, logits_late, temperature)


def compute_spectral_score(
    attention_weights: torch.Tensor,
    layer_divergence: Optional[torch.Tensor] = None,
    response_start: int = 0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Compute combined Spectral-Structural hallucination score.

    Combines Laplacian Entropy (graph structure) with Layer Divergence (depth).

    Score = alpha * LaplacianEntropy - beta * LayerDivergence
    Higher score = more likely hallucination

    Args:
        attention_weights: (B, H, S, S) attention weights
        layer_divergence: (B, S) optional layer divergence scores
        response_start: Index where response begins (for scoring)
        alpha: Weight for Laplacian entropy (higher entropy = bad)
        beta: Weight for layer divergence (higher divergence = good)

    Returns:
        score: (B,) hallucination score per batch item
    """
    # Compute Laplacian entropy
    laplacian_entropy = compute_laplacian_entropy_per_token(attention_weights)

    # Average over response tokens
    response_entropy = laplacian_entropy[:, response_start:].mean(dim=-1)

    # Combine with layer divergence if available
    if layer_divergence is not None:
        response_divergence = layer_divergence[:, response_start:].mean(dim=-1)
        score = alpha * response_entropy - beta * response_divergence
    else:
        score = alpha * response_entropy

    return score


def compute_attention_structure_score(
    attention_weights: torch.Tensor,
    response_start: int = 0,
) -> torch.Tensor:
    """
    Compute attention structure score using multiple graph metrics.

    Combines:
    1. Laplacian Spectral Entropy (overall graph structure)
    2. Attention Concentration (how focused the attention is)
    3. Subject Connectivity (attention to early tokens)

    Args:
        attention_weights: (B, H, S, S) attention weights
        response_start: Index where response begins

    Returns:
        score: (B, S) structure score per token
               Higher = more structured/factual
               Lower = more diffuse/hallucination
    """
    if attention_weights.dim() == 4:
        A = attention_weights.mean(dim=1)
    else:
        A = attention_weights

    B, S, _ = A.shape
    device = A.device
    dtype = A.dtype

    # 1. Attention concentration (entropy of attention distribution)
    # Low entropy = focused attention = structured
    attn_entropy = -torch.sum(A * torch.log(A + 1e-10), dim=-1)  # (B, S)
    max_entropy = torch.log(torch.tensor(S, dtype=torch.float32, device=device))
    concentration = 1.0 - (attn_entropy / max_entropy)  # Higher = more focused

    # 2. Subject connectivity (attention to first N tokens)
    subject_end = min(response_start + 5, S)
    subject_attn = A[:, :, :subject_end].sum(dim=-1)  # (B, S)

    # 3. Combine into structure score
    # Higher = more structured/factual
    structure_score = 0.5 * concentration + 0.5 * subject_attn

    return structure_score.to(dtype)
