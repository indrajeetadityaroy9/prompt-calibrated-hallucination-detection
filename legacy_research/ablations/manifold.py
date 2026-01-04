"""
Manifold Geometry Measures for Hallucination Detection.

Implements Local Intrinsic Dimension (LID) estimation based on:
- "Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension" (arXiv, 2024)
- Levina & Bickel MLE estimator for intrinsic dimension

Key Insight: Hallucinations traverse high-dimensional, disordered regions of latent space.
Factual generations follow low-dimensional, well-worn manifolds.

v5.1 Enhancement: Prompt-Anchored Calibration
- The user prompt defines the "True Manifold" baseline
- Generated tokens are compared to prompt's geometric complexity
- Zero-shot: No hardcoded statistics, fully dataset-agnostic

LID(hallucination) >> LID(factual)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class ManifoldSignature:
    """
    Stores the geometric signature of a prompt for adaptive calibration.

    Computed once during prefill, used to normalize generated token LID.
    """
    mean: torch.Tensor  # Mean LID of prompt tokens
    std: torch.Tensor   # Std of prompt token LIDs
    count: int          # Number of samples used for estimation

    def is_valid(self) -> bool:
        """Check if signature has enough samples for reliable calibration."""
        return self.count >= 5 and self.std.item() > 1e-6


def compute_lid(
    hidden_states: torch.Tensor,
    k: int = 20,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute Local Intrinsic Dimension (LID) using MLE estimator.

    For each token position, measures how "complex" its local neighborhood is
    relative to the preceding context. High LID = disordered manifold = confabulation.

    Uses Levina & Bickel (2004) MLE estimator:
        LID = -1 / mean(log(r_i / r_k))
    where r_i are distances to k-nearest neighbors and r_k is the k-th neighbor distance.

    Args:
        hidden_states: (B, S, D) hidden states from transformer layer
        k: Number of nearest neighbors for LID estimation
        eps: Numerical stability constant

    Returns:
        lid: (B, S) Local Intrinsic Dimension per token position
             Higher values indicate potential hallucination
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Initialize LID scores
    lid = torch.zeros(B, S, device=device, dtype=dtype)

    # For each position, compute LID relative to preceding context
    # Start from position k (need enough history for k neighbors)
    for t in range(k, S):
        # Current token: (B, 1, D)
        current = hidden_states[:, t:t+1, :]

        # History tokens: (B, t, D)
        history = hidden_states[:, :t, :]

        # Compute pairwise Euclidean distances: (B, 1, t)
        dists = torch.cdist(current, history, p=2).squeeze(1)  # (B, t)

        # Get k-nearest neighbors
        actual_k = min(k, t)
        topk_dists, _ = torch.topk(dists, k=actual_k, largest=False, dim=-1)  # (B, k)

        # MLE estimation of LID
        # LID = -1 / mean(log(r_i / r_max))
        r_max = topk_dists[:, -1:] + eps  # k-th neighbor distance (B, 1)
        r_i = topk_dists + eps  # All neighbor distances (B, k)

        # Log ratio: log(r_i / r_max)
        log_ratios = torch.log(r_i / r_max)

        # Mean of log ratios (excluding the last one which is log(1)=0)
        mean_log = log_ratios[:, :-1].mean(dim=-1)  # (B,)

        # LID = -1 / mean_log
        # Clamp to avoid division by zero and extreme values
        mean_log = mean_log.clamp(max=-eps)
        lid[:, t] = (-1.0 / mean_log).clamp(0, 100)

    # For positions < k, use a default value (not enough context)
    lid[:, :k] = lid[:, k:k+1].expand(-1, k) if S > k else 0

    return lid


def compute_lid_fast(
    hidden_states: torch.Tensor,
    k: int = 20,
    window: int = 100,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Fast vectorized LID computation using sliding window.

    Instead of computing against all history, uses a fixed-size window
    for O(S * k * window) complexity instead of O(S^2 * k).

    Args:
        hidden_states: (B, S, D) hidden states
        k: Number of nearest neighbors
        window: Size of context window (how far back to look)
        eps: Numerical stability

    Returns:
        lid: (B, S) LID scores per position
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device
    original_dtype = hidden_states.dtype

    # Cast to float32 for cdist (doesn't support bfloat16)
    hidden_states = hidden_states.float()

    # Initialize LID scores
    lid = torch.zeros(B, S, device=device, dtype=torch.float32)

    # Ensure k doesn't exceed window
    k = min(k, window - 1)

    for t in range(k, S):
        # Window start (look back up to 'window' tokens)
        start = max(0, t - window)

        # Current token
        current = hidden_states[:, t:t+1, :]  # (B, 1, D)

        # Context window (excluding current)
        context = hidden_states[:, start:t, :]  # (B, window_size, D)

        # Distances (Euclidean)
        dists = torch.cdist(current, context, p=2).squeeze(1)  # (B, window_size)

        # k-NN
        actual_k = min(k, dists.shape[-1])
        if actual_k < 2:
            continue

        topk_dists, _ = torch.topk(dists, k=actual_k, largest=False, dim=-1)

        # MLE LID estimation
        r_max = topk_dists[:, -1:] + eps
        r_i = topk_dists + eps
        log_ratios = torch.log(r_i / r_max)
        mean_log = log_ratios[:, :-1].mean(dim=-1).clamp(max=-eps)
        lid[:, t] = (-1.0 / mean_log).clamp(0, 100)

    # Cast back to original dtype
    return lid.to(original_dtype)


def compute_lid_penalty(
    hidden_states: torch.Tensor,
    k: int = 20,
    lid_mean: float = 15.0,
    lid_std: float = 10.0,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute normalized LID penalty for authority adjustment.

    Transforms raw LID to a [0, 1] penalty using sigmoid normalization.
    High LID (complex manifold) -> high penalty -> low authority.

    Args:
        hidden_states: (B, S, D) hidden states
        k: Number of nearest neighbors for LID
        lid_mean: Expected mean LID for z-score normalization
        lid_std: Expected std of LID for z-score normalization
        eps: Numerical stability

    Returns:
        penalty: (B, S) LID penalty in [0, 1]
                 0 = low LID (factual), 1 = high LID (hallucination)
    """
    lid = compute_lid_fast(hidden_states, k=k, eps=eps)

    # Z-score normalization
    z_score = (lid - lid_mean) / lid_std

    # Sigmoid to [0, 1]
    penalty = torch.sigmoid(z_score)

    return penalty


def compute_curvature_proxy(
    attention_weights: torch.Tensor,
    subject_start: int = 0,
    subject_end: int = 5,
) -> torch.Tensor:
    """
    Compute Ollivier-Ricci curvature proxy using attention profile similarity.

    Approximates graph curvature by measuring how similar the attention
    patterns are between subject tokens and generated tokens.

    Positive curvature (similar patterns) = Community = Facts
    Negative curvature (dissimilar patterns) = Bridge = Hallucinations

    Args:
        attention_weights: (B, H, S, S) or (B, S, S) attention weights
        subject_start: Start index of subject tokens
        subject_end: End index of subject tokens

    Returns:
        curvature: (B, S) curvature proxy per token
                   High = connected to subject (factual)
                   Low = disconnected (hallucination)
    """
    # Mean-pool over heads if needed
    if attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=1)  # (B, S, S)
    else:
        attn = attention_weights

    B, S, _ = attn.shape
    device = attn.device
    dtype = attn.dtype

    curvature = torch.zeros(B, S, device=device, dtype=dtype)

    # Subject attention profile (average over subject tokens)
    subject_profile = attn[:, subject_start:subject_end, :].mean(dim=1)  # (B, S)

    # For each generated token, compute similarity to subject profile
    for t in range(subject_end, S):
        token_profile = attn[:, t, :]  # (B, S)

        # Cosine similarity of attention distributions
        similarity = F.cosine_similarity(token_profile, subject_profile, dim=-1)

        curvature[:, t] = similarity

    # Subject tokens have high curvature by definition
    curvature[:, subject_start:subject_end] = 1.0

    return curvature


# =============================================================================
# v5.1: Prompt-Anchored Adaptive Calibration (Zero-Shot)
# =============================================================================

def compute_manifold_signature(
    hidden_states: torch.Tensor,
    k: int = 10,
    stride: int = 3,
    eps: float = 1e-10,
) -> ManifoldSignature:
    """
    Compute the geometric signature of a prompt for adaptive LID calibration.

    The prompt's LID distribution defines the "True Manifold" baseline.
    Generated tokens are compared to this baseline, making the system
    fully zero-shot and dataset-agnostic.

    Args:
        hidden_states: (B, S, D) hidden states of prompt tokens
        k: Number of nearest neighbors for LID estimation
        stride: Step size for sampling (reduces compute on long prompts)
        eps: Numerical stability

    Returns:
        ManifoldSignature containing mean, std, and sample count
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device

    # Need enough tokens for meaningful LID estimation
    min_tokens = k + 5
    if S < min_tokens:
        # Fallback for very short prompts: return default signature
        return ManifoldSignature(
            mean=torch.tensor(5.0, device=device),
            std=torch.tensor(2.0, device=device),
            count=0,
        )

    # Cast to float32 for cdist compatibility
    hidden_states = hidden_states.float()

    # Compute LID at sampled positions throughout the prompt
    lids = []
    for t in range(k, S, stride):
        # Current token
        current = hidden_states[:, t:t+1, :]  # (B, 1, D)

        # Context window
        start = max(0, t - 50)  # Use up to 50 tokens of history
        context = hidden_states[:, start:t, :]  # (B, context_len, D)

        # Euclidean distances
        dists = torch.cdist(current, context, p=2).squeeze(1)  # (B, context_len)

        # k-NN
        actual_k = min(k, dists.shape[-1])
        if actual_k < 2:
            continue

        topk_dists, _ = torch.topk(dists, k=actual_k, largest=False, dim=-1)

        # MLE LID estimation
        r_max = topk_dists[:, -1:] + eps
        r_i = topk_dists + eps
        log_ratios = torch.log(r_i / r_max)
        mean_log = log_ratios[:, :-1].mean(dim=-1).clamp(max=-eps)
        lid_val = (-1.0 / mean_log).clamp(0, 50)  # Clamp to reasonable range

        lids.append(lid_val.mean())  # Average across batch

    if len(lids) < 3:
        # Not enough samples for reliable statistics
        return ManifoldSignature(
            mean=torch.tensor(5.0, device=device),
            std=torch.tensor(2.0, device=device),
            count=len(lids),
        )

    lids_tensor = torch.stack(lids)

    return ManifoldSignature(
        mean=lids_tensor.mean(),
        std=lids_tensor.std().clamp(min=0.5),  # Ensure minimum variance
        count=len(lids),
    )


def compute_adaptive_lid_penalty(
    current_lid: torch.Tensor,
    signature: ManifoldSignature,
    sensitivity: float = 1.0,
) -> torch.Tensor:
    """
    Compute LID penalty based on deviation from the prompt's manifold.

    Zero-shot and dataset-agnostic: compares generated tokens to the
    prompt's own geometric complexity, not hardcoded statistics.

    Args:
        current_lid: (B, S) LID values for current tokens
        signature: ManifoldSignature from compute_manifold_signature()
        sensitivity: Scaling factor for penalty (higher = more aggressive)

    Returns:
        penalty: (B, S) penalty in [0, 1]
                 0 = matches prompt manifold (grounded)
                 1 = high deviation (likely hallucination)
    """
    if not signature.is_valid():
        # Fallback: return zero penalty if signature is unreliable
        return torch.zeros_like(current_lid)

    # Z-score relative to prompt's manifold
    z_score = (current_lid - signature.mean) / signature.std

    # Only penalize INCREASED complexity (Z > 0)
    # Simplification (Z < 0) is usually grounded, not hallucination
    z_clipped = torch.relu(z_score)

    # Smooth saturation: tanh maps [0, inf) -> [0, 1)
    # sensitivity controls how quickly we reach full penalty
    penalty = torch.tanh(sensitivity * z_clipped)

    return penalty


def compute_lid_with_calibration(
    hidden_states: torch.Tensor,
    prompt_length: int,
    k: int = 10,
    sensitivity: float = 1.0,
    eps: float = 1e-10,
) -> Tuple[torch.Tensor, ManifoldSignature]:
    """
    Compute LID penalty with automatic prompt-anchored calibration.

    Convenience function that combines signature computation and
    adaptive penalty in a single call.

    For short prompts (< k+5 tokens), uses early response tokens as
    the baseline, since they typically establish the subject and are
    more likely to be grounded.

    Args:
        hidden_states: (B, S, D) full sequence hidden states
        prompt_length: Number of prompt tokens (for signature computation)
        k: Number of nearest neighbors
        sensitivity: Penalty sensitivity
        eps: Numerical stability

    Returns:
        Tuple of:
        - penalty: (B, S) LID penalty for all tokens
        - signature: ManifoldSignature used for calibration
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device

    # Minimum tokens needed for signature
    min_tokens = k + 5

    # Determine calibration region
    # For short prompts, extend into early response (subject tokens)
    if prompt_length >= min_tokens:
        # Sufficient prompt: use prompt only
        calib_end = prompt_length
    else:
        # Short prompt: use prompt + early response (up to min_tokens)
        # This covers the "subject" which is typically grounded
        calib_end = min(min_tokens + 5, S)

    # Compute manifold signature from calibration region
    calib_states = hidden_states[:, :calib_end, :]
    signature = compute_manifold_signature(calib_states, k=k, stride=2, eps=eps)

    # Compute LID for all tokens
    lid = compute_lid_fast(hidden_states, k=k, window=50, eps=eps)

    # Compute adaptive penalty
    penalty = compute_adaptive_lid_penalty(lid, signature, sensitivity)

    # Zero out penalty for calibration region (they define the baseline)
    penalty[:, :calib_end] = 0.0

    return penalty, signature
