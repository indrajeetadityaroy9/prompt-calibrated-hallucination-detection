"""
Local Intrinsic Dimension (LID) for geometric hallucination detection.

v7 Innovation: Truthful text lies on low-dimensional coherent manifold.
Hallucinations "leave the manifold" with higher intrinsic dimensionality.

Key insight: "Confident Lie" has low Varentropy (scalar certainty) but HIGH LID
(geometric inconsistency). This fixes RAGTruth failures where v6's scalar
Gaussian complexity matching fails.

Research Foundation:
- Yin et al., ICML 2024: "Characterizing Truthfulness in LLM Generations with Local Intrinsic Dimension"
- Meister et al., 2023: "Revisiting Entropy Rate Constancy in Text"
"""

from typing import Optional, Dict
import torch


def compute_participation_ratio(
    eigenvalues: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    Participation Ratio - effective number of dimensions.

    PR = (Σλᵢ)² / Σλᵢ²

    Interpretation:
    - PR ≈ 1: Single dominant direction (low complexity, coherent)
    - PR ≈ k: All k directions equally important (high complexity, scattered)

    Args:
        eigenvalues: (..., K) eigenvalues (any batch dimensions)
        epsilon: Numerical stability floor

    Returns:
        participation_ratio: (...) PR values for each batch element
    """
    sum_eig = eigenvalues.sum(dim=-1) + epsilon
    sum_sq_eig = (eigenvalues ** 2).sum(dim=-1) + epsilon
    return (sum_eig ** 2) / sum_sq_eig


def _compute_lid_loop_fallback(
    hidden_states: torch.Tensor,
    window_size: int,
    normalize: bool,
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Fallback loop-based LID computation for when batched SVD fails.

    This is slower but handles edge cases like degenerate matrices.
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    lid_scores = torch.zeros(B, S, device=device, dtype=dtype)

    for t in range(S):
        start = max(0, t - window_size + 1)
        window = hidden_states[:, start:t+1, :]  # (B, W, D)

        if window.shape[1] < 2:
            # Not enough context, assume coherent (low LID)
            lid_scores[:, t] = 0.0
            continue

        # Center the window
        window_centered = window - window.mean(dim=1, keepdim=True)

        try:
            singular_values = torch.linalg.svdvals(window_centered)  # (B, min(W,D))
        except RuntimeError:
            # Degenerate case
            lid_scores[:, t] = 0.0
            continue

        eigenvalues = singular_values ** 2
        pr = compute_participation_ratio(eigenvalues)

        if normalize:
            max_lid = min(window.shape[1], D)
            lid_scores[:, t] = (pr / max_lid).clamp(0.0, 1.0)
        else:
            lid_scores[:, t] = pr

    if attention_mask is not None:
        lid_scores = lid_scores * attention_mask.float()

    return lid_scores


def compute_local_intrinsic_dimension(
    hidden_states: torch.Tensor,
    window_size: int = 8,
    normalize: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute Local Intrinsic Dimension via batched sliding window SVD.

    Uses `unfold` for efficient batched computation - O(1) kernel launches
    instead of O(S) with a naive loop.

    Args:
        hidden_states: (B, S, D) - block outputs or attention outputs
        window_size: Number of tokens in local context window (default 8)
        normalize: If True, normalize LID to [0, 1] range
        attention_mask: Optional (B, S) padding mask

    Returns:
        lid_scores: (B, S) local intrinsic dimension per token
            - Low LID (near 0) = coherent manifold = trusted
            - High LID (near 1) = scattered/hallucinated = penalized
    """
    B, S, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Handle edge case: very short sequences
    if S < 2:
        lid = torch.zeros(B, S, device=device, dtype=dtype)
        if attention_mask is not None:
            lid = lid * attention_mask.float()
        return lid

    # Effective window size (can't exceed sequence length)
    W = min(window_size, S)

    # Pad input for unfolding: prepend W-1 zeros so window[t] covers [t-W+1:t+1]
    # This ensures position 0 has a valid (smaller) window
    padding = torch.zeros(B, W - 1, D, device=device, dtype=dtype)
    padded_states = torch.cat([padding, hidden_states], dim=1)  # [B, S+W-1, D]

    # Create sliding windows: [B, S, W, D]
    # unfold(dim, size, step) - we unfold along sequence dimension
    windows = padded_states.unfold(1, W, 1)  # [B, S, D, W]
    windows = windows.permute(0, 1, 3, 2)     # [B, S, W, D]

    # Center each window (subtract mean along window dimension)
    windows_centered = windows - windows.mean(dim=2, keepdim=True)  # [B, S, W, D]

    # Reshape for batched SVD: [B*S, W, D]
    windows_flat = windows_centered.reshape(B * S, W, D)

    # Compute singular values via batched SVD
    # svdvals returns shape [B*S, min(W, D)]
    try:
        singular_values = torch.linalg.svdvals(windows_flat)
    except RuntimeError:
        # Fallback to loop-based computation for numerical issues
        return _compute_lid_loop_fallback(
            hidden_states, window_size, normalize, attention_mask
        )

    # Eigenvalues of covariance = singular values squared
    eigenvalues = singular_values ** 2  # [B*S, min(W, D)]

    # Participation Ratio for each position
    pr = compute_participation_ratio(eigenvalues)  # [B*S]
    pr = pr.reshape(B, S)  # [B, S]

    # Normalize to [0, 1] by dividing by max possible PR
    if normalize:
        max_lid = min(W, D)
        pr = (pr / max_lid).clamp(0.0, 1.0)

    # Apply attention mask (padding tokens get LID = 0)
    if attention_mask is not None:
        pr = pr * attention_mask.float()

    return pr


def compute_manifold_score(
    hidden_states: torch.Tensor,
    window_size: int = 8,
    calibration: Optional[Dict[str, float]] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Manifold adherence score: M(t) = 1 - LID_norm(t)

    Interpretation:
    - High M = low LID = coherent manifold = trusted
    - Low M = high LID = scattered/hallucinated = penalized

    Args:
        hidden_states: (B, S, D) from block_outputs or attn_outputs
        window_size: Local context window for LID computation
        calibration: Optional dict with 'lid_mu' baseline for relative scoring
        attention_mask: Optional (B, S) padding mask

    Returns:
        manifold_scores: (B, S) in [0, 1]
    """
    lid = compute_local_intrinsic_dimension(
        hidden_states,
        window_size=window_size,
        normalize=True,
        attention_mask=attention_mask,
    )

    # Optional: Calibrate relative to prompt baseline
    # If prompt has lid_mu, normalize response LID relative to it
    if calibration and 'lid_mu' in calibration:
        lid_mu = calibration['lid_mu']
        if lid_mu > 0.01:  # Only calibrate if meaningful baseline
            # Relative deviation from baseline, clamp to prevent extreme values
            lid = (lid / lid_mu).clamp(0.0, 2.0) / 2.0

    # Manifold adherence = inverse of LID
    return (1.0 - lid).clamp(0.0, 1.0)
