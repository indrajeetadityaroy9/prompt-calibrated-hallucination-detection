"""
Candidate-set JSD signal for MLP-induced distribution shift.

Computes JSD between pre-MLP and post-MLP token distributions,
restricted to a candidate set for efficiency.

IMPORTANT: Uses model's learned final LayerNorm (with gain weights γ) for
proper Logit Lens projection. The lm_head expects normalized features that
match the training distribution - using parameter-free RMS norm produces
unscaled features that degrade signal quality.

Reference: "Logit Lens" technique requires matching the model's normalization.
"""

from typing import Dict, List, Optional
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..hooks import LayerHiddenStates
from ..numerics import safe_softmax, safe_jsd
from ..ops.triton_kernels import fused_rmsnorm_linear_subset


class CandidateJSDSignal:
    """
    MLP-induced distribution shift computed on CANDIDATE SET (approximation).

    jsd_cand = JSD(p_pre_cand, p_post_cand)

    Where:
    - z_pre = W_U · final_norm(h_resid_attn)  # logits from pre-MLP residual
    - z_post = W_U · final_norm(h_resid_mlp)  # logits from post-MLP residual
    - p_*_cand = softmax(z_*[candidate_set])  # restricted to candidates

    CRITICAL: Uses model's learned final_norm (with γ weights) instead of
    parameter-free rms_norm. The lm_head was trained with final_norm applied,
    so intermediate projections must use the same normalization for valid
    logit space comparisons.

    This is an approximation to full-vocab JSD, trading accuracy for efficiency.
    Approximation quality is validated by topk_mass sanity metric.
    """

    def __init__(self, lm_head: nn.Linear, final_norm: Optional[nn.Module] = None):
        """
        Initialize JSD signal computer.

        Args:
            lm_head: The model's language model head (unembedding matrix)
            final_norm: The model's final LayerNorm (model.model.norm for LLaMA).
                       If None, falls back to parameter-free rms_norm (not recommended).
        """
        self.lm_head = lm_head
        self.final_norm = final_norm

        if final_norm is None:
            print(
                "JSD signal initialized without final_norm. Using parameter-free "
                "rms_norm which may produce suboptimal results. Pass model.model.norm "
                "for proper Logit Lens projection."
            )

    def compute_layer_jsd(
        self,
        h_resid_attn: Tensor,
        h_resid_mlp: Tensor,
        candidate_set: Tensor,
    ) -> float:
        """
        Compute candidate-set JSD for MLP-induced shift on residual stream.

        Args:
            h_resid_attn: Pre-MLP residual hidden state [batch, hidden_dim] or [hidden_dim]
            h_resid_mlp: Post-MLP residual hidden state [batch, hidden_dim] or [hidden_dim]
            candidate_set: Indices of candidate tokens [candidate_size]

        Returns:
            JSD in bits, bounded [0, 1]
        """
        # Handle batch dimension - squeeze if present
        if h_resid_attn.dim() == 1:
            h_resid_attn = h_resid_attn.unsqueeze(0)
            h_resid_mlp = h_resid_mlp.unsqueeze(0)

        # Get lm_head dtype and device for consistent computation
        lm_head_dtype = self.lm_head.weight.dtype
        lm_head_device = self.lm_head.weight.device

        # Convert to same dtype as lm_head for matrix multiplication
        h_resid_attn = h_resid_attn.to(dtype=lm_head_dtype, device=lm_head_device)
        h_resid_mlp = h_resid_mlp.to(dtype=lm_head_dtype, device=lm_head_device)

        # Use Triton Kernel if on CUDA and contiguous AND no learned final_norm
        # When final_norm is available, prefer PyTorch path for proper Logit Lens
        # (Triton kernel uses parameter-free rms_norm internally)
        use_triton = (
            h_resid_attn.is_cuda
            and candidate_set.is_cuda
            and self.final_norm is None  # Disable Triton when using learned norm
        )

        if use_triton:
            try:
                z_pre_cand = fused_rmsnorm_linear_subset(
                    h_resid_attn, self.lm_head.weight, candidate_set
                )
                z_post_cand = fused_rmsnorm_linear_subset(
                    h_resid_mlp, self.lm_head.weight, candidate_set
                )
            except Exception as e:
                print(f"JSD Triton kernel failed: {e}. Fallback to PyTorch.")
                use_triton = False
        
        if not use_triton:
            # PyTorch Fallback (Optimized)
            # 1. Normalize using model's final_norm (with learned γ) if available
            if self.final_norm is not None:
                # Use model's learned normalization for proper Logit Lens
                with torch.no_grad():
                    h_pre_norm = self.final_norm(h_resid_attn)
                    h_post_norm = self.final_norm(h_resid_mlp)
            else:
                # Fallback to parameter-free RMS norm (suboptimal)
                def _rms_norm(x, eps=1e-6):
                    return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
                h_pre_norm = _rms_norm(h_resid_attn)
                h_post_norm = _rms_norm(h_resid_mlp)

            # 2. Slice weights (copy overhead)
            w_subset = self.lm_head.weight[candidate_set] # [K, Dim]

            # 3. Compute logits
            z_pre_cand = F.linear(h_pre_norm, w_subset)
            z_post_cand = F.linear(h_post_norm, w_subset)

        # Ensure float32 for softmax stability
        z_pre_cand = z_pre_cand.float()
        z_post_cand = z_post_cand.float()

        # Compute softmax over candidates
        p_pre_cand = safe_softmax(z_pre_cand, dim=-1)
        p_post_cand = safe_softmax(z_post_cand, dim=-1)

        # Compute JSD
        jsd = safe_jsd(p_pre_cand, p_post_cand)

        return jsd

    def compute_all_layers(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        candidate_set: Tensor,
    ) -> Dict[int, float]:
        """
        Compute JSD for all captured layers.

        Args:
            layer_states: Dict mapping layer_idx to LayerHiddenStates
            candidate_set: Indices of candidate tokens

        Returns:
            Dict mapping layer_idx to JSD value
        """
        results = {}
        for layer_idx, states in layer_states.items():
            jsd = self.compute_layer_jsd(
                states.h_resid_attn,
                states.h_resid_mlp,
                candidate_set,
            )
            results[layer_idx] = jsd
        return results

    def compute_aggregated(
        self,
        layer_states: Dict[int, LayerHiddenStates],
        candidate_set: Tensor,
        aggregation: str = "sum",
    ) -> float:
        """
        Compute aggregated JSD across layers.

        Args:
            layer_states: Dict mapping layer_idx to LayerHiddenStates
            candidate_set: Indices of candidate tokens
            aggregation: How to aggregate ("sum", "mean", "max")

        Returns:
            Aggregated JSD value
        """
        layer_jsds = self.compute_all_layers(layer_states, candidate_set)

        if not layer_jsds:
            return 0.0

        values = list(layer_jsds.values())

        if aggregation == "sum":
            return sum(values)
        elif aggregation == "mean":
            return sum(values) / len(values)
        elif aggregation == "max":
            return max(values)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")