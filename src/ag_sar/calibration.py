"""
Self-Calibrating AG-SAR (SC-AGSAR) v10.0.

Replaces hardcoded task-specific parameters with mathematically-derived
calibration from internal model dynamics.

Key Insight: The model's attention patterns, authority variance, and
MLP divergence contain information about how to weight and aggregate
uncertainty signals - no task labels needed.

Components:
1. Entropy-Adaptive Dispersion: k_effective derived from attention entropy
2. Variance-Adaptive Aggregation: percentile interpolation from authority variance
3. Confidence-Modulated Temperature: calibration from confidence-entropy gap
4. Online Statistics: Welford's algorithm for streaming normalization
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import math
import torch
import torch.nn.functional as F


@dataclass
class OnlineStats:
    """
    Welford's algorithm for streaming mean/variance computation.

    Numerically stable single-pass algorithm that maintains running
    statistics without storing all values.

    Reference: Welford (1962), "Note on a Method for Calculating
    Corrected Sums of Squares and Products"
    """

    count: int = 0
    mean: float = 0.0
    M2: float = 0.0  # Sum of squared deviations

    def update(self, x: float) -> Tuple[float, float]:
        """
        Update statistics with new value.

        Args:
            x: New observation

        Returns:
            (current_mean, current_variance)
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        var = self.M2 / self.count if self.count > 1 else 1.0
        return self.mean, var

    @property
    def variance(self) -> float:
        return self.M2 / self.count if self.count > 1 else 1.0

    @property
    def std(self) -> float:
        return self.variance ** 0.5

    def z_score(self, x: float) -> float:
        """Compute z-score for value x."""
        if self.count < 2:
            return 0.0
        return (x - self.mean) / (self.std + 1e-8)


class SelfCalibrator:
    """
    Self-Calibrating mechanism for AG-SAR.

    Tracks running statistics and derives calibration parameters from
    internal model signals rather than hardcoded presets.

    Mathematical Principles:
    1. Entropy-Adaptive k: More tokens needed when attention is diffuse
    2. Variance-Adaptive Aggregation: Conservative when scores vary wildly
    3. Temperature from Confidence-Entropy Gap: Miscalibration correction
    """

    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 15,
        warmup_samples: int = 10,
        aggregation_gamma: float = 2.0,
    ):
        """
        Initialize self-calibrator.

        Args:
            k_min: Minimum dispersion k (for focused attention)
            k_max: Maximum dispersion k (for diffuse attention)
            warmup_samples: Number of samples before using adaptive params
            aggregation_gamma: Sensitivity for aggregation interpolation
        """
        self.k_min = k_min
        self.k_max = k_max
        self.warmup_samples = warmup_samples
        self.aggregation_gamma = aggregation_gamma

        # Online statistics trackers
        self.entropy_stats = OnlineStats()
        self.authority_var_stats = OnlineStats()
        self.divergence_stats = OnlineStats()
        self.confidence_stats = OnlineStats()

    def reset(self):
        """Reset all statistics (e.g., for new evaluation run)."""
        self.entropy_stats = OnlineStats()
        self.authority_var_stats = OnlineStats()
        self.divergence_stats = OnlineStats()
        self.confidence_stats = OnlineStats()

    @property
    def is_warmed_up(self) -> bool:
        """Check if enough samples seen for reliable calibration."""
        return self.entropy_stats.count >= self.warmup_samples

    def compute_attention_entropy(
        self,
        attention_weights: torch.Tensor,
        response_start: int,
    ) -> torch.Tensor:
        """
        Compute attention entropy for response tokens.

        H(t) = -Σⱼ A[t,j] log A[t,j]

        High entropy = diffuse attention = need more tokens for dispersion
        Low entropy = focused attention = fewer tokens suffice

        Args:
            attention_weights: (B, H, S, S) or (B, S, S) attention
            response_start: Index where response begins

        Returns:
            entropy: (B, S) per-token entropy, normalized to [0, 1]
        """
        # Handle head dimension
        if attention_weights.dim() == 4:
            attn = attention_weights.mean(dim=1)  # (B, S, S)
        else:
            attn = attention_weights

        # Entropy per token
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # (B, S)

        # Normalize by maximum possible entropy
        seq_len = attn.size(-1)
        max_entropy = math.log(seq_len)
        entropy_normalized = entropy / max_entropy  # [0, 1]

        return entropy_normalized

    def compute_adaptive_k(
        self,
        attention_weights: torch.Tensor,
        response_start: int,
    ) -> int:
        """
        Compute entropy-adaptive dispersion k.

        k_effective = k_min + (k_max - k_min) × mean(H_normalized)

        Args:
            attention_weights: Attention weights tensor
            response_start: Response start index

        Returns:
            k_effective: Adaptive k value (integer)
        """
        entropy = self.compute_attention_entropy(attention_weights, response_start)

        # Mean entropy over response tokens
        response_entropy = entropy[:, response_start:].mean().item()

        # Update statistics
        self.entropy_stats.update(response_entropy)

        # Compute adaptive k
        k_float = self.k_min + (self.k_max - self.k_min) * response_entropy
        k_effective = int(torch.clamp(
            torch.tensor(k_float),
            self.k_min,
            self.k_max
        ).item())

        return k_effective

    def compute_adaptive_aggregation(
        self,
        authority_scores: torch.Tensor,
        response_start: int,
    ) -> Tuple[float, str]:
        """
        Compute variance-adaptive aggregation.

        When authority variance is typical, use mean.
        When variance is high (indicating uncertainty), use conservative.

        α = sigmoid(γ × (σ_normalized - 1))
        score = α × percentile_10 + (1 - α) × mean

        Args:
            authority_scores: (B, S) authority per token
            response_start: Response start index

        Returns:
            aggregated_score: Float score after adaptive aggregation
            method_used: String describing interpolation ("mean", "conservative", "mixed")
        """
        response_auth = authority_scores[:, response_start:]

        # Compute variance
        auth_var = response_auth.var().item()

        # Update statistics
        self.authority_var_stats.update(auth_var)

        # Compute scores
        auth_mean = response_auth.mean().item()
        auth_p10 = torch.quantile(response_auth.flatten(), 0.10).item()

        # If not warmed up, use mean (safe default)
        if not self.is_warmed_up:
            return auth_mean, "mean (warmup)"

        # Variance-adaptive interpolation
        # Normalized variance relative to running mean
        var_normalized = auth_var / (self.authority_var_stats.mean + 1e-8)

        # Sigmoid interpolation: high variance → conservative
        # gamma controls sensitivity (higher = sharper transition)
        alpha = torch.sigmoid(
            torch.tensor(self.aggregation_gamma * (var_normalized - 1.0))
        ).item()

        # Interpolate
        aggregated = alpha * auth_p10 + (1 - alpha) * auth_mean

        # Determine method description
        if alpha < 0.2:
            method = "mean"
        elif alpha > 0.8:
            method = "conservative"
        else:
            method = f"mixed (α={alpha:.2f})"

        return aggregated, method

    def compute_adaptive_temperature(
        self,
        confidence: torch.Tensor,
        logits: torch.Tensor,
        response_start: int,
        base_temperature: float = 1.0,
    ) -> float:
        """
        Compute confidence-modulated temperature.

        Temperature adjusts calibration based on confidence-entropy gap:
        - If confidence >> uncertainty → overconfident → T > 1 (soften)
        - If confidence << uncertainty → underconfident → T < 1 (sharpen)

        T_effective = T_base × (1 + β × (confidence - entropy_ratio))

        Args:
            confidence: (B, S) softmax confidence per token
            logits: (B, S, V) output logits
            response_start: Response start index
            base_temperature: Base temperature to modulate

        Returns:
            T_effective: Adaptive temperature
        """
        response_conf = confidence[:, response_start:]
        response_logits = logits[:, response_start:, :]

        # Mean confidence
        conf_mean = response_conf.mean().item()

        # Logit entropy (uncertainty measure)
        probs = F.softmax(response_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        vocab_size = logits.size(-1)
        max_entropy = math.log(vocab_size)
        entropy_ratio = (entropy / max_entropy).mean().item()

        # Update statistics
        self.confidence_stats.update(conf_mean)

        # Temperature modulation
        # β controls sensitivity
        beta = 0.5
        gap = conf_mean - entropy_ratio

        T_effective = base_temperature * (1.0 + beta * gap)

        # Clamp to reasonable range
        T_effective = max(0.5, min(3.0, T_effective))

        return T_effective

    def compute_self_calibrating_score(
        self,
        authority_scores: torch.Tensor,
        attention_weights: torch.Tensor,
        h_attn: torch.Tensor,
        h_block: torch.Tensor,
        logits: torch.Tensor,
        embed_matrix: torch.Tensor,
        response_start: int,
        attention_mask: Optional[torch.Tensor] = None,
        base_parametric_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Compute fully self-calibrating uncertainty score.

        All parameters derived from internal signals - no hardcoded presets.

        Args:
            authority_scores: (B, S) pre-computed authority flow
            attention_weights: (B, H, S, S) attention for entropy
            h_attn: (B, S, D) pre-MLP hidden states
            h_block: (B, S, D) post-MLP hidden states
            logits: (B, S, V) output logits
            embed_matrix: (V, D) output embeddings
            response_start: Response start index
            attention_mask: Optional padding mask
            base_parametric_weight: Base weight for parametric trust

        Returns:
            Dict with:
                - score: Final uncertainty score
                - authority_aggregated: Aggregated authority
                - k_effective: Adaptive dispersion k
                - temperature: Adaptive temperature
                - aggregation_method: Aggregation description
                - calibration_stats: Current statistics
        """
        from .measures.semantics import compute_semantic_dispersion
        from .ops import compute_stability_gate

        # === 1. Entropy-Adaptive Dispersion K ===
        k_effective = self.compute_adaptive_k(attention_weights, response_start)

        # === 2. Compute Semantic Dispersion with Adaptive K ===
        dispersion = compute_semantic_dispersion(
            logits, embed_matrix, k=k_effective
        )
        response_dispersion = dispersion[:, response_start:].mean().item()

        # === 3. Variance-Adaptive Aggregation ===
        authority_aggregated, agg_method = self.compute_adaptive_aggregation(
            authority_scores, response_start
        )

        # === 4. Compute MLP Divergence ===
        cos_sim = F.cosine_similarity(h_attn, h_block, dim=-1)
        divergence = 1.0 - cos_sim
        response_divergence = divergence[:, response_start:].mean().item()
        self.divergence_stats.update(response_divergence)

        # === 5. Stability-Gated Parametric Weight ===
        # High divergence → MLP is overriding → trust attention less
        w_parametric = base_parametric_weight + (1 - base_parametric_weight) * response_divergence
        w_parametric = min(0.9, w_parametric)  # Cap at 0.9

        # === 6. Compute Confidence ===
        probs = F.softmax(logits[:, response_start:, :], dim=-1)
        confidence = probs.max(dim=-1).values
        conf_mean = confidence.mean().item()

        # === 7. Compute Adaptive Temperature ===
        T_effective = self.compute_adaptive_temperature(
            probs.max(dim=-1).values,
            logits,
            response_start
        )

        # === 8. Compute Raw Score ===
        # Fuse authority with dispersion-penalized confidence
        dispersion_penalty = 1.0 + response_dispersion  # Higher dispersion = lower trust
        adjusted_confidence = conf_mean / dispersion_penalty

        # Gate-weighted fusion
        stability_gate = math.exp(-response_divergence)
        raw_authority = (
            stability_gate * authority_aggregated +
            (1 - stability_gate) * adjusted_confidence * w_parametric
        )

        # Uncertainty = 1 - authority
        raw_score = 1.0 - raw_authority

        # === 9. Apply Adaptive Temperature ===
        # Temperature scaling via logit transform
        eps = 1e-7
        raw_score_clamped = max(eps, min(1 - eps, raw_score))
        logit_score = math.log(raw_score_clamped / (1 - raw_score_clamped))
        calibrated_logit = logit_score / T_effective
        final_score = 1 / (1 + math.exp(-calibrated_logit))

        return {
            "score": final_score,
            "authority_aggregated": authority_aggregated,
            "k_effective": k_effective,
            "temperature": T_effective,
            "aggregation_method": agg_method,
            "dispersion": response_dispersion,
            "divergence": response_divergence,
            "stability_gate": stability_gate,
            "parametric_weight": w_parametric,
            "calibration_stats": {
                "entropy_mean": self.entropy_stats.mean,
                "authority_var_mean": self.authority_var_stats.mean,
                "divergence_mean": self.divergence_stats.mean,
                "samples_seen": self.entropy_stats.count,
                "is_warmed_up": self.is_warmed_up,
            },
        }

    def compute_full_self_calibrating_score(
        self,
        authority_per_token: torch.Tensor,
        attention_weights: torch.Tensor,
        h_attn: torch.Tensor,
        h_block: torch.Tensor,
        logits: torch.Tensor,
        embed_matrix: torch.Tensor,
        response_start: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Compute FULL self-calibrating uncertainty score.

        Replicates v8.0 gated authority computation but with adaptive parameters:
        1. Entropy-adaptive dispersion k
        2. Variance-adaptive aggregation
        3. Confidence-modulated temperature

        Master Equation (same as v8.0):
            A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × parametric_weight

        Args:
            authority_per_token: (B, S) raw authority flow scores
            attention_weights: (B, H, S, S) attention weights
            h_attn: (B, S, D) pre-MLP hidden states
            h_block: (B, S, D) post-MLP hidden states
            logits: (B, S, V) output logits
            embed_matrix: (V, D) output embeddings
            response_start: Index where response begins
            attention_mask: Optional padding mask

        Returns:
            Dict with score and all calibration details
        """
        from .measures.semantics import compute_semantic_trust
        from .ops import compute_stability_gate

        device = authority_per_token.device

        # === 1. Entropy-Adaptive Dispersion K ===
        k_effective = self.compute_adaptive_k(attention_weights, response_start)

        # === 2. Compute Stability Gate (same as v8.0) ===
        # Gate = exp(-sensitivity × divergence)
        stability_sensitivity = 1.0  # v8.0 default
        if h_attn is not None and h_block is not None:
            gate = compute_stability_gate(h_attn, h_block, stability_sensitivity)
            response_gate = gate[:, response_start:]
            gate_mean = response_gate.mean().item()

            # Also compute raw divergence for logging
            cos_sim = F.cosine_similarity(
                h_attn[:, response_start:, :],
                h_block[:, response_start:, :],
                dim=-1
            )
            response_divergence = (1.0 - cos_sim).mean().item()
        else:
            gate = torch.ones_like(authority_per_token)
            response_gate = gate[:, response_start:]
            gate_mean = 1.0
            response_divergence = 0.0
        self.divergence_stats.update(response_divergence)

        # === 3. Compute Semantic Trust with Adaptive K (replaces confidence) ===
        # Trust = 1 - (Dispersion × sensitivity)
        dispersion_sensitivity = 5.0  # v8.0 default
        if embed_matrix is not None:
            trust = compute_semantic_trust(
                logits, embed_matrix, k=k_effective, sensitivity=dispersion_sensitivity
            )
            response_trust = trust[:, response_start:]
            trust_mean = response_trust.mean().item()

            # Also get raw dispersion for logging
            from .measures.semantics import compute_semantic_dispersion
            dispersion = compute_semantic_dispersion(
                logits[:, response_start:, :], embed_matrix, k=k_effective
            )
            response_dispersion = dispersion.mean().item()
        else:
            trust = torch.ones_like(authority_per_token)
            response_trust = trust[:, response_start:]
            trust_mean = 1.0
            response_dispersion = 0.0

        # === 4. Master Equation: Gated Authority (same as v8.0) ===
        # A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × weight
        parametric_weight = 0.5  # v8.0 default
        gated_authority = (
            gate * authority_per_token +
            (1.0 - gate) * trust * parametric_weight
        )

        # Apply attention mask
        if attention_mask is not None:
            gated_authority = gated_authority * attention_mask.float()

        gated_authority = gated_authority.clamp(0.0, 1.0)

        # === 5. Variance-Adaptive Aggregation ===
        response_auth = gated_authority[:, response_start:]
        auth_var = response_auth.var().item() if response_auth.numel() > 1 else 0.0
        self.authority_var_stats.update(auth_var)

        auth_mean = response_auth.mean().item()
        auth_p10 = torch.quantile(response_auth.flatten(), 0.1).item() if response_auth.numel() > 0 else auth_mean
        auth_p25 = torch.quantile(response_auth.flatten(), 0.25).item() if response_auth.numel() > 0 else auth_mean

        if self.is_warmed_up:
            # Normalized variance relative to running mean
            var_normalized = auth_var / (self.authority_var_stats.mean + 1e-8)
            # Sigmoid interpolation: high variance → conservative (use p10)
            alpha = torch.sigmoid(
                torch.tensor(self.aggregation_gamma * (var_normalized - 1.0))
            ).item()
            # Interpolate between mean and conservative (p10)
            authority_aggregated = alpha * auth_p10 + (1 - alpha) * auth_mean
            agg_method = f"adaptive (α={alpha:.2f})"
        else:
            authority_aggregated = auth_mean
            agg_method = "mean (warmup)"

        # === 6. Compute Confidence for Temperature Calibration ===
        response_logits = logits[:, response_start:, :]
        probs = F.softmax(response_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        conf_mean = confidence.mean().item()
        self.confidence_stats.update(conf_mean)

        # === 7. Compute Adaptive Temperature ===
        if self.is_warmed_up:
            # Logit entropy for uncertainty estimate
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            vocab_size = logits.size(-1)
            max_entropy = math.log(vocab_size)
            entropy_ratio = (entropy / max_entropy).mean().item()

            # Temperature from confidence-entropy gap (conservative range)
            gap = conf_mean - entropy_ratio
            T_effective = 1.0 + 0.3 * gap
            T_effective = max(0.8, min(1.5, T_effective))
        else:
            T_effective = 1.0

        # === 8. Compute Uncertainty Score ===
        # Uncertainty = 1 - aggregated gated authority (same as v8.0)
        raw_score = 1.0 - authority_aggregated
        raw_score = max(0.0, min(1.0, raw_score))

        # === 9. Apply Adaptive Temperature ===
        eps = 1e-7
        raw_score_clamped = max(eps, min(1 - eps, raw_score))
        logit_score = math.log(raw_score_clamped / (1 - raw_score_clamped))
        calibrated_logit = logit_score / T_effective
        final_score = 1 / (1 + math.exp(-calibrated_logit))

        return {
            "score": final_score,
            "raw_score": raw_score,
            "authority_aggregated": authority_aggregated,
            "authority_mean": auth_mean,
            "authority_p10": auth_p10,
            "authority_p25": auth_p25,
            "authority_var": auth_var,
            "k_effective": k_effective,
            "temperature": T_effective,
            "aggregation_method": agg_method,
            "dispersion": response_dispersion,
            "divergence": response_divergence,
            "stability_gate": gate_mean,
            "trust_mean": trust_mean,
            "parametric_weight": parametric_weight,
            "confidence_mean": conf_mean,
            "samples_seen": self.entropy_stats.count,
            "is_warmed_up": self.is_warmed_up,
        }


def apply_temperature_scaling(score: float, temperature: float) -> float:
    """
    Apply temperature scaling to calibrate uncertainty scores.

    Uses logit transformation:
        calibrated = sigmoid(logit(score) / T)

    Args:
        score: Raw uncertainty score in [0, 1]
        temperature: Calibration temperature

    Returns:
        Calibrated score in [0, 1]
    """
    if temperature == 1.0:
        return score

    eps = 1e-7
    score = max(eps, min(1 - eps, score))

    logit = math.log(score / (1 - score))
    scaled_logit = logit / temperature
    calibrated = 1 / (1 + math.exp(-scaled_logit))

    return calibrated
