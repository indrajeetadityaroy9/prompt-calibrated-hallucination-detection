"""Configuration for AG-SAR uncertainty quantification pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class AGSARConfig:
    """
    Configuration for AG-SAR v3.1 uncertainty quantification pipeline.

    Implements Recursive Authority Flow for Zero-Latency Hallucination Detection.
    Optimized for NVIDIA H100 with bfloat16 precision.

    v3.1 Mechanisms:
        1. Register Filter: EMA Z-score + Sigmoid gate (Papers 1 & 2)
        2. Authority Flow: Prompt Recharge + Gen Flow (Paper 6 corrected)
        3. Spectral Roughness: Pre-MLP deviation (Paper 9 approximation)
        4. SnapKV Eviction: Authority-weighted voting (Paper 5)

    Attributes:
        semantic_layers: Number of final layers to use for semantic analysis.
        residual_weight: Weight for identity matrix in residual correction.
        power_iteration_steps: Maximum iterations for eigenvector centrality.
        hallucination_threshold: GSE threshold for hallucination detection.
        preferred_dtype: Preferred tensor dtype (bfloat16 recommended).

    Example:
        >>> config = AGSARConfig(
        ...     semantic_layers=4,
        ...     enable_register_filter=True,
        ...     lambda_roughness=10.0,
        ... )
    """

    # ==========================================================================
    # v3.1 Mechanism 1: Register Filter (Papers 1 & 2)
    # ==========================================================================
    # M(t) = (t > sink_token_count) × Sigmoid(-Z(t) + τ)
    # where Z(t) = (Kurt(v_t) - μ_EMA) / σ_EMA
    enable_register_filter: bool = True
    ema_decay: float = 0.995          # Welford EMA decay (high = stable)
    kurtosis_threshold: float = 2.0   # τ: Gate threshold for sigmoid

    # ==========================================================================
    # v3.1 Mechanism 2: Authority Flow (Paper 6 corrected)
    # ==========================================================================
    # 𝒜(t) = [Σ_Prompt A_{t,j}] + [Σ_Gen A_{t,j} × 𝒜(j)] × M(t)
    enable_authority_flow: bool = True
    recharge_weight: float = 1.0      # Weight for prompt token contribution

    # ==========================================================================
    # v3.1 Mechanism 3: Spectral Roughness (Paper 9 approximation)
    # ==========================================================================
    # δ(t) = ||h_attn(t) - Σ A_{t,j} × v_j||_2
    # 𝒜_final = 𝒜 × (1 / (1 + λ × δ))
    enable_spectral_roughness: bool = True
    lambda_roughness: float = 10.0    # Sensitivity to pre-MLP deviation

    # ==========================================================================
    # Legacy: Semantic Layer Selection
    # ==========================================================================
    semantic_layers: int = 4

    # Power iteration for centrality
    residual_weight: float = 0.5

    # Centrality computation
    power_iteration_steps: int = 3  # Uses fast unrolled version; converges in 2-3 iterations
    power_iteration_tol: float = 1e-4  # Relaxed tolerance for faster convergence

    # Hallucination detection
    hallucination_threshold: float = 0.7

    # torch.compile optimization for hot paths (entropy, GSE computation)
    # Reduces Python overhead by ~10-20% after warmup
    use_torch_compile: bool = True

    # Precision - CRITICAL: Use bfloat16 on H100, NEVER float16 for GPT-2
    preferred_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)

    # Model-specific (auto-detected if None)
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None

    # Model architecture: "auto", "gpt2", or "llama"
    # "auto" attempts to detect from model structure
    model_architecture: str = "auto"

    # GQA (Grouped Query Attention) configuration for Llama-3
    # num_kv_heads < num_attention_heads means GQA is used
    # For Llama-3-8B: num_q_heads=32, num_kv_heads=8 (4 Q-heads per KV-head)
    num_kv_heads: Optional[int] = None

    # Sink token masking (StreamingLLM-style)
    # First N tokens are structural attention sinks, not semantic
    # Llama/Mistral/Qwen use <s> as a sink - mask it out for cleaner centrality
    sink_token_count: int = 4

    # MC-SS (Manifold-Consistent Spectral Surprisal) configuration
    # Alternative uncertainty metric using Hebbian-filtered centrality
    uncertainty_metric: str = "gse"  # Options: "gse", "gss", "mcss", "authority"
    mcss_beta: float = 5.0  # Soft clamp for bounded surprisal: tanh(-log P / beta)
    mcss_hebbian_tau: float = 0.1  # ReLU threshold for Hebbian prior: ReLU(sim - tau)
    mcss_penalty_weight: float = 1.0  # λ weight for additive penalty term in MC-SS

    # Surprisal-Gated Spectral Steering (SGSS)
    # Dynamically modulates head contributions based on model confidence
    # Formula: w_{h,t} = max(0, 1 + α × (1 - tanh(S_t/β)) × Ω_h)
    # When confident (low S), steering is active; when uncertain, steering disabled
    use_spectral_steering: bool = False
    steering_alpha: float = 2.0  # Strength of steering intervention
    steering_beta: float = 5.0   # Gate sensitivity (surprisal threshold)
    head_scores_path: Optional[str] = None  # Path to Z-scores JSON

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.residual_weight <= 1.0:
            raise ValueError(
                f"residual_weight must be in [0, 1], got {self.residual_weight}"
            )
        if self.power_iteration_steps < 1:
            raise ValueError(
                f"power_iteration_steps must be >= 1, got {self.power_iteration_steps}"
            )
        if self.power_iteration_tol <= 0:
            raise ValueError(
                f"power_iteration_tol must be > 0, got {self.power_iteration_tol}"
            )
        if self.semantic_layers < 1:
            raise ValueError(
                f"semantic_layers must be >= 1, got {self.semantic_layers}"
            )

        # Warn about float16 on GPT-2
        if self.preferred_dtype == torch.float16:
            import warnings
            warnings.warn(
                "float16 may cause NaN overflow with GPT-2. "
                "Consider using bfloat16 instead.",
                UserWarning
            )

        # Validate MC-SS parameters
        valid_metrics = ("gse", "gss", "mcss", "v31", "authority")
        if self.uncertainty_metric not in valid_metrics:
            raise ValueError(
                f"uncertainty_metric must be one of {valid_metrics}, "
                f"got {self.uncertainty_metric}"
            )
        if self.mcss_beta <= 0:
            raise ValueError(f"mcss_beta must be > 0, got {self.mcss_beta}")
        if not 0.0 <= self.mcss_hebbian_tau <= 1.0:
            raise ValueError(
                f"mcss_hebbian_tau must be in [0, 1], got {self.mcss_hebbian_tau}"
            )
        if self.mcss_penalty_weight < 0:
            raise ValueError(
                f"mcss_penalty_weight must be >= 0, got {self.mcss_penalty_weight}"
            )

        # Validate SGSS parameters
        if self.steering_alpha < 0:
            raise ValueError(
                f"steering_alpha must be >= 0, got {self.steering_alpha}"
            )
        if self.steering_beta <= 0:
            raise ValueError(
                f"steering_beta must be > 0, got {self.steering_beta}"
            )

        # Validate v3.1 parameters
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(
                f"ema_decay must be in (0, 1), got {self.ema_decay}"
            )
        if self.lambda_roughness < 0:
            raise ValueError(
                f"lambda_roughness must be >= 0, got {self.lambda_roughness}"
            )
        if self.recharge_weight < 0:
            raise ValueError(
                f"recharge_weight must be >= 0, got {self.recharge_weight}"
            )

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            # v3.1 Mechanism 1: Register Filter
            'enable_register_filter': self.enable_register_filter,
            'ema_decay': self.ema_decay,
            'kurtosis_threshold': self.kurtosis_threshold,
            # v3.1 Mechanism 2: Authority Flow
            'enable_authority_flow': self.enable_authority_flow,
            'recharge_weight': self.recharge_weight,
            # v3.1 Mechanism 3: Spectral Roughness
            'enable_spectral_roughness': self.enable_spectral_roughness,
            'lambda_roughness': self.lambda_roughness,
            # Legacy parameters
            'semantic_layers': self.semantic_layers,
            'residual_weight': self.residual_weight,
            'power_iteration_steps': self.power_iteration_steps,
            'power_iteration_tol': self.power_iteration_tol,
            'hallucination_threshold': self.hallucination_threshold,
            'use_torch_compile': self.use_torch_compile,
            'preferred_dtype': str(self.preferred_dtype),
            'num_attention_heads': self.num_attention_heads,
            'num_hidden_layers': self.num_hidden_layers,
            'model_architecture': self.model_architecture,
            'num_kv_heads': self.num_kv_heads,
            'sink_token_count': self.sink_token_count,
            'uncertainty_metric': self.uncertainty_metric,
            'mcss_beta': self.mcss_beta,
            'mcss_hebbian_tau': self.mcss_hebbian_tau,
            'mcss_penalty_weight': self.mcss_penalty_weight,
            'use_spectral_steering': self.use_spectral_steering,
            'steering_alpha': self.steering_alpha,
            'steering_beta': self.steering_beta,
            'head_scores_path': self.head_scores_path,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AGSARConfig':
        """Create config from dictionary."""
        # Handle dtype conversion
        if 'preferred_dtype' in config_dict:
            dtype_str = config_dict['preferred_dtype']
            if isinstance(dtype_str, str):
                dtype_map = {
                    'torch.bfloat16': torch.bfloat16,
                    'torch.float16': torch.float16,
                    'torch.float32': torch.float32,
                }
                config_dict['preferred_dtype'] = dtype_map.get(dtype_str, torch.bfloat16)
        return cls(**config_dict)
