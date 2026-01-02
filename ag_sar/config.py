"""Configuration for AG-SAR uncertainty quantification pipeline."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class AGSARConfig:
    """
    Configuration for AG-SAR uncertainty quantification pipeline.

    Optimized for NVIDIA H100 with bfloat16 precision.

    Attributes:
        semantic_layers: Number of final layers to use for semantic analysis.
        residual_weight: Weight for identity matrix in residual correction.
            A = (1 - residual_weight) * W_att + residual_weight * I
        power_iteration_steps: Maximum iterations for eigenvector centrality.
        power_iteration_tol: Convergence tolerance for power iteration.
        hallucination_threshold: GSE threshold for hallucination detection.
        preferred_dtype: Preferred tensor dtype. Use bfloat16 on H100
            (NEVER use float16 with GPT-2 - causes NaN overflow).

    Example:
        >>> config = AGSARConfig(
        ...     semantic_layers=4,
        ...     hallucination_threshold=0.7
        ... )
    """

    # Semantic layer selection
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
    uncertainty_metric: str = "gse"  # Options: "gse", "gss", "mcss"
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
    head_scores_path: Optional[str] = None  # Path to Z-scores JSON (or legacy weights)

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
        if self.uncertainty_metric not in ("gse", "gss", "mcss"):
            raise ValueError(
                f"uncertainty_metric must be 'gse', 'gss', or 'mcss', "
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

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
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
    def _migrate_legacy_config(cls, config_dict: dict) -> dict:
        """Filter deprecated keys for backwards compatibility."""
        deprecated = {
            'value_norm_type', 'compile_mode', 'use_residual_correction',
            'entropy_threshold_low', 'entropy_threshold_high', 'use_flash_attn',
            'use_head_weighting', 'head_weights_path', 'align_with_tokens',
        }
        return {k: v for k, v in config_dict.items() if k not in deprecated}

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AGSARConfig':
        """Create config from dictionary."""
        # Filter out deprecated keys
        config_dict = cls._migrate_legacy_config(config_dict)

        # Handle legacy 'use_compile' -> 'use_torch_compile' migration
        if 'use_compile' in config_dict:
            config_dict['use_torch_compile'] = config_dict.pop('use_compile')

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
