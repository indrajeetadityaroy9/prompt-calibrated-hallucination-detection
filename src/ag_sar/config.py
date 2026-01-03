"""Configuration for AG-SAR uncertainty quantification pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class AGSARConfig:
    """
    Configuration for AG-SAR v3.1 uncertainty quantification pipeline.

    Implements Recursive Authority Flow for Zero-Latency Hallucination Detection.
    Optimized for NVIDIA H100 with bfloat16 precision and Flash Attention 2.

    v3.1/v3.2 Mechanisms:
        1. Register Filter: EMA Z-score + Sigmoid gate (kurtosis-based)
        2. Authority Flow: Prompt Recharge + Gen Flow (recursive)
        3. MLP Divergence: Detects when MLP overrides attention (cosine distance)

    H100 Optimizations:
        - BFloat16 precision (3x faster than FP32, stable numerics)
        - Flash Attention 2 (~2x speedup over SDPA)
        - TF32 for matrix operations (~3x speedup)
        - Hopper-tuned Triton kernels (BLOCK_SIZE=256)
    """

    # ===== Register Filter (Mechanism 1) =====
    enable_register_filter: bool = True
    ema_decay: float = 0.995
    kurtosis_threshold: float = 2.0

    # ===== Authority Flow (Mechanism 2) =====
    enable_authority_flow: bool = True
    recharge_weight: float = 1.0

    # ===== MLP Divergence / Spectral Roughness (Mechanism 3) =====
    enable_spectral_roughness: bool = True
    lambda_roughness: float = 10.0

    # ===== Semantic Layer Selection =====
    semantic_layers: int = 4

    # ===== Centrality Computation =====
    residual_weight: float = 0.5
    power_iteration_steps: int = 3
    power_iteration_tol: float = 1e-4

    # ===== Hallucination Detection =====
    hallucination_threshold: float = 0.7

    # ===== H100 Optimization Defaults =====
    # Precision: BFloat16 is optimal for H100 (range of FP32, speed of FP16)
    preferred_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)

    # Attention: Flash Attention 2 provides ~2x speedup on H100
    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"

    # TF32: 3x throughput vs FP32 on Hopper Tensor Cores
    use_tf32: bool = True

    # torch.compile: Enables Inductor backend optimizations
    use_torch_compile: bool = True
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"

    # Batch size: H100 80GB can handle large batches
    eval_batch_size: int = 32

    # Multi-GPU: Device placement strategy for model parallelism
    device_map: Literal["auto", "balanced", "sequential"] = "balanced"
    low_cpu_mem_usage: bool = True

    # ===== Model Architecture =====
    num_attention_heads: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    model_architecture: str = "auto"
    num_kv_heads: Optional[int] = None
    sink_token_count: int = 4

    # ===== Uncertainty Metric =====
    uncertainty_metric: str = "v31"  # v31 or authority

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.residual_weight <= 1.0:
            raise ValueError(f"residual_weight must be in [0, 1], got {self.residual_weight}")
        if self.power_iteration_steps < 1:
            raise ValueError(f"power_iteration_steps must be >= 1")
        if self.semantic_layers < 1:
            raise ValueError(f"semantic_layers must be >= 1")

        if self.preferred_dtype == torch.float16:
            import warnings
            warnings.warn(
                "float16 may cause NaN overflow with GPT-2. Use bfloat16 instead.",
                UserWarning
            )

        valid_metrics = ("v31", "authority")
        if self.uncertainty_metric not in valid_metrics:
            raise ValueError(f"uncertainty_metric must be one of {valid_metrics}")

        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1)")
        if self.lambda_roughness < 0:
            raise ValueError(f"lambda_roughness must be >= 0")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            # Mechanisms
            'enable_register_filter': self.enable_register_filter,
            'ema_decay': self.ema_decay,
            'kurtosis_threshold': self.kurtosis_threshold,
            'enable_authority_flow': self.enable_authority_flow,
            'recharge_weight': self.recharge_weight,
            'enable_spectral_roughness': self.enable_spectral_roughness,
            'lambda_roughness': self.lambda_roughness,
            # Computation
            'semantic_layers': self.semantic_layers,
            'residual_weight': self.residual_weight,
            'power_iteration_steps': self.power_iteration_steps,
            'power_iteration_tol': self.power_iteration_tol,
            'hallucination_threshold': self.hallucination_threshold,
            # H100 Optimization
            'preferred_dtype': str(self.preferred_dtype),
            'attn_implementation': self.attn_implementation,
            'use_tf32': self.use_tf32,
            'use_torch_compile': self.use_torch_compile,
            'compile_mode': self.compile_mode,
            'eval_batch_size': self.eval_batch_size,
            'device_map': self.device_map,
            'low_cpu_mem_usage': self.low_cpu_mem_usage,
            # Model
            'model_architecture': self.model_architecture,
            'sink_token_count': self.sink_token_count,
            # Metric
            'uncertainty_metric': self.uncertainty_metric,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AGSARConfig':
        """Create config from dictionary."""
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
