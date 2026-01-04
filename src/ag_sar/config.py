"""Configuration for AG-SAR uncertainty quantification pipeline."""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class AGSARConfig:
    """
    Configuration for AG-SAR uncertainty quantification pipeline.

    AG-SAR (Attention-Graph Shifting Attention to Relevance) detects hallucinations
    by analyzing internal attention graph structure without external semantic models.

    SOTA v8.0 Mechanism (Default):
        1. Register Filter: Kurtosis-based attention sink detection
        2. Authority Flow: Tracks signal provenance from prompt to response
        3. Unified Gating: Dynamically balances context vs parametric trust
        4. Semantic Dispersion: Measures consistency over raw confidence

    For ablation studies (paper reproduction):
        - Set enable_unified_gating=False for v3.1 pure Authority Flow
        - Set enable_semantic_dispersion=False for v7.0 gating without dispersion
        - Archived ablation code (v4/v5/v6) is in legacy_research/

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

    # ===== Unified Gating (Context-Dependent) =====
    # Unified framework for RAG and Free Generation (enabled by default).
    # Dynamically shifts trust between Provenance (Flow) and Confidence (Parametric)
    # based on whether the model is attending to context or ignoring it.
    #
    # Master Equation:
    #   A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × parametric_weight
    #
    # - In RAG: attn_to_context ≈ 1.0 → Gate ≈ 1 → Trust Flow
    # - In Free Gen: attn_to_context ≈ 0.0 → Gate ≈ 0 → Trust Parametric
    enable_unified_gating: bool = True  # Enable context-dependent gating (DEFAULT ON)
    stability_sensitivity: float = 1.0  # Controls conductivity gate sharpness (1.0 optimal)
    parametric_weight: float = 0.5  # Weight for confidence injection when ignoring context

    # ===== Semantic Dispersion (Consistency over Confidence) =====
    # Replaces raw confidence with semantic consistency of top-k predictions (enabled by default).
    # Key insight: "Confidently wrong" vs "Semantically confused"
    # - Low dispersion: Top-k tokens are synonyms (US, USA, America) → Grounded
    # - High dispersion: Top-k tokens are unrelated (Paris, London, Rome) → Hallucination
    #
    # Master Equation:
    #   A(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t) × parametric_weight
    #   where Trust(t) = 1 - Dispersion(t) × sensitivity
    enable_semantic_dispersion: bool = True  # Use semantic dispersion instead of confidence (DEFAULT ON)
    dispersion_k: int = 5  # Number of top tokens to consider
    dispersion_sensitivity: float = 1.0  # Scale factor for dispersion penalty (1.0 optimal)

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

        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1)")
        if self.lambda_roughness < 0:
            raise ValueError(f"lambda_roughness must be >= 0")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            # Register Filter (Mechanism 1)
            'enable_register_filter': self.enable_register_filter,
            'ema_decay': self.ema_decay,
            'kurtosis_threshold': self.kurtosis_threshold,
            # Authority Flow (Mechanism 2)
            'enable_authority_flow': self.enable_authority_flow,
            'recharge_weight': self.recharge_weight,
            # MLP Divergence (Mechanism 3)
            'enable_spectral_roughness': self.enable_spectral_roughness,
            'lambda_roughness': self.lambda_roughness,
            # Unified Gating (v7.0+)
            'enable_unified_gating': self.enable_unified_gating,
            'stability_sensitivity': self.stability_sensitivity,
            'parametric_weight': self.parametric_weight,
            # Semantic Dispersion (v8.0)
            'enable_semantic_dispersion': self.enable_semantic_dispersion,
            'dispersion_k': self.dispersion_k,
            'dispersion_sensitivity': self.dispersion_sensitivity,
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
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AGSARConfig':
        """Create config from dictionary."""
        # Handle deprecated field names (backward compatibility)
        if 'enable_v7_gating' in config_dict and 'enable_unified_gating' not in config_dict:
            import warnings
            warnings.warn(
                "enable_v7_gating is deprecated, use enable_unified_gating",
                DeprecationWarning,
                stacklevel=2
            )
            config_dict['enable_unified_gating'] = config_dict.pop('enable_v7_gating')

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
