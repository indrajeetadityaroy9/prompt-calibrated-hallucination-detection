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

    # ===== Entropy-Weighted Divergence (v4.0 - WikiBio enhancement) =====
    # Amplifies divergence when model is uncertain (high entropy)
    # Score(t) = Divergence(t) * (1 + entropy_beta * H(p_t))
    entropy_beta: float = 0.0  # 0.0 = disabled, 1.0 = recommended for WikiBio

    # ===== Subject Anchor (v4.0 - WikiBio context-free enhancement) =====
    # Boosts authority contribution from first N tokens (the "subject")
    # In WikiBio-style generation without context, the subject serves as anchor
    # Valid facts link back to subject; hallucinations drift away
    subject_boost: float = 0.0  # 0.0 = disabled, 5.0 = recommended for WikiBio
    subject_token_count: int = 5  # First N tokens treated as subject

    # ===== Local Intrinsic Dimension (v5.0/v5.1 - Manifold Geometry) =====
    # Detects confabulation by measuring manifold complexity
    # Hallucinations traverse high-dimensional, disordered regions
    # Factual generations follow low-dimensional, well-worn manifolds
    # Based on: "Characterizing Truthfulness in LLM Generations with LID" (arXiv 2024)
    #
    # v5.1 Enhancement: Prompt-Anchored Calibration
    # - "prompt": Use prompt's LID as baseline (zero-shot, recommended)
    # - "fixed": Use hardcoded lid_mean/lid_std (legacy, not recommended)
    enable_lid: bool = False  # Enable LID-based confabulation detection
    lid_k: int = 10  # Number of nearest neighbors for LID estimation
    lid_weight: float = 1.0  # Sensitivity for LID penalty (higher = more aggressive)
    lid_calibration: Literal["prompt", "fixed"] = "prompt"  # Calibration mode
    # Legacy fixed calibration (only used if lid_calibration="fixed")
    lid_mean: float = 6.0  # Expected mean LID (dataset-specific, deprecated)
    lid_std: float = 2.0  # Expected std of LID (dataset-specific, deprecated)

    # ===== Spectral-Structural Methods (v6.0) =====
    # Combines Laplacian Spectral Entropy (graph structure) with
    # Layer-Contrastive Divergence (DoLa-style depth dynamics)
    # Based on: "Hallucination Detection Using Spectral Features" (arXiv 2025)
    #           "DoLa: Decoding by Contrasting Layers" (ICLR 2024)
    enable_spectral: bool = False  # Enable spectral-structural hallucination detection
    spectral_window: int = 20  # Window size for local Laplacian entropy
    spectral_alpha: float = 1.0  # Weight for Laplacian entropy (higher entropy = bad)
    spectral_beta: float = 1.0  # Weight for layer divergence (higher divergence = good)
    # Layer selection for DoLa-style contrastive divergence
    early_layer_ratio: float = 0.25  # Early layer = this fraction of total layers
    late_layer_ratio: float = 0.875  # Late layer = this fraction of total layers

    # ===== Context-Dependent Gating (v7.0) =====
    # Unified framework for RAG and Free Generation
    # Dynamically shifts trust between Provenance (Flow) and Confidence (Parametric)
    # based on whether the model is attending to context or ignoring it.
    #
    # Master Equation:
    #   A(t) = Flow(t) + (1 - Σ A_{prompt}) × Confidence(t) × parametric_weight
    #
    # - In RAG: attn_to_context ≈ 1.0 → Injection OFF → Trust Flow
    # - In Free Gen: attn_to_context ≈ 0.0 → Injection ON → Trust Confidence
    enable_v7_gating: bool = False  # Enable context-dependent gating
    stability_sensitivity: float = 1.0  # Controls conductivity gate sharpness (1.0 optimal)
    parametric_weight: float = 0.5  # Weight for confidence injection when ignoring context

    # ===== Semantic Dispersion (v8.0 - Consistency over Confidence) =====
    # Replaces raw confidence with semantic consistency of top-k predictions
    # Key insight: "Confidently wrong" vs "Semantically confused"
    # - Low dispersion: Top-k tokens are synonyms (US, USA, America) → Grounded
    # - High dispersion: Top-k tokens are unrelated (Paris, London, Rome) → Hallucination
    #
    # Upgrade to Master Equation:
    #   A(t) = Flow(t) + (1 - Gate(t)) × (1 - Dispersion(t)) × parametric_weight
    enable_semantic_dispersion: bool = False  # Use semantic dispersion instead of confidence
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
            'entropy_beta': self.entropy_beta,
            'subject_boost': self.subject_boost,
            'subject_token_count': self.subject_token_count,
            'enable_lid': self.enable_lid,
            'lid_k': self.lid_k,
            'lid_weight': self.lid_weight,
            'lid_calibration': self.lid_calibration,
            'lid_mean': self.lid_mean,
            'lid_std': self.lid_std,
            # Spectral-Structural (v6.0)
            'enable_spectral': self.enable_spectral,
            'spectral_window': self.spectral_window,
            'spectral_alpha': self.spectral_alpha,
            'spectral_beta': self.spectral_beta,
            'early_layer_ratio': self.early_layer_ratio,
            'late_layer_ratio': self.late_layer_ratio,
            # v7.0 Context-Dependent Gating
            'enable_v7_gating': self.enable_v7_gating,
            'stability_sensitivity': self.stability_sensitivity,
            'parametric_weight': self.parametric_weight,
            # v8.0 Semantic Dispersion
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
