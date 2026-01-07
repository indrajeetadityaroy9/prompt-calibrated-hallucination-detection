"""
AG-SAR Configuration Module.

This module defines AGSARConfig, the validated configuration dataclass that
controls all aspects of the uncertainty quantification pipeline.

Mechanism Configuration:
    AG-SAR's three-pillar architecture is configured here:
    - Authority Flow parameters (recharge_weight, residual_weight)
    - Unified Gating parameters (stability_sensitivity, parametric_weight)
    - Semantic Dispersion parameters (dispersion_k, dispersion_method)

Pipeline Position:
    Configuration is consumed by the AGSAR engine at initialization time.
    Most parameters are immutable after engine construction.

Validation:
    All parameters are validated in __post_init__ with explicit bounds and
    error messages. Invalid configurations raise ValueError immediately.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import torch


@dataclass
class AGSARConfig:
    """
    Configuration for the AG-SAR uncertainty quantification pipeline.

    This dataclass defines all tunable parameters for hallucination detection.
    Parameters are organized by mechanism (Authority Flow, Unified Gating,
    Semantic Dispersion) with empirically validated defaults.

    Mechanism Overview:
        AG-SAR computes uncertainty U(t) for each response token t as:

            U(t) = 1 - A(t)

        where A(t) is the aggregated authority score computed via:

            A(t) = G(t) × Flow(t) + (1 - G(t)) × Trust(t) × parametric_weight

        Components:
            - Flow(t): Authority derived from attention to prompt tokens
            - G(t): Stability gate ∈ [0,1] based on attention vs MLP residual
            - Trust(t): Semantic trust = 1 - dispersion × sensitivity

    Detection Scope:
        AG-SAR detects EXTRINSIC hallucinations (unfaithful to provided context),
        NOT INTRINSIC hallucinations (factually incorrect but consistent with
        model's parametric knowledge). Optimal for RAG faithfulness monitoring.

    Attributes:
        recharge_weight: Initial authority assigned to prompt tokens. Higher
            values increase sensitivity to prompt-derived information.
            Range: (0, ∞), Default: 1.0

        enable_unified_gating: Whether to use the stability gate mechanism.
            When True, dynamically balances context authority vs parametric
            confidence. When False, uses pure authority flow (ablation mode).
            Default: True

        stability_sensitivity: Sharpness of the stability gate sigmoid.
            Higher values make the gate more binary (context vs parametric).
            Range: (0, ∞), Default: 1.0

        parametric_weight: Weight for parametric confidence when gate is low.
            Controls how much to trust model's internal confidence when it
            ignores context. Range: [0, 1], Default: 0.5

        enable_semantic_dispersion: Whether to use semantic consistency instead
            of raw confidence. When True, measures embedding-space coherence of
            top-k predictions. Default: True

        dispersion_k: Number of top tokens to consider for dispersion.
            Larger k captures more alternatives but may include noise.
            Range: [1, ∞), Default: 5

        dispersion_sensitivity: Scale factor for dispersion penalty.
            Higher values penalize semantic inconsistency more strongly.
            Range: [0, ∞), Default: 1.0

        dispersion_method: Algorithm for computing semantic dispersion.
            - "top1_projection": Distance from top-1 embedding (QA-optimized)
            - "centroid_variance": Variance around weighted centroid (summarization)
            - "nucleus_variance": Adaptive top-p clustering (dynamic)
            Default: "top1_projection"

        aggregation_method: How to aggregate per-token authority into sequence score.
            - "mean": Average (good for ranking)
            - "min": Minimum (conservative, catches worst-case)
            - "percentile_10": 10th percentile (robust conservative)
            - "percentile_25": 25th percentile (moderate)
            - "importance_weighted": Weight by self-information (rare tokens count more)
            Default: "mean"

        semantic_layers: Number of final layers to analyze. Late layers contain
            consolidated decisions; middle layers may have uncommitted representations.
            Range: [1, num_layers], Default: 4

        hallucination_threshold: Decision boundary for binary classification.
            Uncertainty scores above this threshold are classified as violations.
            Range: [0, 1], Default: 0.7
    """

    # =========================================================================
    # AUTHORITY FLOW PARAMETERS
    # =========================================================================
    # Authority Flow tracks information provenance from prompt to response.
    # Tokens deriving authority from prompt are grounded; those from memory
    # are flagged as potentially hallucinated.

    recharge_weight: float = 1.0
    """Initial authority for prompt tokens. Default 1.0 normalizes flow."""

    # =========================================================================
    # UNIFIED GATING PARAMETERS
    # =========================================================================
    # The stability gate G(t) determines whether to trust context-derived
    # authority (G≈1) or parametric confidence (G≈0). Computed from the
    # ratio of attention output to MLP residual norms.

    enable_unified_gating: bool = True
    """Enable dynamic context/parametric trust balancing. Set False for ablation."""

    stability_sensitivity: float = 1.0
    """Sigmoid sharpness for gate computation. Higher = more binary gate."""

    parametric_weight: float = 0.5
    """Weight for parametric confidence when gate is low (ignoring context)."""

    # =========================================================================
    # SEMANTIC DISPERSION PARAMETERS
    # =========================================================================
    # Semantic Dispersion measures top-k prediction consistency in embedding
    # space. Low dispersion (synonyms) indicates grounded generation; high
    # dispersion (unrelated alternatives) indicates hallucination risk.

    enable_semantic_dispersion: bool = True
    """Use embedding-space consistency instead of raw confidence."""

    dispersion_k: int = 5
    """Number of top tokens for dispersion computation."""

    dispersion_sensitivity: float = 1.0
    """Scale factor for dispersion penalty in trust computation."""

    dispersion_method: Literal["top1_projection", "centroid_variance", "nucleus_variance"] = "top1_projection"
    """Algorithm for semantic dispersion. See class docstring for details."""

    nucleus_top_p: float = 0.95
    """Cumulative probability threshold for nucleus_variance method only."""

    # =========================================================================
    # AGGREGATION PARAMETERS
    # =========================================================================
    # Controls how per-token authority scores are reduced to a sequence score.

    aggregation_method: Literal["mean", "min", "percentile_10", "percentile_25", "importance_weighted"] = "mean"
    """Reduction method for per-token authority. See class docstring."""

    # =========================================================================
    # LAYER SELECTION
    # =========================================================================
    # Which transformer layers to analyze for authority computation.

    semantic_layers: int = 4
    """Number of final layers to analyze. Late-layer focus is empirically optimal."""

    # =========================================================================
    # CENTRALITY COMPUTATION
    # =========================================================================
    # Parameters for power iteration eigenvector computation.

    residual_weight: float = 0.5
    """Self-loop weight in attention graph. Balances direct vs indirect authority."""

    power_iteration_steps: int = 3
    """Iterations for eigenvector approximation. 3 sufficient for convergence."""

    power_iteration_tol: float = 1e-4
    """Convergence tolerance for early stopping."""

    # =========================================================================
    # DETECTION THRESHOLD
    # =========================================================================

    hallucination_threshold: float = 0.7
    """Decision boundary for binary classification. Uncertainty > threshold = violation."""

    # =========================================================================
    # HARDWARE OPTIMIZATION
    # =========================================================================
    # Parameters for GPU inference optimization. Defaults tuned for H100.

    preferred_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)
    """Compute precision. BFloat16 optimal for Ampere/Hopper (FP32 range, FP16 speed)."""

    attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    """Attention backend. Flash Attention 2 provides ~2x speedup when available."""

    use_tf32: bool = True
    """Enable TF32 for matmul. ~3x speedup on Ampere/Hopper tensor cores."""

    use_torch_compile: bool = True
    """Enable torch.compile for graph optimization."""

    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "reduce-overhead"
    """Compilation mode. reduce-overhead minimizes kernel launch latency."""

    eval_batch_size: int = 32
    """Batch size for batched inference."""

    device_map: Literal["auto", "balanced", "sequential"] = "balanced"
    """Multi-GPU placement strategy."""

    low_cpu_mem_usage: bool = True
    """Minimize CPU memory during model loading."""

    # =========================================================================
    # MODEL ARCHITECTURE (AUTO-DETECTED)
    # =========================================================================

    num_attention_heads: Optional[int] = None
    """Override auto-detected attention head count."""

    num_hidden_layers: Optional[int] = None
    """Override auto-detected layer count."""

    model_architecture: str = "auto"
    """Architecture family. Auto-detected from model config."""

    num_kv_heads: Optional[int] = None
    """Key-value head count for GQA models (Llama-3+). None = auto-detect."""

    def __post_init__(self):
        """
        Validate all configuration parameters.

        Raises:
            ValueError: If any parameter is outside its valid range.

        Side Effects:
            Emits UserWarning if float16 dtype is selected (numerically unstable).
        """
        import warnings

        # Authority Flow validation
        if self.recharge_weight <= 0:
            raise ValueError(f"recharge_weight must be > 0, got {self.recharge_weight}")

        # Centrality validation
        if not 0.0 <= self.residual_weight <= 1.0:
            raise ValueError(f"residual_weight must be in [0, 1], got {self.residual_weight}")
        if self.power_iteration_steps < 1:
            raise ValueError(f"power_iteration_steps must be >= 1, got {self.power_iteration_steps}")
        if self.power_iteration_tol <= 0:
            raise ValueError(f"power_iteration_tol must be > 0, got {self.power_iteration_tol}")

        # Layer selection validation
        if self.semantic_layers < 1:
            raise ValueError(f"semantic_layers must be >= 1, got {self.semantic_layers}")

        # Unified Gating validation
        if self.stability_sensitivity <= 0:
            raise ValueError(f"stability_sensitivity must be > 0, got {self.stability_sensitivity}")
        if not 0.0 <= self.parametric_weight <= 1.0:
            raise ValueError(f"parametric_weight must be in [0, 1], got {self.parametric_weight}")

        # Semantic Dispersion validation
        if self.dispersion_k < 1:
            raise ValueError(f"dispersion_k must be >= 1, got {self.dispersion_k}")
        if self.dispersion_sensitivity < 0:
            raise ValueError(f"dispersion_sensitivity must be >= 0, got {self.dispersion_sensitivity}")
        if not 0.0 < self.nucleus_top_p <= 1.0:
            raise ValueError(f"nucleus_top_p must be in (0, 1], got {self.nucleus_top_p}")

        # Detection threshold validation
        if not 0.0 <= self.hallucination_threshold <= 1.0:
            raise ValueError(f"hallucination_threshold must be in [0, 1], got {self.hallucination_threshold}")

        # Batch size validation
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")

        # Dtype stability warning
        if self.preferred_dtype == torch.float16:
            warnings.warn(
                "float16 may cause NaN overflow with some models. Use bfloat16 for stability.",
                UserWarning
            )

    def to_dict(self) -> dict:
        """
        Serialize configuration to dictionary.

        Returns:
            dict: All configuration parameters with string dtype representation.
        """
        return {
            # Authority Flow
            'recharge_weight': self.recharge_weight,
            # Unified Gating
            'enable_unified_gating': self.enable_unified_gating,
            'stability_sensitivity': self.stability_sensitivity,
            'parametric_weight': self.parametric_weight,
            # Semantic Dispersion
            'enable_semantic_dispersion': self.enable_semantic_dispersion,
            'dispersion_k': self.dispersion_k,
            'dispersion_sensitivity': self.dispersion_sensitivity,
            'dispersion_method': self.dispersion_method,
            'nucleus_top_p': self.nucleus_top_p,
            # Aggregation
            'aggregation_method': self.aggregation_method,
            # Centrality
            'semantic_layers': self.semantic_layers,
            'residual_weight': self.residual_weight,
            'power_iteration_steps': self.power_iteration_steps,
            'power_iteration_tol': self.power_iteration_tol,
            # Detection
            'hallucination_threshold': self.hallucination_threshold,
            # Hardware
            'preferred_dtype': str(self.preferred_dtype),
            'attn_implementation': self.attn_implementation,
            'use_tf32': self.use_tf32,
            'use_torch_compile': self.use_torch_compile,
            'compile_mode': self.compile_mode,
            'eval_batch_size': self.eval_batch_size,
            'device_map': self.device_map,
            'low_cpu_mem_usage': self.low_cpu_mem_usage,
            # Architecture
            'model_architecture': self.model_architecture,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AGSARConfig':
        """
        Deserialize configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters.
                String dtype values are automatically converted.

        Returns:
            AGSARConfig: Validated configuration instance.
        """
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

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'AGSARConfig':
        """
        Create configuration from a task-specific preset.

        Presets provide empirically calibrated parameters for specific tasks:
        - "qa": Question answering (conservative aggregation, low dispersion_k)
        - "rag": Retrieval-augmented generation (trust context, high sensitivity)
        - "summarization": Long-form text (higher dispersion_k, centroid method)
        - "attribution": Source attribution (moderate conservative)
        - "default": Baseline parameters

        Args:
            preset_name: Preset identifier (e.g., "qa", "rag").
            **overrides: Parameters to override after preset loading.

        Returns:
            AGSARConfig: Configuration with preset values applied.

        Raises:
            ValueError: If preset_name is not recognized.

        Example:
            >>> config = AGSARConfig.from_preset("qa")
            >>> config.aggregation_method
            'percentile_10'
            >>> config = AGSARConfig.from_preset("rag", dispersion_k=10)
            >>> config.dispersion_k
            10
        """
        from ag_sar.presets import load_preset

        preset_params = load_preset(preset_name)

        # Map preset keys to config parameters
        config_kwargs = {}
        key_mapping = [
            "aggregation_method",
            "dispersion_k",
            "dispersion_sensitivity",
            "parametric_weight",
            "dispersion_method",
        ]
        for key in key_mapping:
            if key in preset_params:
                config_kwargs[key] = preset_params[key]

        # Apply user overrides
        config_kwargs.update(overrides)

        return cls(**config_kwargs)
