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

    SOTA v8.0 Mechanism (Default - Gold Master):
        1. Authority Flow: Tracks signal provenance from prompt to response
        2. Unified Gating: Dynamically balances context vs parametric trust
        3. Semantic Dispersion: Measures consistency over raw confidence
        4. Late-Layer Focus: Final 4 layers where decisions are consolidated

    Performance (Llama-3.1-8B, H100):
        - HaluEval QA: 0.89 AUROC (+30% vs LogProb baseline)
        - RAGTruth: 0.72 AUROC (+18% vs LogProb baseline)
        - Latency: ~1.1ms per inference

    Scope: AG-SAR detects EXTRINSIC hallucinations (unfaithful to source context),
    not INTRINSIC hallucinations (unfaithful to reality). For RAG faithfulness
    monitoring, the ground truth is provided in the context.

    For ablation studies:
        - Set enable_unified_gating=False for v3.1 pure Authority Flow
        - Set enable_semantic_dispersion=False for v7.0 gating without dispersion

    H100 Optimizations:
        - BFloat16 precision (3x faster than FP32, stable numerics)
        - Flash Attention 2 (~2x speedup over SDPA)
        - TF32 for matrix operations (~3x speedup)
        - Hopper-tuned Triton kernels (BLOCK_SIZE=256)
    """

    # ===== Authority Flow (Core Mechanism) =====
    recharge_weight: float = 1.0  # Initial prompt authority

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

    # Gate Sharpening (v12.1 - Fixes "Gate Leak" on TruthfulQA)
    # Problem: Intermediate gate values (~0.5) allow JEPA noise to pollute Truth Vector signal
    # Solution: Binarize gate - force to extremes if < low_thresh or > high_thresh
    # Effect: On TruthfulQA (no context), gate → 0.0, so intrinsic trust dominates
    enable_gate_sharpening: bool = False  # Enable gate binarization
    gate_sharpen_low: float = 0.2   # Below this → force to 0.0 (trust intrinsic only)
    gate_sharpen_high: float = 0.8  # Above this → force to 1.0 (trust JEPA only)

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

    # Dispersion Algorithm Selection
    # - "top1_projection": Legacy/QA - measures distance from top-1 prediction
    # - "centroid_variance": JEPA/Summ - measures spread around weighted centroid (Top-K)
    # - "nucleus_variance": SOTA/Summ - adaptive Top-P clustering (dynamic k)
    dispersion_method: Literal["top1_projection", "centroid_variance", "nucleus_variance"] = "top1_projection"

    # Nucleus Variance (Top-P) Parameters
    # Only used when dispersion_method="nucleus_variance"
    # Controls the cumulative probability threshold for dynamic clustering
    # - 0.95: Captures 95% of probability mass (larger clusters for uncertain tokens)
    # - 0.90: Tighter clusters (more aggressive)
    nucleus_top_p: float = 0.95

    # ===== Layer Drift (Mind-Change Detection) =====
    # EXPERIMENTAL - NEGATIVE RESULT ON RAGTruth (DO NOT USE FOR PRODUCTION)
    #
    # Original Hypothesis: Hallucinations show semantic suppression - the model considers
    # the correct answer in middle layers but overrides it with parametric memory.
    #
    # EMPIRICAL FINDING (Feature Inversion):
    # Layer Drift showed AUROC=0.23 (worse than random 0.5) on RAGTruth. This means
    # High Drift correlates with FAITHFULNESS, not hallucination.
    #
    # The "Effort Hypothesis" explains this:
    # - Faithful RAG answers require HIGH cognitive effort to suppress pre-trained priors
    #   and attend to novel context. This creates large vector rotation (high drift).
    # - Hallucinations follow the path of least resistance (pre-trained priors).
    #   The model flows smoothly from input to output without fighting its weights.
    #
    # Conclusion: Layer Drift measures "Thinking Effort", not "Deception".
    # For RAG faithfulness, use JEPA Centroid Variance (dispersion_method="centroid_variance").
    enable_layer_drift: bool = False  # KEEP DISABLED - negative result on RAGTruth
    drift_layer_ratio: float = 0.5  # Which layer to use as mid-layer (0.5 = middle of model)
    drift_sensitivity: float = 1.0  # Scale factor for drift penalty (1.0 = linear)

    # ===== Intrinsic Hallucination Detection (Truth Vector) =====
    # Enables detection of hallucinations WITHOUT context by projecting onto a calibrated
    # "truthfulness direction" in the model's residual stream.
    #
    # Requires pre-calibrated truth vector from scripts/calibrate_truth_vector.py
    #
    # DYNAMIC BLENDING: Uses Gate signal to control trust blend:
    #   trust = Gate × JEPA_trust + (1-Gate) × intrinsic_trust
    #
    # - High Gate (good RAG) → trust JEPA more (safe for novel data)
    # - Low Gate (hallucination) → trust Truth Vector more (catches lies)
    enable_intrinsic_detection: bool = False  # Enable Truth Vector integration
    truth_vector_path: Optional[str] = None  # Path to calibrated .pt file

    # ===== JEPA Drift Monitor (Phase 3 - Learned Predictor) =====
    # Replaces heuristic trust computation with a LEARNED metric: Semantic Drift.
    #
    # The JEPA Predictor is a neural network trained on "normal" text to predict
    # how thought vectors should evolve: h_{t+1} = Predictor(h_t)
    #
    # Drift = MSE(Predicted_h_{t+1}, Actual_h_{t+1})
    # High Drift = Model's thoughts are moving "abnormally" = Hallucination
    #
    # Key Advantage: No manual threshold tuning. The threshold is derived from
    # training loss (~0.2 MSE). Drift > 2×TrainingLoss indicates hallucination.
    enable_jepa_monitor: bool = False  # Enable JEPA drift-based detection
    jepa_predictor_path: Optional[str] = "data/models/jepa_predictor.pt"  # Trained predictor
    jepa_drift_threshold: float = 0.4  # Default ~2x Training MSE (0.2)
    jepa_monitor_layer: int = 16  # Layer to extract hidden states (middle layer)
    predictor_hidden_dim: int = 1024  # Must match training architecture

    # ===== Online Adaptation (Test-Time Training) =====
    # DEPRECATED: TTT approach hit the "Semantic Resolution Limit" - MLP predictors
    # cannot distinguish between entities in the same category (Paris vs London).
    # Kept for backward compatibility but not recommended for production.
    enable_online_adaptation: bool = False  # DISABLED - use Hybrid Controller instead
    online_adaptation_epochs: int = 15  # TTT epochs (if enabled)
    online_adaptation_lr: float = 0.02  # Aggressive LR (if enabled)

    # ===== Hybrid Controller (v13.0 - Production Solution) =====
    # Combines proven components:
    # 1. JEPA Variance (Phase 1): Detects confusion/semantic instability
    # 2. Truth Vector (Phase 2): Detects categorical lies (Watermelon)
    # 3. Symbolic Overlap (Phase 3): Detects RAG violations (Paris vs London)
    #
    # Trust Equation:
    #   extrinsic = jepa_weight * jepa_trust + symbolic_weight * symbolic_trust
    #   final = gate * extrinsic + (1 - gate) * intrinsic
    enable_hybrid_controller: bool = True  # Use the production hybrid approach
    symbolic_weight: float = 0.6  # Weight for symbolic (entity) overlap
    jepa_weight: float = 0.4  # Weight for JEPA variance
    enable_numeric_check: bool = True  # Also check numeric consistency

    # ===== Authority Aggregation (Safety-Focused) =====
    # Controls how authority scores across response tokens are aggregated.
    # - "mean": Average authority (default, good for ranking)
    # - "min": Minimum authority (conservative, catches worst-case tokens)
    # - "percentile_10": 10th percentile (robust conservative)
    # - "percentile_25": 25th percentile (moderate conservative)
    # - "importance_weighted": Weights by self-information (-log p) to emphasize rare tokens
    #
    # SOTA Recommendation: "importance_weighted" for hallucination detection.
    # Errors on rare tokens (names, dates) count 5x more than common tokens (the, and).
    # This aligns uncertainty with human judgment of severity.
    #
    # For safety-critical applications, use "min" or "percentile_10" to improve TPR@5%FPR.
    aggregation_method: Literal["mean", "min", "percentile_10", "percentile_25", "importance_weighted"] = "mean"

    # ===== Semantic Layer Selection =====
    # Controls which layers are analyzed for Authority Flow computation.
    # Default: Use the last 4 layers (e.g., layers 28-31 for Llama-3-8B's 32 layers)
    #
    # EMPIRICAL FINDING (contradicts DoLa/ROME hypothesis):
    # We tested middle-layer focus (layers 16-24) and found it DEGRADES performance
    # by -3.3% AUROC on RAGTruth. The "Consolidation Hypothesis" explains this:
    # - While facts may be retrieved in middle layers, the STRUCTURAL DECISION
    #   to commit to those facts is a late-stage phenomenon (layers 28-31).
    # - Late layers do not "smooth away" errors; they CONSOLIDATE them,
    #   making hallucinations topologically visible in the attention graph.
    # - Therefore, late-layer focus (default) is optimal for hallucination detection.
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

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            # Authority Flow (Core)
            'recharge_weight': self.recharge_weight,
            # Unified Gating (v7.0+)
            'enable_unified_gating': self.enable_unified_gating,
            'stability_sensitivity': self.stability_sensitivity,
            'parametric_weight': self.parametric_weight,
            'enable_gate_sharpening': self.enable_gate_sharpening,
            'gate_sharpen_low': self.gate_sharpen_low,
            'gate_sharpen_high': self.gate_sharpen_high,
            # Semantic Dispersion (v8.0)
            'enable_semantic_dispersion': self.enable_semantic_dispersion,
            'dispersion_k': self.dispersion_k,
            'dispersion_sensitivity': self.dispersion_sensitivity,
            'dispersion_method': self.dispersion_method,
            'nucleus_top_p': self.nucleus_top_p,
            # Layer Drift
            'enable_layer_drift': self.enable_layer_drift,
            'drift_layer_ratio': self.drift_layer_ratio,
            'drift_sensitivity': self.drift_sensitivity,
            # Intrinsic Detection (Truth Vector)
            'enable_intrinsic_detection': self.enable_intrinsic_detection,
            'truth_vector_path': self.truth_vector_path,
            # JEPA Drift Monitor
            'enable_jepa_monitor': self.enable_jepa_monitor,
            'jepa_predictor_path': self.jepa_predictor_path,
            'jepa_drift_threshold': self.jepa_drift_threshold,
            'jepa_monitor_layer': self.jepa_monitor_layer,
            'predictor_hidden_dim': self.predictor_hidden_dim,
            # Online Adaptation (TTT) - DEPRECATED
            'enable_online_adaptation': self.enable_online_adaptation,
            'online_adaptation_epochs': self.online_adaptation_epochs,
            'online_adaptation_lr': self.online_adaptation_lr,
            # Hybrid Controller (v13.0)
            'enable_hybrid_controller': self.enable_hybrid_controller,
            'symbolic_weight': self.symbolic_weight,
            'jepa_weight': self.jepa_weight,
            'enable_numeric_check': self.enable_numeric_check,
            # Aggregation
            'aggregation_method': self.aggregation_method,
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

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'AGSARConfig':
        """
        Create config from a task-specific preset.

        Presets provide task-optimized calibration parameters for:
        - qa: Question answering (conservative aggregation)
        - rag: Retrieval-augmented generation (trust context more)
        - summarization: Long-form text (higher dispersion_k)
        - attribution: Source attribution (moderate conservative)
        - default: Baseline parameters

        Args:
            preset_name: Name of the preset (e.g., "qa", "rag")
            **overrides: Additional parameters to override preset values

        Returns:
            AGSARConfig with preset parameters applied

        Example:
            >>> config = AGSARConfig.from_preset("qa")
            >>> config.aggregation_method
            'percentile_10'

            >>> config = AGSARConfig.from_preset("rag", dispersion_k=10)
            >>> config.dispersion_k
            10
        """
        from ag_sar.presets import load_preset

        # Load preset parameters
        preset_params = load_preset(preset_name)

        # Map preset keys to config parameter names
        config_kwargs = {}
        if "aggregation_method" in preset_params:
            config_kwargs["aggregation_method"] = preset_params["aggregation_method"]
        if "dispersion_k" in preset_params:
            config_kwargs["dispersion_k"] = preset_params["dispersion_k"]
        if "dispersion_sensitivity" in preset_params:
            config_kwargs["dispersion_sensitivity"] = preset_params["dispersion_sensitivity"]
        if "parametric_weight" in preset_params:
            config_kwargs["parametric_weight"] = preset_params["parametric_weight"]
        if "dispersion_method" in preset_params:
            config_kwargs["dispersion_method"] = preset_params["dispersion_method"]

        # Note: calibration_temperature is handled by the wrapper, not AGSARConfig

        # Apply overrides
        config_kwargs.update(overrides)

        return cls(**config_kwargs)
