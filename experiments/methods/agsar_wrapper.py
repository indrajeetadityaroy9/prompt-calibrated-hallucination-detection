"""
AG-SAR wrapper implementing UncertaintyMethod interface.

Supports multiple calibration modes:
- v8.0 Gold Master: Fixed parameters (default)
- v9.0 Task-Adaptive: Hardcoded per-task presets
- v10.0 Self-Calibrating: Mathematically derived from internal signals

Wraps the core AG-SAR library (src/ag_sar/) with the standardized
interface for benchmarking. Ensures proper cleanup of hooks.
"""

import time
from typing import Optional
import torch
import torch.nn as nn

from ag_sar import AGSAR, AGSARConfig
from ag_sar.calibration import SelfCalibrator, apply_temperature_scaling
from experiments.methods.base import UncertaintyMethod, MethodResult
from experiments.configs.schema import AGSARMethodConfig


# =============================================================================
# Task-Adaptive Calibration (v9.0)
# =============================================================================

# Task-specific calibration presets (empirically tuned)
# These parameters are selected based on task characteristics:
# - QA: Focused attention, conservative aggregation for safety
# - RAG: Trust context more (lower parametric_weight)
# - Summarization: Diffuse attention, higher dispersion_k for long-form
# - Attribution: Fine-grained spans, moderate conservative aggregation
TASK_PRESETS = {
    "qa": {
        "aggregation_method": "percentile_10",
        "calibration_temperature": 1.2,
        "dispersion_k": 5,
        "dispersion_sensitivity": 1.0,
        "parametric_weight": 0.4,
    },
    "rag": {
        "aggregation_method": "mean",
        "calibration_temperature": 1.5,
        "dispersion_k": 7,
        "dispersion_sensitivity": 1.2,
        "parametric_weight": 0.3,  # Trust context more
    },
    "summarization": {
        "aggregation_method": "percentile_25",
        "calibration_temperature": 2.5,
        "dispersion_k": 10,  # More tokens for long-form
        "dispersion_sensitivity": 0.8,
        "parametric_weight": 0.6,
    },
    "attribution": {
        "aggregation_method": "percentile_25",
        "calibration_temperature": 2.0,
        "dispersion_k": 8,
        "dispersion_sensitivity": 0.9,
        "parametric_weight": 0.5,
    },
    "default": {
        "aggregation_method": "mean",
        "calibration_temperature": 1.0,
        "dispersion_k": 5,
        "dispersion_sensitivity": 1.0,
        "parametric_weight": 0.5,
    },
}


def detect_task_type(dataset_name: str) -> str:
    """
    Detect task type from dataset name.

    Uses simple heuristics based on dataset naming conventions.
    Can be overridden via config.task_type_override.

    Args:
        dataset_name: Name of the dataset (e.g., "halueval_qa", "ragtruth")

    Returns:
        Task type: "qa", "rag", "summarization", "attribution", or "default"
    """
    if not dataset_name:
        return "default"

    dataset_lower = dataset_name.lower()

    if "qa" in dataset_lower or "question" in dataset_lower:
        return "qa"
    elif "rag" in dataset_lower or "ragtruth" in dataset_lower:
        return "rag"
    elif "summ" in dataset_lower or "summarization" in dataset_lower:
        return "summarization"
    elif "fava" in dataset_lower or "attribution" in dataset_lower:
        return "attribution"
    else:
        return "default"


def apply_temperature_scaling(score: float, temperature: float) -> float:
    """
    Apply temperature scaling to calibrate uncertainty scores.

    Temperature scaling adjusts the "sharpness" of the score distribution:
    - T > 1.0: Softens scores toward 0.5 (reduces overconfidence)
    - T < 1.0: Sharpens scores toward 0 or 1 (increases confidence)
    - T = 1.0: No change

    The transformation uses logit scaling:
        calibrated = sigmoid(logit(score) / T)

    Args:
        score: Raw uncertainty score in [0, 1]
        temperature: Calibration temperature (default 1.0)

    Returns:
        Calibrated score in [0, 1]
    """
    if temperature == 1.0:
        return score

    # Clamp to avoid numerical issues with logit
    eps = 1e-7
    score = max(eps, min(1 - eps, score))

    # Apply temperature scaling via logit transformation
    import math
    logit = math.log(score / (1 - score))
    scaled_logit = logit / temperature
    calibrated = 1 / (1 + math.exp(-scaled_logit))

    return calibrated


class AGSARMethod(UncertaintyMethod):
    """
    AG-SAR uncertainty method wrapper.

    Supports three calibration modes:
    - v8.0: Fixed parameters (default)
    - v9.0: Task-adaptive presets based on dataset name
    - v10.0: Self-calibrating (parameters derived from internal signals)

    Example:
        >>> method = AGSARMethod(model, tokenizer, config=AGSARMethodConfig())
        >>> result = method.compute_score(prompt, response)
        >>> method.cleanup()  # Important: removes hooks
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[AGSARMethodConfig] = None,
        device: Optional[torch.device] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Initialize AG-SAR method.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
            config: AG-SAR configuration (uses defaults if None)
            device: Compute device
            dataset_name: Dataset name for task-adaptive parameter selection (v9.0)
        """
        super().__init__(model, tokenizer, device)

        config = config or AGSARMethodConfig()
        self._config = config

        # Self-Calibrating mode (v10.0) - highest priority
        if config.enable_self_calibration:
            self._calibrator = SelfCalibrator(
                k_min=config.sc_k_min,
                k_max=config.sc_k_max,
                warmup_samples=config.sc_warmup_samples,
                aggregation_gamma=config.sc_aggregation_gamma,
            )
            self._mode = "self_calibrating"
            self._task_type = None
            # Use base config values - they'll be overridden per-sample
            effective_aggregation = config.aggregation_method
            effective_temperature = config.calibration_temperature
            effective_dispersion_k = config.dispersion_k
            effective_dispersion_sens = config.dispersion_sensitivity
            effective_parametric_weight = config.parametric_weight

        # Task-adaptive parameter selection (v9.0)
        elif config.enable_task_adaptive and dataset_name:
            task_type = config.task_type_override or detect_task_type(dataset_name)
            preset = TASK_PRESETS.get(task_type, TASK_PRESETS["default"])

            effective_aggregation = preset["aggregation_method"]
            effective_temperature = preset["calibration_temperature"]
            effective_dispersion_k = preset["dispersion_k"]
            effective_dispersion_sens = preset["dispersion_sensitivity"]
            effective_parametric_weight = preset["parametric_weight"]
            self._task_type = task_type
            self._calibrator = None
            self._mode = "task_adaptive"

        # v8.0 behavior - use config as-is
        else:
            effective_aggregation = config.aggregation_method
            effective_temperature = config.calibration_temperature
            effective_dispersion_k = config.dispersion_k
            effective_dispersion_sens = config.dispersion_sensitivity
            effective_parametric_weight = config.parametric_weight
            self._task_type = None
            self._calibrator = None
            self._mode = "fixed"

        # Map experiment config to library config
        ag_config = AGSARConfig(
            # Core parameters
            semantic_layers=config.semantic_layers,
            power_iteration_steps=config.power_iteration_steps,
            residual_weight=config.residual_weight,
            # Unified Gating (v7.0+)
            enable_unified_gating=config.enable_unified_gating,
            stability_sensitivity=config.stability_sensitivity,
            parametric_weight=effective_parametric_weight,
            # Semantic Dispersion (v8.0)
            enable_semantic_dispersion=config.enable_semantic_dispersion,
            dispersion_k=effective_dispersion_k,
            dispersion_sensitivity=effective_dispersion_sens,
            # Authority Aggregation (Safety-Focused)
            aggregation_method=effective_aggregation,
        )

        self._engine = AGSAR(model, tokenizer, config=ag_config)
        self._temperature = effective_temperature

    @property
    def name(self) -> str:
        return "AG-SAR"

    @property
    def requires_sampling(self) -> bool:
        return False

    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute uncertainty using AG-SAR.

        Supports three modes:
        - v8.0 Fixed: Uses configured parameters
        - v9.0 Task-Adaptive: Uses preset parameters based on task type
        - v10.0 Self-Calibrating: Full SC pipeline with all adaptive parameters

        Post-hoc calibration via temperature scaling is applied.
        """
        t0 = time.perf_counter()

        # Self-Calibrating mode (v10.0): Use full SC pipeline with raw signals
        if self._mode == "self_calibrating" and self._calibrator is not None:
            # Get all raw signals from engine
            raw_result = self._engine.compute_uncertainty_raw(prompt, response)

            latency = (time.perf_counter() - t0) * 1000

            # Use full self-calibrating score computation
            sc_result = self._calibrator.compute_full_self_calibrating_score(
                authority_per_token=raw_result["authority_per_token"],
                attention_weights=raw_result["attention_weights"],
                h_attn=raw_result["h_attn"],
                h_block=raw_result["h_block"],
                logits=raw_result["logits"],
                embed_matrix=raw_result["embed_matrix"],
                response_start=raw_result["response_start"],
                attention_mask=raw_result["attention_mask"],
            )

            extra = {
                "authority": sc_result["authority_aggregated"],
                "authority_mean": sc_result["authority_mean"],
                "authority_p10": sc_result["authority_p10"],
                "authority_var": sc_result["authority_var"],
                "metric": "v10_full_sc",
                "raw_score": sc_result["raw_score"],
                "temperature": sc_result["temperature"],
                "k_effective": sc_result["k_effective"],
                "aggregation_method": sc_result["aggregation_method"],
                "dispersion": sc_result["dispersion"],
                "divergence": sc_result["divergence"],
                "stability_gate": sc_result["stability_gate"],
                "parametric_weight": sc_result["parametric_weight"],
                "mode": "self_calibrating_full",
                "samples_seen": sc_result["samples_seen"],
                "is_warmed_up": sc_result["is_warmed_up"],
            }

            return MethodResult(
                score=sc_result["score"],
                confidence=sc_result["confidence_mean"],
                latency_ms=latency,
                extra=extra,
            )

        # v8.0 / v9.0: Standard path
        result = self._engine.compute_uncertainty(prompt, response, return_details=True)

        latency = (time.perf_counter() - t0) * 1000

        raw_score = result["score"]
        calibrated_score = apply_temperature_scaling(raw_score, self._temperature)

        extra = {
            "authority": float(result.get("authority", 0)),
            "metric": f"v9_{self._task_type}" if self._task_type else "v8",
            "raw_score": raw_score,
            "temperature": self._temperature,
            "task_type": self._task_type,
            "mode": self._mode,
        }

        return MethodResult(
            score=calibrated_score,
            confidence=result.get("model_confidence"),
            latency_ms=latency,
            extra=extra,
        )

    def cleanup(self) -> None:
        """
        Release AG-SAR resources.

        CRITICAL: This removes the monkey-patched attention hooks.
        Must be called before using another method on the same model.
        """
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.cleanup()
            self._engine.reset()
