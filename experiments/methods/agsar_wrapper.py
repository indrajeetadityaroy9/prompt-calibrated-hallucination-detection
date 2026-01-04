"""
AG-SAR wrapper implementing UncertaintyMethod interface.

Wraps the core AG-SAR library (src/ag_sar/) with the standardized
interface for benchmarking. Ensures proper cleanup of hooks.
"""

import time
from typing import Optional
import torch
import torch.nn as nn

from ag_sar import AGSAR, AGSARConfig
from experiments.methods.base import UncertaintyMethod, MethodResult
from experiments.configs.schema import AGSARMethodConfig


class AGSARMethod(UncertaintyMethod):
    """
    AG-SAR uncertainty method wrapper.

    Wraps the production AG-SAR implementation with the experiment interface.
    Automatically handles hook registration and cleanup.

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
    ):
        """
        Initialize AG-SAR method.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
            config: AG-SAR configuration (uses defaults if None)
            device: Compute device
        """
        super().__init__(model, tokenizer, device)

        config = config or AGSARMethodConfig()

        # Map experiment config to library config
        ag_config = AGSARConfig(
            semantic_layers=config.semantic_layers,
            power_iteration_steps=config.power_iteration_steps,
            residual_weight=config.residual_weight,
            enable_register_filter=config.enable_register_filter,
            enable_spectral_roughness=config.enable_spectral_roughness,
            kurtosis_threshold=config.kurtosis_threshold,
            lambda_roughness=config.lambda_roughness,
            ema_decay=config.ema_decay,
            sink_token_count=config.sink_token_count,
        )

        self._engine = AGSAR(model, tokenizer, config=ag_config)
        self._config = config

    @property
    def name(self) -> str:
        return "AG-SAR"

    @property
    def requires_sampling(self) -> bool:
        return False

    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute uncertainty using AG-SAR v3.1/v3.2.

        Uses Authority Flow + MLP Divergence for zero-latency hallucination detection.
        """
        t0 = time.perf_counter()

        result = self._engine.compute_uncertainty(prompt, response, return_details=True)

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=result["score"],
            confidence=result.get("model_confidence"),
            latency_ms=latency,
            extra={
                "authority": float(result.get("authority", 0)),
                "metric": result.get("metric", "v31"),
            },
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
