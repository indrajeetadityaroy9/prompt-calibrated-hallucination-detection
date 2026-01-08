"""
AG-SAR wrapper for experiment benchmarking.

Wraps the core AG-SAR library with the standardized UncertaintyMethod interface.
"""

import time
from typing import Optional
import torch
import torch.nn as nn

from ag_sar import AGSAR, AGSARConfig
from experiments.methods.base import UncertaintyMethod, MethodResult
from experiments.configs.schema import AGSARMethodConfig


class AGSARMethod(UncertaintyMethod):
    """AG-SAR uncertainty method wrapper."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[AGSARMethodConfig] = None,
        device: Optional[torch.device] = None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__(model, tokenizer, device)

        config = config or AGSARMethodConfig()
        self._config = config

        ag_config = AGSARConfig(
            semantic_layers=config.semantic_layers,
            varentropy_lambda=config.varentropy_lambda,
            sigma_multiplier=config.sigma_multiplier,
            calibration_window=config.calibration_window,
        )

        self._engine = AGSAR(model, tokenizer, config=ag_config)

    @property
    def name(self) -> str:
        return "AG-SAR"

    @property
    def requires_sampling(self) -> bool:
        return False

    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """Compute uncertainty using AG-SAR."""
        t0 = time.perf_counter()

        result = self._engine.compute_uncertainty(prompt, response, return_details=True)
        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=result["score"],
            confidence=result.get("model_confidence"),
            latency_ms=latency,
            extra={"authority": result.get("authority", 0)},
        )

    def cleanup(self) -> None:
        """Release AG-SAR resources and remove hooks."""
        if hasattr(self, "_engine") and self._engine is not None:
            self._engine.cleanup()
