"""Experiment configuration schema — typed dataclasses loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import yaml


@dataclass
class RunConfig:
    mode: Literal["evaluation", "ablation"]


@dataclass
class ModelConfig:
    name: str
    torch_dtype: str
    attn_implementation: str


@dataclass
class EvaluationConfig:
    datasets: list[str]
    n_samples: int
    max_new_tokens: int
    f1_threshold: float
    max_context_chars: int
    seed: int


@dataclass
class AblationConfig:
    signals: list[str]


@dataclass
class OutputConfig:
    dir: str


@dataclass
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    ablation: AblationConfig | None = None

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)

        for key in ("run", "model", "evaluation", "output"):
            if key not in raw:
                raise ValueError(f"Missing required config section: {key!r} in {path}")

        ablation = None
        if "ablation" in raw:
            ablation = AblationConfig(**raw["ablation"])

        config = cls(
            run=RunConfig(**raw["run"]),
            model=ModelConfig(**raw["model"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            output=OutputConfig(**raw["output"]),
            ablation=ablation,
        )

        if config.run.mode == "ablation" and config.ablation is None:
            raise ValueError("Config mode is 'ablation' but no 'ablation' section provided.")

        return config
