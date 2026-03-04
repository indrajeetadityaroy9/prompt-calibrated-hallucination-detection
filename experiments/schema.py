"""Experiment configuration schema — typed dataclasses loaded from YAML."""

from dataclasses import dataclass
from typing import List, Optional

import yaml


@dataclass
class RunConfig:
    mode: str  # "evaluation" or "ablation"


@dataclass
class ModelConfig:
    name: str
    torch_dtype: str
    attn_implementation: str


@dataclass
class EvaluationConfig:
    datasets: List[str]
    n_samples: int
    max_new_tokens: int
    f1_threshold: float
    max_context_chars: int
    seed: int


@dataclass
class AblationConfig:
    signals: List[str]


@dataclass
class OutputConfig:
    dir: str


@dataclass
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    ablation: Optional[AblationConfig] = None

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        ablation = None
        if "ablation" in raw:
            ablation = AblationConfig(**raw["ablation"])

        return cls(
            run=RunConfig(**raw["run"]),
            model=ModelConfig(**raw["model"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            output=OutputConfig(**raw["output"]),
            ablation=ablation,
        )
