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
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        ablation = AblationConfig(**raw["ablation"]) if "ablation" in raw else None

        return cls(
            run=RunConfig(**raw["run"]),
            model=ModelConfig(**raw["model"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            output=OutputConfig(**raw["output"]),
            ablation=ablation,
        )
