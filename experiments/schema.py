from dataclasses import dataclass
from typing import Literal

import yaml

VALID_DATASETS = ("triviaqa", "squad")


@dataclass
class ModelConfig:
    name: str
    dtype: str
    attn_implementation: str


@dataclass
class EvaluationConfig:
    datasets: list[str]
    n_samples: int
    max_new_tokens: int
    max_context_chars: int
    seed: int


@dataclass
class ExperimentConfig:
    mode: Literal["evaluation", "ablation"]
    model: ModelConfig
    evaluation: EvaluationConfig
    output_dir: str
    ablation_signals: list[str] | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        datasets = raw["evaluation"]["datasets"]
        for d in datasets:
            if d not in VALID_DATASETS:
                raise ValueError(f"Unknown dataset '{d}'. Valid: {VALID_DATASETS}")
        return cls(
            mode=raw["run"]["mode"],
            model=ModelConfig(**raw["model"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            output_dir=raw["output"]["dir"],
            ablation_signals=raw["ablation"]["signals"] if "ablation" in raw else None,
        )
