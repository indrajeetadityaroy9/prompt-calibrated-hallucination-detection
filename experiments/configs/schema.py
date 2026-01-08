"""
Configuration schema for AG-SAR experiments.

Uses Pydantic for validation and YAML serialization.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: Literal[
        "halueval_qa",
        "halueval_summarization",
        "halueval_dialogue",
        "ragtruth",
        "truthfulqa",
        "wikitext",
        "fava",
        "ALL",
    ] = Field(..., description="Dataset identifier")
    num_samples: Optional[int] = Field(None, ge=1)
    seed: int = Field(42)
    task_type: Optional[str] = Field(None)


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="HuggingFace model ID")
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field("sdpa")
    dtype: Literal["bfloat16", "float16", "float32"] = Field("bfloat16")
    device_map: str = Field("auto")
    trust_remote_code: bool = Field(True)
    batch_size: int = Field(8, ge=1, le=64)


class AGSARMethodConfig(BaseModel):
    """AG-SAR configuration - minimal dynamic architecture."""

    semantic_layers: int = Field(4, ge=1, le=32)
    varentropy_lambda: float = Field(1.0, ge=0.0, le=5.0)
    sigma_multiplier: float = Field(-1.0)
    calibration_window: int = Field(64, ge=1)


class SelfCheckMethodConfig(BaseModel):
    """SelfCheck method configuration."""

    num_samples: int = Field(5, ge=1, le=20)
    max_new_tokens: int = Field(100, ge=10, le=500)
    temperature: float = Field(1.0, gt=0.0, le=2.0)


class EigenScoreMethodConfig(BaseModel):
    """EigenScore method configuration."""

    num_samples: int = Field(5, ge=2, le=20)
    max_new_tokens: int = Field(50, ge=10, le=200)
    temperature: float = Field(1.0, gt=0.0, le=2.0)


class SemanticEntropyMethodConfig(BaseModel):
    """Semantic Entropy method configuration."""

    num_samples: int = Field(5, ge=2, le=20)
    similarity_threshold: float = Field(0.8, ge=0.5, le=1.0)
    embedding_model: str = Field("all-MiniLM-L6-v2")
    max_new_tokens: int = Field(256, ge=10, le=500)
    temperature: float = Field(0.7, gt=0.0, le=2.0)


class MethodsConfig(BaseModel):
    """Methods to run in the experiment."""

    agsar: Optional[AGSARMethodConfig] = Field(None)
    logprob: bool = Field(False)
    entropy: bool = Field(False)
    selfcheck: Optional[SelfCheckMethodConfig] = Field(None)
    eigenscore: Optional[EigenScoreMethodConfig] = Field(None)
    semantic_entropy: Optional[SemanticEntropyMethodConfig] = Field(None)
    saplma: bool = Field(False)


class EvaluationConfig(BaseModel):
    """Evaluation and metrics configuration."""

    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    metrics: List[str] = Field(default=["auroc", "auprc", "f1"])
    bootstrap_samples: int = Field(1000, ge=100, le=10000)
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        valid = {
            "auroc", "auprc", "auprc_factual",
            "f1", "precision", "recall", "accuracy",
            "tpr_at_5fpr", "ece", "brier",
            "aurc", "risk_80", "risk_90", "risk_95",
            "spearman", "pearson", "pointbiserial",
        }
        for m in v:
            if m not in valid:
                raise ValueError(f"Invalid metric '{m}'. Valid: {valid}")
        return v


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = Field("results")
    save_predictions: bool = Field(True)
    save_scores: bool = Field(True)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    experiment: Dict[str, str] = Field(default={"name": "unnamed", "description": ""})
    dataset: DatasetConfig
    model: ModelConfig
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        import yaml
        from pathlib import Path

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        import yaml
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_enabled_methods(self) -> List[str]:
        methods = []
        if self.methods.agsar:
            methods.append("agsar")
        if self.methods.logprob:
            methods.append("logprob")
        if self.methods.entropy:
            methods.append("entropy")
        if self.methods.selfcheck:
            methods.append("selfcheck")
        if self.methods.eigenscore:
            methods.append("eigenscore")
        if self.methods.semantic_entropy:
            methods.append("semantic_entropy")
        if self.methods.saplma:
            methods.append("saplma")
        return methods
