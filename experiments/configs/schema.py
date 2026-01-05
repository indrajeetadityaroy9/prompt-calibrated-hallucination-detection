"""
Configuration schema for AG-SAR experiments.

Uses Pydantic for validation and YAML serialization.
Maps directly to paper sections for reproducibility.

Each ExperimentConfig corresponds to one experiment in the paper:
    - experiment.name -> Paper experiment number (e.g., "exp1_halueval_qa")
    - dataset -> Section 4.1 Datasets
    - model -> Section 4.2 Models
    - methods -> Section 3 Method + Baselines
    - evaluation -> Section 4.3 Metrics
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
    ] = Field(..., description="Dataset identifier")
    num_samples: Optional[int] = Field(
        None, description="Max samples to load (None = all)", ge=1
    )
    seed: int = Field(42, description="Random seed for sampling")
    task_type: Optional[str] = Field(
        None, description="Task filter for RAGTruth (QA, Summary, Data2txt)"
    )


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="HuggingFace model ID or local path")
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        "sdpa", description="Attention implementation"
    )
    dtype: Literal["bfloat16", "float16", "float32"] = Field(
        "bfloat16", description="Model precision"
    )
    device_map: str = Field("auto", description="Device placement strategy")
    trust_remote_code: bool = Field(True, description="Trust remote code for custom models")


class AGSARMethodConfig(BaseModel):
    """AG-SAR specific configuration (maps to AGSARConfig in src/ag_sar/)."""

    semantic_layers: int = Field(4, ge=1, le=32, description="Number of final layers to analyze")
    power_iteration_steps: int = Field(3, ge=1, le=10, description="Power iteration convergence steps")
    residual_weight: float = Field(0.5, ge=0.0, le=1.0, description="Residual weight for centrality")
    enable_register_filter: bool = Field(True, description="Enable kurtosis-based sink filter")
    enable_spectral_roughness: bool = Field(True, description="Enable MLP divergence penalty")
    kurtosis_threshold: float = Field(2.0, ge=0.0, description="Register filter threshold")
    lambda_roughness: float = Field(10.0, ge=0.0, description="MLP divergence penalty weight")
    ema_decay: float = Field(0.995, gt=0.0, lt=1.0, description="EMA decay for online adaptation")
    sink_token_count: int = Field(4, ge=0, description="Number of sink tokens to mask")


class SelfCheckMethodConfig(BaseModel):
    """SelfCheck method configuration."""

    num_samples: int = Field(5, ge=1, le=20, description="Number of stochastic samples to generate")
    max_new_tokens: int = Field(100, ge=10, le=500, description="Max tokens per generation")
    temperature: float = Field(1.0, gt=0.0, le=2.0, description="Sampling temperature")


class MethodsConfig(BaseModel):
    """
    Methods to run in the experiment.

    Set a method config to enable it, or use bool for simple baselines.
    Example:
        methods:
          agsar:
            semantic_layers: 4
          logprob: true
          selfcheck:
            num_samples: 5
    """

    agsar: Optional[AGSARMethodConfig] = Field(None, description="AG-SAR configuration")
    logprob: bool = Field(False, description="Enable LogProb baseline")
    entropy: bool = Field(False, description="Enable Predictive Entropy baseline")
    selfcheck: Optional[SelfCheckMethodConfig] = Field(None, description="SelfCheck configuration")
    eigenscore: bool = Field(False, description="Enable EigenScore baseline")
    saplma: bool = Field(False, description="Enable SAPLMA baseline")


class EvaluationConfig(BaseModel):
    """Evaluation and metrics configuration."""

    confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Threshold for confident subset analysis"
    )
    metrics: List[str] = Field(
        default=["auroc", "auprc", "f1", "precision", "recall"],
        description="Metrics to compute",
    )
    bootstrap_samples: int = Field(
        1000, ge=100, le=10000, description="Bootstrap samples for CI"
    )
    confidence_level: float = Field(
        0.95, gt=0.0, lt=1.0, description="Confidence level for CI"
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v):
        # ICML/NeurIPS-grade metrics: discrimination + calibration + utility + correlation
        valid = {
            # Discrimination (higher=better)
            "auroc", "auprc", "auprc_factual",
            # Classification (higher=better)
            "f1", "precision", "recall", "accuracy",
            # Calibration (lower=better)
            "ece", "brier",
            # Coverage & Utility (lower=better, consistent polarity)
            "aurc", "risk_80", "risk_90", "risk_95",
            # Correlation (higher=better)
            "spearman", "pearson", "pointbiserial",
        }
        for m in v:
            if m not in valid:
                raise ValueError(f"Invalid metric '{m}'. Valid: {valid}")
        return v


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = Field("results", description="Output directory for results")
    save_predictions: bool = Field(True, description="Save per-sample predictions")
    save_scores: bool = Field(True, description="Save raw uncertainty scores")


class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    Maps to paper sections:
        - experiment.name -> Paper experiment identifier
        - dataset -> Section 4.1 Datasets
        - model -> Section 4.2 Models
        - methods.agsar -> Section 3 Method
        - evaluation -> Section 4.3 Metrics

    Example YAML:
        experiment:
          name: "exp1_halueval_qa"
          description: "Main benchmark on HaluEval QA"
        dataset:
          name: "halueval_qa"
          num_samples: 2500
        model:
          name: "meta-llama/Llama-3.1-8B-Instruct"
        methods:
          agsar:
            semantic_layers: 4
          logprob: true
        evaluation:
          bootstrap_samples: 1000
    """

    experiment: Dict[str, str] = Field(
        default={"name": "unnamed", "description": ""},
        description="Experiment metadata",
    )
    dataset: DatasetConfig
    model: ModelConfig
    methods: MethodsConfig = Field(default_factory=MethodsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    model_config = {"extra": "forbid"}  # Pydantic v2: reject unknown fields

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to YAML config file

        Returns:
            ExperimentConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config is invalid
        """
        import yaml
        from pathlib import Path

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """
        Save config to YAML file.

        Args:
            path: Output path for YAML file
        """
        import yaml
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def get_enabled_methods(self) -> List[str]:
        """Return list of enabled method names."""
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
        if self.methods.saplma:
            methods.append("saplma")
        return methods
