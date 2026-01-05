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
        "truthfulqa",
        "wikitext",
        "fava",
        "ALL",  # Special marker for all datasets (HaluEval QA, RAGTruth QA, HaluEval Summ, FAVA)
    ] = Field(..., description="Dataset identifier or 'ALL' for all datasets")
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
    batch_size: int = Field(8, ge=1, le=64, description="Inference batch size (reduce for large models)")


class AGSARMethodConfig(BaseModel):
    """AG-SAR v8.0 configuration (maps to AGSARConfig in src/ag_sar/)."""

    # Core parameters
    semantic_layers: int = Field(4, ge=1, le=32, description="Number of final layers to analyze")
    power_iteration_steps: int = Field(3, ge=1, le=10, description="Power iteration convergence steps")
    residual_weight: float = Field(0.5, ge=0.0, le=1.0, description="Residual weight for centrality")

    # Unified Gating - v7.0+
    enable_unified_gating: bool = Field(True, description="Enable context-dependent RAG/free-gen gating")
    stability_sensitivity: float = Field(1.0, ge=0.0, description="Gate sharpness for MLP stability")
    parametric_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for confidence when ignoring context")

    # Semantic Dispersion - v8.0
    enable_semantic_dispersion: bool = Field(True, description="Enable semantic consistency over raw confidence")
    dispersion_k: int = Field(5, ge=2, le=20, description="Top-k tokens for dispersion")
    dispersion_sensitivity: float = Field(1.0, ge=0.0, description="Scale factor for dispersion penalty")

    # Authority Aggregation (Safety-Focused)
    aggregation_method: Literal["mean", "min", "percentile_10", "percentile_25"] = Field(
        "mean", description="How to aggregate authority: mean (ranking) vs min/percentile (safety)"
    )

    # Post-hoc Calibration (fixes ECE without hurting ranking)
    calibration_temperature: float = Field(
        1.0, gt=0.0, le=10.0, description="Temperature for score calibration (1.0=no calibration, <1=sharper, >1=softer)"
    )

    # Task-Adaptive Calibration (v9.0)
    enable_task_adaptive: bool = Field(
        False, description="Enable automatic task-specific parameter selection based on dataset"
    )
    task_type_override: Optional[Literal["qa", "rag", "summarization", "attribution"]] = Field(
        None, description="Manual task type override (None = auto-detect from dataset name)"
    )

    # Self-Calibrating Mode (v10.0) - Mathematically derived parameters
    enable_self_calibration: bool = Field(
        False, description="Enable self-calibrating mode (derives all parameters from internal signals)"
    )
    sc_k_min: int = Field(3, ge=2, le=10, description="Minimum dispersion k (for focused attention)")
    sc_k_max: int = Field(15, ge=5, le=30, description="Maximum dispersion k (for diffuse attention)")
    sc_warmup_samples: int = Field(10, ge=1, le=100, description="Warmup samples before adaptive params")
    sc_aggregation_gamma: float = Field(2.0, ge=0.1, le=10.0, description="Sensitivity for aggregation interpolation")


class SelfCheckMethodConfig(BaseModel):
    """SelfCheck method configuration."""

    num_samples: int = Field(5, ge=1, le=20, description="Number of stochastic samples to generate")
    max_new_tokens: int = Field(100, ge=10, le=500, description="Max tokens per generation")
    temperature: float = Field(1.0, gt=0.0, le=2.0, description="Sampling temperature")
    generation_batch_size: int = Field(5, ge=1, le=20, description="Batch size for parallel generation")


class EigenScoreMethodConfig(BaseModel):
    """
    EigenScore method configuration.

    Based on "INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"
    Chen et al., ICLR 2024. https://github.com/D2I-ai/eigenscore

    Measures semantic diversity across multiple sampled responses using
    hidden state covariance eigenvalues.
    """

    num_samples: int = Field(5, ge=2, le=20, description="Number of samples for covariance (min 2)")
    max_new_tokens: int = Field(50, ge=10, le=200, description="Max tokens per sample")
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
    eigenscore: Optional[EigenScoreMethodConfig] = Field(None, description="EigenScore configuration (sampling-based)")
    saplma: bool = Field(False, description="Enable SAPLMA baseline")
    # LLM-Check methods (NeurIPS 2024) - zero-shot, single-pass
    llmcheck_attn: bool = Field(False, description="Enable LLM-Check Attention Score")
    llmcheck_hidden: bool = Field(False, description="Enable LLM-Check Hidden Score")
    llmcheck_logit: bool = Field(False, description="Enable LLM-Check Logit Score")


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
            # Operating point (higher=better)
            "tpr_at_5fpr",
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
        if self.methods.llmcheck_attn:
            methods.append("llmcheck_attn")
        if self.methods.llmcheck_hidden:
            methods.append("llmcheck_hidden")
        if self.methods.llmcheck_logit:
            methods.append("llmcheck_logit")
        return methods

    def eigenscore_is_enabled(self) -> bool:
        """Check if EigenScore is enabled (handles both bool and config)."""
        return self.methods.eigenscore is not None
