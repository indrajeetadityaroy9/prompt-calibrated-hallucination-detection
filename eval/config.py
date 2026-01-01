"""Evaluation configuration."""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import torch


@dataclass
class EvalConfig:
    """
    Configuration for AG-SAR evaluation framework.

    Attributes:
        model_name: HuggingFace model identifier
        device: Compute device
        dtype: Tensor dtype (bfloat16 recommended for H100)
        results_dir: Directory for saving results
        num_samples: Number of samples for each experiment
        seq_lengths: Sequence lengths for profiling
        warmup_runs: Warmup iterations before timing
        benchmark_runs: Number of runs for timing
    """

    # Model settings
    model_name: str = "gpt2"
    device: str = "cuda"
    dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)

    # Paths
    results_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Optional[Path] = None

    # Experiment settings
    num_samples: int = 1000
    batch_size: int = 1

    # Profiling settings
    seq_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    warmup_runs: int = 10
    benchmark_runs: int = 100

    # Dataset settings
    truthfulqa_split: str = "validation"
    wikitext_name: str = "wikitext-103-v1"

    # Thresholds
    rouge_threshold: float = 0.3  # ROUGE-L threshold for factual label
    ece_num_bins: int = 10

    # NLI-based ground truth labeling (NeurIPS-grade)
    use_nli_labeling: bool = False
    nli_model_name: str = "cross-encoder/nli-deberta-v3-large"
    nli_threshold: float = 0.5  # Entailment threshold for factuality

    # Multi-seed for statistical significance
    seeds: List[int] = field(default_factory=lambda: [42])

    # Ablation flags
    ablation_configs: List[str] = field(default_factory=lambda: [
        "full",
        "no_residual",
        "no_head_filter",
        "no_value_norms",
        "uniform_graph"
    ])

    def __post_init__(self):
        """Validate configuration."""
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_name: str
    metrics: dict
    config: dict
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())

    def save(self, path: Path):
        """Save result to JSON."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'experiment_name': self.experiment_name,
                'metrics': self.metrics,
                'config': self.config,
                'timestamp': self.timestamp
            }, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentResult':
        """Load result from JSON."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
