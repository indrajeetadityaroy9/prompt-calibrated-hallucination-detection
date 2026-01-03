# AG-SAR: Attention-Graph Shifting Attention to Relevance

Zero-latency uncertainty quantification for LLMs via internal attention graph analysis.

## Overview

AG-SAR detects hallucinations by analyzing the internal attention structure of language models, without requiring external semantic models or multiple forward passes. It achieves **O(N) memory** complexity through matrix-free eigenvector centrality computation.

### Key Features

- **Zero External Latency**: Pure internal analysis—no NLI models or sampling
- **Architecture Support**: GPT-2, Llama-3/3.1/3.2, Mistral, Qwen
- **H100 Optimized**: BFloat16 precision with TF32 acceleration
- **Multiple Metrics**: GSE, MC-SS, Authority Flow

## Installation

```bash
# Basic installation
pip install -e .

# With development dependencies
pip install -e ".[dev]"

# Full installation (includes evaluation dependencies)
pip install -e ".[all]"
```

## Quick Start

```python
from ag_sar import AGSAR, AGSARConfig

# Initialize
config = AGSARConfig(semantic_layers=4)
agsar = AGSAR(model, tokenizer, config)

# Compute uncertainty
score = agsar.compute_uncertainty(
    "What is the capital of France?",
    "Paris"
)

# Detect hallucination
is_hallucination, confidence, details = agsar.detect_hallucination(
    "Who was the first person on Mars?",
    "Neil Armstrong landed on Mars in 1969."
)
```

## Uncertainty Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| `gse` | Graph-Shifted Entropy | Default, general purpose |
| `mcss` | Manifold-Consistent Spectral Surprisal | Catches "confident lies" |
| `v31` | Authority Flow | Streaming inference |

## Project Structure

```
src/ag_sar/       # Core library
├── engine.py     # Main AGSAR class
├── config.py     # Configuration
├── measures/     # Uncertainty algorithms
├── modeling/     # Attention extraction
├── ops/          # Backend operations
└── utils/        # Utilities

benchmarks/       # Evaluation framework
├── run.py        # Benchmark runner
├── baselines/    # Competitor implementations
└── configs/      # Experiment configurations

examples/         # Usage demos
tests/            # Test suite
```

## Running Tests

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=ag_sar tests/
```

## Running Benchmarks

```bash
# Main benchmark suite
python benchmarks/run.py --config benchmarks/configs/main_benchmark.yaml

# Ablation study
python benchmarks/run.py --config benchmarks/configs/ablation.yaml
```

## Citation

```bibtex
@article{agsar2025,
  title={AG-SAR: Zero-Latency Hallucination Detection via Attention Graph Analysis},
  author={...},
  year={2025}
}
```

## License

MIT License
