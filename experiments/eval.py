#!/usr/bin/env python3
"""
AG-SAR Experiment Entrypoint.

Single config-only CLI:
    python -m experiments.eval --config experiments/configs/main.yaml
    python -m experiments.eval --config experiments/configs/ablation.yaml
"""

import argparse

from .schema import ExperimentConfig
from .common import load_model


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Experiment Runner")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    model, tokenizer = load_model(config.model, config.evaluation.seed)

    if config.run.mode == "evaluation":
        from .run_eval import run_evaluation
        run_evaluation(model, tokenizer, config)
    elif config.run.mode == "ablation":
        from .run_ablation import run_ablation
        run_ablation(model, tokenizer, config)
    else:
        raise ValueError(f"Unknown run mode: {config.run.mode!r}. Expected 'evaluation' or 'ablation'.")


if __name__ == "__main__":
    main()
