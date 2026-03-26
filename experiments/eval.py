#!/usr/bin/env python3
import argparse

from experiments.common import load_model
from experiments.run_ablation import run_ablation
from experiments.run_eval import run_evaluation
from experiments.schema import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Experiment Runner")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    model, tokenizer = load_model(config.model, config.evaluation.seed)

    if config.mode == "evaluation":
        run_evaluation(model, tokenizer, config)
    else:
        run_ablation(model, tokenizer, config)


if __name__ == "__main__":
    main()
