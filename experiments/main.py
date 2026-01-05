#!/usr/bin/env python3
"""
AG-SAR Experiments CLI - Single entry point for all experiments.

This is the unified interface for running hallucination detection experiments.
It replaces the old scattered scripts (run.py, compare_baselines.py, etc.).

Usage:
    # Run a full experiment
    python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml

    # Dry run (print config and exit)
    python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml --dry-run

    # Override output directory
    python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml --output-dir results/custom
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for ag_sar imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ag_sar import enable_h100_optimizations, get_optimal_dtype

# Phase 0.2: Set global seed BEFORE any model or data loading
from experiments.core.determinism import set_global_seed, DEFAULT_SEED
set_global_seed(DEFAULT_SEED)

from experiments.configs.schema import ExperimentConfig
from experiments.core.engine import BenchmarkEngine
from experiments.data.halueval import HaluEvalDataset
from experiments.data.ragtruth import RAGTruthDataset
from experiments.methods.agsar_wrapper import AGSARMethod
from experiments.methods.logprob import LogProbMethod
from experiments.methods.entropy import PredictiveEntropyMethod
from experiments.methods.selfcheck import SelfCheckNgramMethod
from experiments.methods.eigenscore import EigenScoreMethod, SAPLMAMethod


def load_model_and_tokenizer(config: ExperimentConfig):
    """
    Load model with H100 optimizations.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model.name}")

    # Enable optimizations
    enable_h100_optimizations()

    # Determine dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype, get_optimal_dtype())

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        device_map=config.model.device_map,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def build_datasets(config: ExperimentConfig) -> Dict:
    """
    Build dataset instances from config.

    Args:
        config: Experiment configuration

    Returns:
        Dict of {dataset_name: EvaluationDataset}
    """
    datasets = {}
    ds_config = config.dataset

    if ds_config.name.startswith("halueval_"):
        variant = ds_config.name.replace("halueval_", "")
        datasets[ds_config.name] = HaluEvalDataset(
            variant=variant,
            num_samples=ds_config.num_samples,
            seed=ds_config.seed,
        )

    elif ds_config.name == "ragtruth":
        datasets["ragtruth"] = RAGTruthDataset(
            task_type=ds_config.task_type,
            num_samples=ds_config.num_samples,
            seed=ds_config.seed,
        )

    else:
        raise ValueError(f"Unknown dataset: {ds_config.name}")

    return datasets


def build_methods(config: ExperimentConfig, model, tokenizer) -> Dict:
    """
    Build method instances from config.

    Args:
        config: Experiment configuration
        model: Loaded language model
        tokenizer: Loaded tokenizer

    Returns:
        Dict of {method_name: UncertaintyMethod}
    """
    methods = {}
    mc = config.methods

    if mc.agsar:
        methods["AG-SAR"] = AGSARMethod(model, tokenizer, config=mc.agsar)
        print("  [+] AG-SAR enabled")

    if mc.logprob:
        methods["LogProb"] = LogProbMethod(model, tokenizer)
        print("  [+] LogProb enabled")

    if mc.entropy:
        methods["Entropy"] = PredictiveEntropyMethod(model, tokenizer)
        print("  [+] Entropy enabled")

    if mc.selfcheck:
        methods["SelfCheck"] = SelfCheckNgramMethod(model, tokenizer, config=mc.selfcheck)
        print(f"  [+] SelfCheck enabled (samples={mc.selfcheck.num_samples})")

    if mc.eigenscore:
        methods["EigenScore"] = EigenScoreMethod(model, tokenizer)
        print("  [+] EigenScore enabled")

    if mc.saplma:
        methods["SAPLMA"] = SAPLMAMethod(model, tokenizer)
        print("  [+] SAPLMA enabled")

    if not methods:
        raise ValueError("No methods enabled in config. Enable at least one method.")

    return methods


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AG-SAR Experiments CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml
  python -m experiments.main --config experiments/configs/exp1_halueval_qa.yaml --dry-run
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without running",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = ExperimentConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Override output dir if specified
    if args.output_dir:
        config.output.output_dir = args.output_dir

    # Dry run mode
    if args.dry_run:
        import yaml

        print("=" * 60)
        print("DRY RUN - Configuration:")
        print("=" * 60)
        print(yaml.dump(config.model_dump(), default_flow_style=False, sort_keys=False))
        print("=" * 60)
        print(f"Enabled methods: {config.get_enabled_methods()}")
        print("=" * 60)
        return 0

    # Print header
    print("=" * 60)
    print("AG-SAR Experiment Runner")
    print("=" * 60)
    print(f"Experiment: {config.experiment.get('name', 'unnamed')}")
    print(f"Description: {config.experiment.get('description', '')}")
    print(f"Model: {config.model.name}")
    print(f"Dataset: {config.dataset.name}")
    print(f"Output: {config.output.output_dir}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading model...")
    try:
        model, tokenizer = load_model_and_tokenizer(config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Build datasets
    print("\n[2/4] Preparing datasets...")
    try:
        datasets = build_datasets(config)
        for name, ds in datasets.items():
            ds.load()
            stats = ds.get_statistics()
            print(f"  {name}: {stats['total_samples']} samples")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return 1

    # Build methods
    print("\n[3/4] Initializing methods...")
    try:
        methods = build_methods(config, model, tokenizer)
    except Exception as e:
        print(f"Error initializing methods: {e}")
        return 1

    # Run benchmark
    print("\n[4/4] Running benchmark...")
    print("-" * 60)

    try:
        engine = BenchmarkEngine(config, methods, datasets)
        results = engine.run()

        # Print summary table
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(engine.get_results_table())

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        # Cleanup all methods
        print("\nCleaning up...")
        for method in methods.values():
            try:
                method.cleanup()
            except Exception:
                pass

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
