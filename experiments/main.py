#!/usr/bin/env python3
"""
AG-SAR Experiments CLI - Single entry point for all experiments.

This is the unified interface for running hallucination detection experiments.
It replaces the old scattered scripts (run.py, compare_baselines.py, etc.).

Usage:
    # Run a full experiment
    python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml

    # Dry run (print config and exit)
    python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --dry-run

    # Override output directory
    python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --output-dir results/custom

    # Reproducibility mode (deterministic, overridden seed)
    python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --seed 123 --deterministic
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Pre-flight installation check (must be first to ensure ag_sar is importable)
from experiments.utils.preflight import check_installation
check_installation()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import enable_h100_optimizations, get_optimal_dtype
from experiments.evaluation.determinism import set_global_seed, DEFAULT_SEED

# Note: Global seed is set in main() after parsing CLI args to support --seed override

from experiments.configs.schema import ExperimentConfig
from experiments.evaluation.engine import BenchmarkEngine
from experiments.data.halueval import HaluEvalDataset
from experiments.data.ragtruth import RAGTruthDataset
from experiments.data.truthfulqa import TruthfulQADataset
from experiments.data.wikitext import WikiTextDataset
from experiments.data.fava import FAVADataset
from experiments.methods.agsar_wrapper import AGSARMethod
from experiments.methods.logprob import LogProbMethod
from experiments.methods.entropy import PredictiveEntropyMethod
from experiments.methods.selfcheck import SelfCheckNLIMethod
from experiments.methods.eigenscore import EigenScoreMethod, SAPLMAMethod
from experiments.methods.llm_check import (
    LLMCheckAttentionMethod,
    LLMCheckHiddenMethod,
    LLMCheckLogitMethod,
)
from experiments.methods.semantic_entropy import SemanticEntropyMethod


def load_model_and_tokenizer(config: ExperimentConfig, deterministic: bool = False):
    """
    Load model with H100 optimizations.

    Args:
        config: Experiment configuration
        deterministic: Enable deterministic mode for reproducibility

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model.name}")

    # Enable optimizations (with optional determinism)
    enable_h100_optimizations(deterministic=deterministic)

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

    # Special case: Load ALL 4 required datasets
    if ds_config.name == "ALL":
        num_samples = ds_config.num_samples
        seed = ds_config.seed

        datasets["halueval_qa"] = HaluEvalDataset(
            variant="qa", num_samples=num_samples, seed=seed
        )
        datasets["ragtruth"] = RAGTruthDataset(
            task_type="QA", num_samples=num_samples, seed=seed
        )
        datasets["halueval_summarization"] = HaluEvalDataset(
            variant="summarization", num_samples=num_samples, seed=seed
        )
        datasets["fava"] = FAVADataset(
            num_samples=num_samples, seed=seed
        )
        return datasets

    # Single dataset loading (original logic)
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

    elif ds_config.name == "truthfulqa":
        datasets["truthfulqa"] = TruthfulQADataset(
            num_samples=ds_config.num_samples,
            seed=ds_config.seed,
        )

    elif ds_config.name == "wikitext":
        datasets["wikitext"] = WikiTextDataset(
            num_samples=ds_config.num_samples,
            seed=ds_config.seed,
        )

    elif ds_config.name == "fava":
        datasets["fava"] = FAVADataset(
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
        methods["SelfCheck"] = SelfCheckNLIMethod(model, tokenizer, config=mc.selfcheck)
        print(f"  [+] SelfCheck-NLI enabled (samples={mc.selfcheck.num_samples})")

    if mc.eigenscore:
        methods["EigenScore"] = EigenScoreMethod(
            model,
            tokenizer,
            num_samples=mc.eigenscore.num_samples,
            max_new_tokens=mc.eigenscore.max_new_tokens,
            temperature=mc.eigenscore.temperature,
        )
        print(f"  [+] EigenScore enabled (samples={mc.eigenscore.num_samples})")

    if mc.semantic_entropy:
        methods["SemanticEntropy"] = SemanticEntropyMethod(
            model,
            tokenizer,
            num_samples=mc.semantic_entropy.num_samples,
            similarity_threshold=mc.semantic_entropy.similarity_threshold,
            embedding_model=mc.semantic_entropy.embedding_model,
            max_new_tokens=mc.semantic_entropy.max_new_tokens,
            temperature=mc.semantic_entropy.temperature,
        )
        print(f"  [+] SemanticEntropy enabled (samples={mc.semantic_entropy.num_samples})")

    if mc.saplma:
        methods["SAPLMA"] = SAPLMAMethod(model, tokenizer)
        print("  [+] SAPLMA enabled")

    # LLM-Check methods (NeurIPS 2024)
    if mc.llmcheck_attn:
        methods["LLMCheck-Attn"] = LLMCheckAttentionMethod(model, tokenizer)
        print("  [+] LLMCheck-Attn enabled")

    if mc.llmcheck_hidden:
        methods["LLMCheck-Hidden"] = LLMCheckHiddenMethod(model, tokenizer)
        print("  [+] LLMCheck-Hidden enabled")

    if mc.llmcheck_logit:
        methods["LLMCheck-Logit"] = LLMCheckLogitMethod(model, tokenizer)
        print("  [+] LLMCheck-Logit enabled")

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
  python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml
  python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --dry-run
  python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --seed 123 --deterministic
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Override random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for bit-exact reproducibility (may reduce performance)",
    )

    args = parser.parse_args()

    # Set global seed BEFORE any model/data loading
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    set_global_seed(seed, deterministic=args.deterministic)

    if args.deterministic:
        print(f"[Deterministic mode enabled] Seed: {seed}")

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
        model, tokenizer = load_model_and_tokenizer(config, deterministic=args.deterministic)
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
