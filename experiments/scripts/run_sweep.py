#!/usr/bin/env python3
"""
Parameter sweep runner with runtime config overrides.

Usage:
    python -m experiments.scripts.run_sweep \
        --base-config experiments/configs/01_main_sota.yaml \
        --param methods.agsar.stability_sensitivity \
        --values 0.5 1.0 2.0 5.0 \
        --output-dir results/sweep_stability

    # With sample count override
    python -m experiments.scripts.run_sweep \
        --base-config experiments/configs/01_main_sota.yaml \
        --param methods.agsar.dispersion_k \
        --values 3 5 10 \
        --num-samples 500 \
        --output-dir results/sweep_dispersion
"""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Union

import yaml


def deep_update(config: dict, path: str, value: Any) -> None:
    """
    Updates nested dict/list using dot-notation path.

    Example: path="methods.agsar.stability_sensitivity"
    Handles both dict keys and list indices (numeric keys).

    Args:
        config: Config dictionary to modify in-place
        path: Dot-separated path to the target key
        value: New value to set
    """
    keys = path.split(".")
    current = config

    for k in keys[:-1]:
        if k.isdigit():
            k = int(k)
        current = current[k]

    last_key = keys[-1]
    if last_key.isdigit():
        current[int(last_key)] = value
    else:
        current[last_key] = value


def parse_value(value_str: str) -> Union[int, float, bool, str]:
    """
    Parse a string value to appropriate Python type.

    Args:
        value_str: String representation of value

    Returns:
        Parsed value (int, float, bool, or str)
    """
    # Try int first
    try:
        return int(value_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(value_str)
    except ValueError:
        pass

    # Try bool
    if value_str.lower() in ("true", "yes", "1"):
        return True
    if value_str.lower() in ("false", "no", "0"):
        return False

    # Return as string
    return value_str


def run_single_experiment(config_dict: dict, output_dir: Path) -> dict:
    """
    Run a single experiment with the given config.

    Args:
        config_dict: Complete experiment config as dict
        output_dir: Output directory for results

    Returns:
        Summary dict with metrics
    """
    # Import here to avoid circular imports and allow proper path setup
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Add src to path for ag_sar imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

    from ag_sar import enable_h100_optimizations, get_optimal_dtype
    from experiments.core.determinism import set_global_seed, DEFAULT_SEED
    from experiments.configs.schema import ExperimentConfig
    from experiments.core.engine import BenchmarkEngine
    from experiments.data.halueval import HaluEvalDataset
    from experiments.data.ragtruth import RAGTruthDataset
    from experiments.data.truthfulqa import TruthfulQADataset
    from experiments.data.wikitext import WikiTextDataset
    from experiments.methods.agsar_wrapper import AGSARMethod
    from experiments.methods.logprob import LogProbMethod
    from experiments.methods.entropy import PredictiveEntropyMethod
    from experiments.methods.selfcheck import SelfCheckNLIMethod
    from experiments.methods.eigenscore import EigenScoreMethod, SAPLMAMethod

    # Set seed
    set_global_seed(DEFAULT_SEED)

    # Create config object
    config = ExperimentConfig(**config_dict)

    # Override output directory
    config.output.output_dir = str(output_dir)

    # Enable H100 optimizations
    enable_h100_optimizations()

    # Load model
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype, get_optimal_dtype())

    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=dtype,
        device_map=config.model.device_map,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
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

    for ds in datasets.values():
        ds.load()

    # Build methods
    methods = {}
    mc = config.methods

    if mc.agsar:
        methods["AG-SAR"] = AGSARMethod(model, tokenizer, config=mc.agsar)

    if mc.logprob:
        methods["LogProb"] = LogProbMethod(model, tokenizer)

    if mc.entropy:
        methods["Entropy"] = PredictiveEntropyMethod(model, tokenizer)

    if mc.selfcheck:
        methods["SelfCheck"] = SelfCheckNLIMethod(model, tokenizer, config=mc.selfcheck)

    if mc.eigenscore:
        methods["EigenScore"] = EigenScoreMethod(
            model,
            tokenizer,
            num_samples=mc.eigenscore.num_samples,
            max_new_tokens=mc.eigenscore.max_new_tokens,
            temperature=mc.eigenscore.temperature,
        )

    if mc.saplma:
        methods["SAPLMA"] = SAPLMAMethod(model, tokenizer)

    # Run benchmark
    engine = BenchmarkEngine(config, methods, datasets)
    results = engine.run()

    # Cleanup
    for method in methods.values():
        try:
            method.cleanup()
        except Exception:
            pass

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    """Main entry point for sweep runner."""
    parser = argparse.ArgumentParser(
        description="AG-SAR Parameter Sweep Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.scripts.run_sweep \\
      --base-config experiments/configs/01_main_sota.yaml \\
      --param methods.agsar.stability_sensitivity \\
      --values 0.5 1.0 2.0 5.0 \\
      --output-dir results/sweep_stability

  python -m experiments.scripts.run_sweep \\
      --base-config experiments/configs/01_main_sota.yaml \\
      --param methods.agsar.dispersion_k \\
      --values 3 5 10 \\
      --num-samples 500 \\
      --output-dir results/sweep_dispersion
        """,
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to base experiment config YAML file",
    )
    parser.add_argument(
        "--param",
        type=str,
        required=True,
        help="Dot-notation path to parameter to sweep (e.g., methods.agsar.stability_sensitivity)",
    )
    parser.add_argument(
        "--values",
        type=str,
        nargs="+",
        required=True,
        help="Values to sweep over",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for sweep results",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of samples (for faster sweeps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configs without running experiments",
    )

    args = parser.parse_args()

    # Load base config
    base_config_path = Path(args.base_config)
    if not base_config_path.exists():
        print(f"Error: Config file not found: {base_config_path}")
        return 1

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    # Parse values
    values = [parse_value(v) for v in args.values]

    # Extract param name for directory naming
    param_name = args.param.split(".")[-1]

    print("=" * 60)
    print("AG-SAR Parameter Sweep")
    print("=" * 60)
    print(f"Base config: {args.base_config}")
    print(f"Parameter: {args.param}")
    print(f"Values: {values}")
    print(f"Output dir: {args.output_dir}")
    if args.num_samples:
        print(f"Sample override: {args.num_samples}")
    print("=" * 60)

    # Create output directory
    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Track results for aggregation
    sweep_results: List[dict] = []

    # Run experiments for each value
    for i, value in enumerate(values):
        print(f"\n[{i+1}/{len(values)}] Running with {param_name}={value}")
        print("-" * 60)

        # Deep copy config and apply override
        config = copy.deepcopy(base_config)
        deep_update(config, args.param, value)

        # Override num_samples if specified
        if args.num_samples:
            config["dataset"]["num_samples"] = args.num_samples

        # Create unique output directory for this value
        value_str = str(value).replace(".", "_")
        output_dir = output_base / f"{param_name}_{value_str}"
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.dry_run:
            print(f"Would run experiment with {args.param}={value}")
            print(f"Output: {output_dir}")
            # Print modified config section
            print(f"Config snippet:")
            print(yaml.dump({args.param.split(".")[0]: config[args.param.split(".")[0]]}, default_flow_style=False))
            continue

        try:
            results = run_single_experiment(config, output_dir)

            # Extract AG-SAR results for summary
            agsar_results = {}
            for ds_name, ds_results in results.items():
                if "AG-SAR" in ds_results:
                    agsar_results = ds_results["AG-SAR"]
                    break

            sweep_results.append({
                "param": args.param,
                "value": value,
                "auroc": agsar_results.get("auroc", None),
                "auprc": agsar_results.get("auprc", None),
                "ece": agsar_results.get("ece", None),
                "output_dir": str(output_dir),
            })

            print(f"  AUROC: {agsar_results.get('auroc', 'N/A'):.4f}")
            if "auprc" in agsar_results:
                print(f"  AUPRC: {agsar_results.get('auprc', 'N/A'):.4f}")
            if "ece" in agsar_results:
                print(f"  ECE: {agsar_results.get('ece', 'N/A'):.4f}")

        except Exception as e:
            print(f"Error running experiment: {e}")
            import traceback
            traceback.print_exc()
            sweep_results.append({
                "param": args.param,
                "value": value,
                "auroc": None,
                "auprc": None,
                "ece": None,
                "error": str(e),
                "output_dir": str(output_dir),
            })

    if args.dry_run:
        print("\nDry run complete. No experiments were run.")
        return 0

    # Write summary CSV
    summary_path = output_base / "sweep_summary.csv"
    if sweep_results:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
            writer.writeheader()
            writer.writerows(sweep_results)
        print(f"\nSweep summary saved to: {summary_path}")

    # Write summary JSON
    summary_json_path = output_base / "sweep_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump({
            "param": args.param,
            "values": values,
            "results": sweep_results,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Value':<12} {'AUROC':<10} {'AUPRC':<10} {'ECE':<10}")
    print("-" * 42)
    for r in sweep_results:
        auroc = f"{r['auroc']:.4f}" if r['auroc'] else "ERROR"
        auprc = f"{r['auprc']:.4f}" if r['auprc'] else "N/A"
        ece = f"{r['ece']:.4f}" if r['ece'] else "N/A"
        print(f"{r['value']:<12} {auroc:<10} {auprc:<10} {ece:<10}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
