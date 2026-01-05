#!/usr/bin/env python3
"""
TruthfulQA Head-to-Head Benchmark: Baseline vs Universal

Validates the "Universal Hallucination Detection" claim by comparing:
- Baseline (JEPA only): No Truth Vector, relies on semantic dispersion
- Universal (Truth Vector): Dynamic blending with intrinsic trust

Key Hypothesis:
- On TruthfulQA (no context), Gate will be LOW (~0)
- Baseline should struggle (random or low performance)
- Universal should WIN significantly because it uses Truth Vector

Usage:
    python scripts/launch_truthfulqa_benchmark.py
    python scripts/launch_truthfulqa_benchmark.py --samples 1000
"""

import argparse
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from datetime import datetime


def create_config(enable_intrinsic: bool, num_samples: int, output_name: str) -> dict:
    """Create benchmark config with specified intrinsic detection setting."""
    return {
        "experiment": {
            "name": output_name,
            "description": f"TruthfulQA {'Universal' if enable_intrinsic else 'Baseline'}"
        },
        "dataset": {
            "name": "truthfulqa",
            "num_samples": num_samples,
            "seed": 42
        },
        "model": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "attn_implementation": "eager",
            "dtype": "bfloat16",
            "device_map": "cuda:0",
            "trust_remote_code": True
        },
        "methods": {
            "agsar": {
                "semantic_layers": 4,
                "power_iteration_steps": 3,
                "residual_weight": 0.5,
                "enable_unified_gating": True,
                "stability_sensitivity": 1.0,
                "parametric_weight": 0.5,
                "enable_semantic_dispersion": True,
                "dispersion_k": 5,
                "dispersion_sensitivity": 1.0,
                "dispersion_method": "top1_projection",
                # The key difference:
                "enable_intrinsic_detection": enable_intrinsic,
                "truth_vector_path": "data/truth_vectors/llama_3.1_8b_instruct.pt" if enable_intrinsic else None
            },
            "logprob": True,
            "entropy": True
        },
        "evaluation": {
            "confidence_threshold": 0.7,
            "metrics": ["auroc", "auprc", "f1", "tpr_at_5fpr", "ece", "brier"],
            "bootstrap_samples": 1000,
            "confidence_level": 0.95
        },
        "output": {
            "output_dir": f"results/truthfulqa_h2h/{output_name}",
            "save_predictions": True,
            "save_scores": True
        }
    }


def run_experiment(config: dict, config_path: Path) -> dict:
    """Run experiment and return results."""
    # Write config to temp file
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Run experiment
    cmd = [
        sys.executable, "-m", "experiments.main",
        "--config", str(config_path)
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config['experiment']['name']}")
    print(f"Intrinsic Detection: {config['methods']['agsar']['enable_intrinsic_detection']}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="TruthfulQA H2H Benchmark")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--universal-only", action="store_true", help="Run universal only")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("TruthfulQA Head-to-Head Benchmark")
    print("Baseline (JEPA) vs Universal (Truth Vector)")
    print("="*60)
    print(f"\nSamples: {args.samples}")
    print(f"Timestamp: {timestamp}")

    results = {}

    # Run Baseline (no Truth Vector)
    if not args.universal_only:
        baseline_config = create_config(
            enable_intrinsic=False,
            num_samples=args.samples,
            output_name=f"baseline_{timestamp}"
        )
        baseline_path = Path(f"experiments/configs/_tmp_baseline_{timestamp}.yaml")
        rc = run_experiment(baseline_config, baseline_path)
        results["baseline"] = rc
        baseline_path.unlink(missing_ok=True)  # Cleanup

    # Run Universal (with Truth Vector)
    if not args.baseline_only:
        universal_config = create_config(
            enable_intrinsic=True,
            num_samples=args.samples,
            output_name=f"universal_{timestamp}"
        )
        universal_path = Path(f"experiments/configs/_tmp_universal_{timestamp}.yaml")
        rc = run_experiment(universal_config, universal_path)
        results["universal"] = rc
        universal_path.unlink(missing_ok=True)  # Cleanup

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: results/truthfulqa_h2h/")
    print("\nTo compare results:")
    print(f"  cat results/truthfulqa_h2h/baseline_{timestamp}/*.jsonl | grep AUROC")
    print(f"  cat results/truthfulqa_h2h/universal_{timestamp}/*.jsonl | grep AUROC")


if __name__ == "__main__":
    main()
