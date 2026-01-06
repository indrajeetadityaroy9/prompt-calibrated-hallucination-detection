#!/usr/bin/env python3
"""
Multi-Model Mechanism Comparison Script.

Compares AG-SAR mechanism signatures across different models to test:
1. Scaling Hypothesis: Do larger models ignore context more (lower Gate)?
2. Architecture Hypothesis: Do different families have different defaults?

Usage:
    python experiments/analysis/compare_models.py

Expects results in:
    results/mechanism_sweep/llama8b/
    results/mechanism_sweep/llama70b/
    results/mechanism_sweep/mistral7b/
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import math


@dataclass
class ModelStats:
    """Statistics for a single model."""
    name: str
    short_name: str
    n_hallucinations: int
    n_factual: int
    auroc: float
    # Hallucination stats
    hall_gate_mean: float
    hall_gate_std: float
    hall_disp_mean: float
    hall_disp_std: float
    # Factual stats
    fact_gate_mean: float
    fact_gate_std: float
    fact_disp_mean: float
    fact_disp_std: float


def find_latest_jsonl(directory: str) -> Optional[str]:
    """Find the most recent JSONL file in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return None
    jsonl_files = list(dir_path.glob("*.jsonl"))
    if not jsonl_files:
        return None
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(jsonl_files[0])


def compute_stats(values: List[float]) -> tuple:
    """Compute mean and std of a list."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    std = math.sqrt(variance)
    return mean, std


def load_model_results(results_dir: str, model_name: str, short_name: str) -> Optional[ModelStats]:
    """Load and compute statistics for a single model."""
    jsonl_path = find_latest_jsonl(results_dir)
    if not jsonl_path:
        print(f"  Warning: No results found in {results_dir}")
        return None

    print(f"  Loading {short_name} from {jsonl_path}")

    hall_gates = []
    hall_disps = []
    fact_gates = []
    fact_disps = []
    auroc = 0.0

    with open(jsonl_path, "r") as f:
        for line in f:
            row = json.loads(line)

            # Extract AUROC from summary
            if row.get("_type") == "summary":
                auroc = row.get("metrics", {}).get("auroc", 0.0)
                continue

            # Skip non-sample rows
            if "method" not in row or row.get("method") != "AG-SAR":
                continue

            extra = row.get("extra", {})
            gate = extra.get("gate")
            disp = extra.get("dispersion")

            if gate is None:
                continue

            if row["label"] == 1:  # Hallucination
                hall_gates.append(gate)
                hall_disps.append(disp if disp is not None else 0.0)
            else:  # Factual
                fact_gates.append(gate)
                fact_disps.append(disp if disp is not None else 0.0)

    if not hall_gates:
        print(f"  Warning: No hallucination samples found for {short_name}")
        return None

    hall_gate_mean, hall_gate_std = compute_stats(hall_gates)
    hall_disp_mean, hall_disp_std = compute_stats(hall_disps)
    fact_gate_mean, fact_gate_std = compute_stats(fact_gates)
    fact_disp_mean, fact_disp_std = compute_stats(fact_disps)

    return ModelStats(
        name=model_name,
        short_name=short_name,
        n_hallucinations=len(hall_gates),
        n_factual=len(fact_gates),
        auroc=auroc,
        hall_gate_mean=hall_gate_mean,
        hall_gate_std=hall_gate_std,
        hall_disp_mean=hall_disp_mean,
        hall_disp_std=hall_disp_std,
        fact_gate_mean=fact_gate_mean,
        fact_gate_std=fact_gate_std,
        fact_disp_mean=fact_disp_mean,
        fact_disp_std=fact_disp_std,
    )


def welch_t_test(mean1: float, std1: float, n1: int,
                  mean2: float, std2: float, n2: int) -> float:
    """
    Compute Welch's t-statistic for two independent samples.
    Returns the t-statistic (positive = mean1 > mean2).
    """
    if n1 < 2 or n2 < 2:
        return 0.0
    se1 = (std1 ** 2) / n1
    se2 = (std2 ** 2) / n2
    se_diff = math.sqrt(se1 + se2)
    if se_diff < 1e-10:
        return 0.0
    t_stat = (mean1 - mean2) / se_diff
    return t_stat


def compare_models(base_dir: str = "results/mechanism_sweep") -> None:
    """Compare mechanism signatures across models."""

    print("=" * 70)
    print(" MULTI-MODEL MECHANISM COMPARISON")
    print("=" * 70)
    print()

    # Define models to compare
    models_config = [
        ("llama8b", "meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
        ("llama70b", "meta-llama/Llama-3.1-70B-Instruct", "Llama-3.1-70B"),
        ("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B"),
    ]

    print("Loading results...")
    stats_list = []
    for subdir, model_name, short_name in models_config:
        results_dir = f"{base_dir}/{subdir}"
        stats = load_model_results(results_dir, model_name, short_name)
        if stats:
            stats_list.append(stats)

    if len(stats_list) < 2:
        print("\nError: Need at least 2 models for comparison.")
        print("Run the experiments first:")
        print("  python -m experiments.main --config experiments/configs/mechanism_sweep_8b.yaml")
        print("  python -m experiments.main --config experiments/configs/mechanism_sweep_70b.yaml")
        print("  python -m experiments.main --config experiments/configs/mechanism_sweep_mistral.yaml")
        return

    # Print comparison table
    print()
    print("=" * 70)
    print(" MECHANISM SIGNATURES BY MODEL (Hallucinations Only)")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'AUROC':<10} {'Gate (α)':<18} {'Dispersion (β)':<18} {'N':<8}")
    print("-" * 70)

    for s in stats_list:
        gate_str = f"{s.hall_gate_mean:.3f} ± {s.hall_gate_std:.3f}"
        disp_str = f"{s.hall_disp_mean:.3f} ± {s.hall_disp_std:.3f}"
        print(f"{s.short_name:<20} {s.auroc:<10.4f} {gate_str:<18} {disp_str:<18} {s.n_hallucinations:<8}")

    print("-" * 70)
    print()

    # Statistical tests
    print("=" * 70)
    print(" HYPOTHESIS TESTING")
    print("=" * 70)
    print()

    # Find specific models for comparisons
    llama8b = next((s for s in stats_list if "8B" in s.short_name and "Llama" in s.short_name), None)
    llama70b = next((s for s in stats_list if "70B" in s.short_name), None)
    mistral = next((s for s in stats_list if "Mistral" in s.short_name), None)

    # 1. Scaling Hypothesis: 70B vs 8B
    if llama8b and llama70b:
        print("1. SCALING HYPOTHESIS (Llama-3.1-70B vs 8B)")
        print("-" * 50)

        t_gate = welch_t_test(
            llama70b.hall_gate_mean, llama70b.hall_gate_std, llama70b.n_hallucinations,
            llama8b.hall_gate_mean, llama8b.hall_gate_std, llama8b.n_hallucinations
        )
        t_disp = welch_t_test(
            llama70b.hall_disp_mean, llama70b.hall_disp_std, llama70b.n_hallucinations,
            llama8b.hall_disp_mean, llama8b.hall_disp_std, llama8b.n_hallucinations
        )

        gate_diff = llama70b.hall_gate_mean - llama8b.hall_gate_mean
        disp_diff = llama70b.hall_disp_mean - llama8b.hall_disp_mean

        print(f"  Gate Difference:       {gate_diff:+.4f} (t = {t_gate:.2f})")
        print(f"  Dispersion Difference: {disp_diff:+.4f} (t = {t_disp:.2f})")
        print()

        if gate_diff < -0.02 and abs(t_gate) > 2.0:
            print("  → CONFIRMED: 70B is MORE STUBBORN (lower Gate)")
            print("    Larger models rely more on parametric memory.")
        elif gate_diff > 0.02 and abs(t_gate) > 2.0:
            print("  → REVERSED: 70B is LESS STUBBORN (higher Gate)")
            print("    Larger models may have better context integration.")
        else:
            print("  → INCONCLUSIVE: No significant difference in Gate")
        print()

    # 2. Architecture Hypothesis: Mistral vs Llama-8B
    if llama8b and mistral:
        print("2. ARCHITECTURE HYPOTHESIS (Mistral-7B vs Llama-8B)")
        print("-" * 50)

        t_gate = welch_t_test(
            mistral.hall_gate_mean, mistral.hall_gate_std, mistral.n_hallucinations,
            llama8b.hall_gate_mean, llama8b.hall_gate_std, llama8b.n_hallucinations
        )
        t_disp = welch_t_test(
            mistral.hall_disp_mean, mistral.hall_disp_std, mistral.n_hallucinations,
            llama8b.hall_disp_mean, llama8b.hall_disp_std, llama8b.n_hallucinations
        )

        gate_diff = mistral.hall_gate_mean - llama8b.hall_gate_mean
        disp_diff = mistral.hall_disp_mean - llama8b.hall_disp_mean

        print(f"  Gate Difference:       {gate_diff:+.4f} (t = {t_gate:.2f})")
        print(f"  Dispersion Difference: {disp_diff:+.4f} (t = {t_disp:.2f})")
        print()

        if abs(gate_diff) > 0.05 and abs(t_gate) > 2.0:
            if gate_diff > 0:
                print("  → Mistral is MORE CONTEXT-RELIANT (higher Gate)")
            else:
                print("  → Mistral is MORE STUBBORN (lower Gate)")
            print("    Architecture significantly affects context reliance.")
        else:
            print("  → SIMILAR: No major architectural difference in Gate")
        print()

    # Summary for paper
    print("=" * 70)
    print(" SUMMARY FOR PAPER")
    print("=" * 70)
    print()

    if len(stats_list) >= 2:
        # Sort by gate (ascending = more stubborn first)
        sorted_by_gate = sorted(stats_list, key=lambda s: s.hall_gate_mean)
        most_stubborn = sorted_by_gate[0]
        least_stubborn = sorted_by_gate[-1]

        print(f"  Most Stubborn Model:  {most_stubborn.short_name} (Gate = {most_stubborn.hall_gate_mean:.3f})")
        print(f"  Least Stubborn Model: {least_stubborn.short_name} (Gate = {least_stubborn.hall_gate_mean:.3f})")
        print()

        # Sort by dispersion (descending = most confused first)
        sorted_by_disp = sorted(stats_list, key=lambda s: s.hall_disp_mean, reverse=True)
        most_confused = sorted_by_disp[0]
        least_confused = sorted_by_disp[-1]

        print(f"  Most Confused Model:  {most_confused.short_name} (Dispersion = {most_confused.hall_disp_mean:.3f})")
        print(f"  Least Confused Model: {least_confused.short_name} (Dispersion = {least_confused.hall_disp_mean:.3f})")

    print()
    print("=" * 70)


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "results/mechanism_sweep"
    compare_models(base_dir)
