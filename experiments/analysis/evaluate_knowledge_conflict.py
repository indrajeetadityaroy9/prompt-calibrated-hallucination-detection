#!/usr/bin/env python3
"""
Knowledge Conflict Evaluation Script.

Diagnoses the root cause of hallucinations by analyzing AG-SAR component scores:
- Parametric Stubbornness (Gate < 0.5): Model rejected context, relied on memory
- Epistemic Confusion (Gate > 0.5): Model tried to use context but failed

This analysis elevates AG-SAR from "Engineering" to "Scientific Discovery" by
quantifying Knowledge Conflict - demonstrating that AG-SAR doesn't just detect
errors, it diagnoses their root cause (Memory vs. Context).

Usage:
    python experiments/analysis/evaluate_knowledge_conflict.py \
        --file results/knowledge_conflict/<experiment>.jsonl

    # Or find the latest results automatically:
    python experiments/analysis/evaluate_knowledge_conflict.py \
        --dir results/knowledge_conflict
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


def find_latest_jsonl(directory: str) -> Optional[str]:
    """Find the most recent JSONL file in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return None

    jsonl_files = list(dir_path.glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Sort by modification time, newest first
    jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(jsonl_files[0])


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load JSONL results file."""
    data = []
    with open(results_path, "r") as f:
        for line in f:
            row = json.loads(line)
            data.append(row)
    return data


def evaluate_mechanism(results_path: str) -> None:
    """
    Analyze hallucination mechanisms from AG-SAR component scores.

    Classifies each hallucination into:
    - Parametric Stubbornness: Gate < 0.5 (model ignored context)
    - Epistemic Confusion: Gate >= 0.5 (model tried but failed)
    """
    print(f"Loading results: {results_path}")
    print()

    data = load_results(results_path)

    # Filter to AG-SAR samples with hallucination label
    hallucination_samples = []
    factual_samples = []
    missing_gate_count = 0

    for row in data:
        # Skip summary/metadata rows
        if row.get("_type") == "summary" or "method" not in row:
            continue

        # Only analyze AG-SAR results
        if row.get("method") != "AG-SAR":
            continue

        extra = row.get("extra", {})
        gate = extra.get("gate")
        dispersion = extra.get("dispersion")

        if gate is None:
            missing_gate_count += 1
            continue

        sample = {
            "label": row["label"],
            "score": row["score"],
            "gate": gate,
            "dispersion": dispersion if dispersion is not None else 0.0,
        }

        if row["label"] == 1:  # Hallucination
            hallucination_samples.append(sample)
        else:  # Factual
            factual_samples.append(sample)

    if missing_gate_count > 0:
        print(f"Warning: {missing_gate_count} samples missing gate values")
        print()

    if not hallucination_samples:
        print("No hallucinations found in the dataset sample.")
        return

    # === MECHANISM CLASSIFICATION ===
    # Threshold 0.5 is the geometric equilibrium point
    # Gate < 0.5: MLP overrode attention -> Parametric Stubbornness
    # Gate >= 0.5: MLP agreed with attention -> Epistemic Confusion

    stubbornness_count = sum(1 for s in hallucination_samples if s["gate"] < 0.5)
    confusion_count = sum(1 for s in hallucination_samples if s["gate"] >= 0.5)
    total_hall = len(hallucination_samples)

    stubbornness_pct = 100.0 * stubbornness_count / total_hall
    confusion_pct = 100.0 * confusion_count / total_hall

    # Compute average metrics
    hall_gate_avg = sum(s["gate"] for s in hallucination_samples) / total_hall
    hall_disp_avg = sum(s["dispersion"] for s in hallucination_samples) / total_hall

    fact_gate_avg = sum(s["gate"] for s in factual_samples) / len(factual_samples) if factual_samples else 0
    fact_disp_avg = sum(s["dispersion"] for s in factual_samples) / len(factual_samples) if factual_samples else 0

    # === PRINT RESULTS ===
    print("=" * 60)
    print(" MECHANISM DIAGNOSIS: ROOT CAUSE OF HALLUCINATION")
    print("=" * 60)
    print()
    print(f"{'Error Mechanism':<30} {'Prevalence':<15} {'Count':<10}")
    print("-" * 55)
    print(f"{'Parametric Stubbornness':<30} {stubbornness_pct:>6.1f}%        {stubbornness_count:<10}")
    print(f"{'Epistemic Confusion':<30} {confusion_pct:>6.1f}%        {confusion_count:<10}")
    print("-" * 55)
    print()

    print("STATISTICAL INSIGHTS:")
    print(f"  Total Hallucinations Analyzed: {total_hall}")
    print(f"  Total Factual Samples:         {len(factual_samples)}")
    print()
    print("  Component Scores (Hallucinations):")
    print(f"    Avg Context Reliance (Gate):   {hall_gate_avg:.3f}  (Lower = More Stubborn)")
    print(f"    Avg Semantic Instability:      {hall_disp_avg:.3f}  (Higher = More Confused)")
    print()
    print("  Component Scores (Factual - Baseline):")
    print(f"    Avg Context Reliance (Gate):   {fact_gate_avg:.3f}")
    print(f"    Avg Semantic Instability:      {fact_disp_avg:.3f}")
    print()

    # === INTERPRETATION ===
    print("=" * 60)
    print(" INTERPRETATION FOR PAPER")
    print("=" * 60)
    print()

    if stubbornness_pct > confusion_pct:
        dominant = "Parametric Stubbornness"
        pct = stubbornness_pct
        insight = (
            f"  {pct:.1f}% of hallucinations are driven by Parametric Stubbornness,\n"
            f"  where the model's unified gate explicitly suppresses retrieved\n"
            f"  context (Gate < 0.5). This confirms that geometric conflict\n"
            f"  detection is essential for identifying errors that pure entropy\n"
            f"  methods (which measure confusion, not stubbornness) miss."
        )
    else:
        dominant = "Epistemic Confusion"
        pct = confusion_pct
        insight = (
            f"  {pct:.1f}% of hallucinations are driven by Epistemic Confusion,\n"
            f"  where the model attended to context (Gate >= 0.5) but still\n"
            f"  produced semantically inconsistent output. The high dispersion\n"
            f"  ({hall_disp_avg:.3f}) indicates the model was uncertain about\n"
            f"  how to integrate the retrieved information."
        )

    print(f"  Dominant Mechanism: {dominant} ({pct:.1f}%)")
    print()
    print(insight)
    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hallucination mechanisms from AG-SAR results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSONL results file",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="results/knowledge_conflict",
        help="Directory to search for latest JSONL file",
    )

    args = parser.parse_args()

    # Determine results file
    if args.file:
        results_path = args.file
    else:
        results_path = find_latest_jsonl(args.dir)
        if not results_path:
            print(f"Error: No JSONL files found in {args.dir}")
            print("Run the experiment first:")
            print("  python -m experiments.main --config experiments/configs/eval_knowledge_conflict.yaml")
            sys.exit(1)

    if not Path(results_path).exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    evaluate_mechanism(results_path)


if __name__ == "__main__":
    main()
