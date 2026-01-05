#!/usr/bin/env python3
"""
Analysis Script for Head-to-Head Benchmark Results.

Generates a conference-ready results table with:
- AUROC (Global separability)
- AUPRC (Precision-focused, important for imbalanced data)
- TPR @ 5% FPR (Production metric: catch rate at low false alarm)
- AURC (Selective generation utility)
- Latency (Zero-latency claim verification)

Usage:
    python scripts/print_h2h_table.py --results-dir results/halueval_h2h/20240105_120000
    python scripts/print_h2h_table.py --latest  # Auto-find latest results
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_tpr_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, fpr_target: float = 0.05) -> float:
    """Compute TPR at a specific Fixed False Positive Rate."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    idx = np.argmin(np.abs(fpr - fpr_target))
    return tpr[idx]


def compute_aurc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under Risk-Coverage (AURC).

    Risk = Error rate in the covered subset
    Coverage = Fraction of samples kept (sorted by confidence)

    Lower is better.
    """
    # Sort by uncertainty ascending (most confident = lowest uncertainty first)
    sorted_idx = np.argsort(y_scores)
    sorted_labels = y_true[sorted_idx]
    n = len(sorted_labels)

    risks = []
    for i in range(1, n + 1):
        # Risk = fraction of hallucinations (label=1) in top-i most confident
        risk = sorted_labels[:i].sum() / i
        risks.append(risk)

    coverages = np.arange(1, n + 1) / n
    return np.trapz(risks, coverages)


def load_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load all result files from a results directory."""
    results = []

    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue

        results_file = method_dir / "results.json"
        if not results_file.exists():
            continue

        with open(results_file) as f:
            data = json.load(f)
            results.append(data)

    return results


def analyze_results(results: List[Dict]) -> pd.DataFrame:
    """Analyze results and create summary DataFrame."""
    rows = []

    for res in results:
        method_name = res["method"]
        metrics = res.get("metrics", {})

        # Extract pre-computed metrics or recompute from raw results
        if metrics:
            row = {
                "Method": method_name,
                "AUROC": metrics.get("auroc", 0),
                "AUPRC": metrics.get("auprc", 0),
                "TPR@5%FPR": metrics.get("tpr_at_5fpr", 0),
                "AURC (↓)": metrics.get("aurc", 0),
                "Max F1": metrics.get("max_f1", 0),
                "Latency (ms)": metrics.get("avg_latency_ms", 0),
                "Unc (Hall)": metrics.get("avg_unc_hallucinated", 0),
                "Unc (Faith)": metrics.get("avg_unc_faithful", 0),
                "N": metrics.get("n_samples", 0),
            }
        else:
            # Recompute from raw results
            predictions = res.get("results", [])
            if not predictions:
                continue

            labels = np.array([p["label"] for p in predictions])
            scores = np.array([p["uncertainty"] for p in predictions])
            latencies = [p.get("latency_ms", 0) for p in predictions]

            row = {
                "Method": method_name,
                "AUROC": roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0,
                "AUPRC": average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0,
                "TPR@5%FPR": compute_tpr_at_fpr(labels, scores),
                "AURC (↓)": compute_aurc(labels, scores),
                "Latency (ms)": np.mean(latencies) if latencies else 0,
                "Unc (Hall)": scores[labels == 1].mean() if (labels == 1).sum() > 0 else 0,
                "Unc (Faith)": scores[labels == 0].mean() if (labels == 0).sum() > 0 else 0,
                "N": len(predictions),
            }

        # Add delta column
        row["Delta (H-F)"] = row["Unc (Hall)"] - row["Unc (Faith)"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by AUROC descending
    if not df.empty:
        df = df.sort_values(by="AUROC", ascending=False)

    return df


def print_table(df: pd.DataFrame, format: str = "markdown"):
    """Print results table in specified format."""
    if df.empty:
        print("No results to display.")
        return

    # Select columns for main table
    main_cols = ["Method", "AUROC", "AUPRC", "TPR@5%FPR", "AURC (↓)", "Latency (ms)"]
    main_df = df[main_cols].copy()

    print("\n" + "=" * 70)
    print("HaluEval Summarization: Head-to-Head Results")
    print("=" * 70)

    if format == "markdown":
        print(main_df.to_markdown(index=False, floatfmt=".4f"))
    elif format == "latex":
        print(main_df.to_latex(index=False, float_format="%.4f"))
    else:
        print(main_df.to_string(index=False))

    # Print detailed analysis
    print("\n" + "-" * 70)
    print("Detailed Uncertainty Analysis")
    print("-" * 70)

    detail_cols = ["Method", "Unc (Hall)", "Unc (Faith)", "Delta (H-F)", "N"]
    detail_df = df[detail_cols].copy()
    print(detail_df.to_markdown(index=False, floatfmt=".4f"))

    # Print interpretation
    print("\n" + "-" * 70)
    print("Interpretation Guide")
    print("-" * 70)

    if len(df) >= 2:
        best_auroc = df.iloc[0]
        comparison = df.iloc[1] if len(df) > 1 else None

        print(f"\nBest Method: {best_auroc['Method']}")
        print(f"  - AUROC: {best_auroc['AUROC']:.4f}")

        if comparison is not None:
            auroc_diff = best_auroc["AUROC"] - comparison["AUROC"]
            tpr_diff = best_auroc["TPR@5%FPR"] - comparison["TPR@5%FPR"]

            print(f"\nvs {comparison['Method']}:")
            print(f"  - AUROC improvement: {auroc_diff:+.4f} ({auroc_diff/comparison['AUROC']*100:+.1f}%)")
            print(f"  - TPR@5%FPR improvement: {tpr_diff:+.4f}")

            if tpr_diff > 0.1:
                print(f"\n  KEY FINDING: At 5% false alarm rate, {best_auroc['Method']} catches")
                print(f"  {tpr_diff*100:.1f}% more hallucinations than {comparison['Method']}.")


def find_latest_results(base_dir: Path) -> Path:
    """Find the most recent results directory."""
    if not base_dir.exists():
        return None

    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None

    # Sort by name (timestamp format) and return latest
    return sorted(subdirs)[-1]


def main():
    parser = argparse.ArgumentParser(description="Analyze H2H Benchmark Results")
    parser.add_argument("--results-dir", type=str, help="Path to results directory")
    parser.add_argument("--latest", action="store_true", help="Auto-find latest results")
    parser.add_argument(
        "--format",
        choices=["markdown", "latex", "plain"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="results/halueval_h2h",
        help="Base directory for results (used with --latest)",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.latest:
        base_dir = PROJECT_ROOT / args.base_dir
        results_dir = find_latest_results(base_dir)
        if results_dir is None:
            print(f"No results found in {base_dir}")
            sys.exit(1)
        print(f"Using latest results: {results_dir}")
    else:
        print("Error: Specify --results-dir or use --latest")
        sys.exit(1)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)

    # Load and analyze results
    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        sys.exit(1)

    df = analyze_results(results)
    print_table(df, format=args.format)

    # Save to CSV
    csv_path = results_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to: {csv_path}")


if __name__ == "__main__":
    main()
