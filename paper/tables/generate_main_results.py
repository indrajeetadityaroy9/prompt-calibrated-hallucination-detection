#!/usr/bin/env python3
"""
Generate main results table (Table 1) for paper.

Compares AG-SAR against baselines across benchmark datasets.

Usage:
    python paper/tables/generate_main_results.py --results results/01_main_sota/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import format_metric, generate_latex_table

# Standard method ordering for tables
METHOD_ORDER = [
    'AG-SAR',
    'LogProb',
    'Entropy',
    'EigenScore',
    'SelfCheck',
    'SemanticEntropy',
]

# Standard dataset ordering
DATASET_ORDER = [
    'HaluEval-QA',
    'HaluEval-Summ',
    'HaluEval-Dial',
    'RAGTruth',
    'TruthfulQA',
    'FAVA',
]

# Short names for table columns
DATASET_SHORT_NAMES = {
    'HaluEval-QA': 'HE-QA',
    'HaluEval-Summ': 'HE-Sum',
    'HaluEval-Dial': 'HE-Dial',
    'RAGTruth': 'RAG',
    'TruthfulQA': 'TQA',
    'FAVA': 'FAVA',
}


def load_results(results_dir: Path) -> Dict[str, Dict[str, List]]:
    """
    Load all results from JSONL files.

    Returns:
        Nested dict: method -> dataset -> {'scores': [...], 'labels': [...]}
    """
    results = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                method = record.get('method', 'unknown')
                dataset = record.get('dataset', 'unknown')

                if method not in results:
                    results[method] = {}
                if dataset not in results[method]:
                    results[method][dataset] = {'scores': [], 'labels': []}

                results[method][dataset]['scores'].append(record.get('score', 0))
                results[method][dataset]['labels'].append(record.get('label', 0))

    return results


def compute_auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    """Compute AUROC from scores and labels."""
    if len(set(labels)) < 2:
        return None
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return None


def compute_metrics(results: Dict[str, Dict[str, List]]) -> Dict[str, Dict[str, float]]:
    """
    Compute AUROC for each method-dataset pair.

    Returns:
        Dict: method -> dataset -> auroc
    """
    metrics = {}

    for method, datasets in results.items():
        metrics[method] = {}
        for dataset, data in datasets.items():
            auroc = compute_auroc(data['scores'], data['labels'])
            if auroc is not None:
                metrics[method][dataset] = auroc

    return metrics


def generate_table(metrics: Dict[str, Dict[str, float]],
                   output_path: Path,
                   caption: str = "Main Results: AUROC comparison across hallucination benchmarks",
                   label: str = "tab:main_results"):
    """Generate LaTeX table from metrics."""
    # Determine which datasets are present
    all_datasets = set()
    for method_data in metrics.values():
        all_datasets.update(method_data.keys())

    datasets = [d for d in DATASET_ORDER if d in all_datasets]
    datasets += [d for d in all_datasets if d not in DATASET_ORDER]

    # Determine which methods are present
    methods = [m for m in METHOD_ORDER if m in metrics]
    methods += [m for m in metrics if m not in METHOD_ORDER]

    # Find best value per dataset
    best_per_dataset = {}
    for dataset in datasets:
        values = [metrics[m].get(dataset, 0) for m in methods]
        best_per_dataset[dataset] = max(values) if values else 0

    # Build table
    col_names = [DATASET_SHORT_NAMES.get(d, d) for d in datasets]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(datasets) + "c}",
        "\\toprule",
        "Method & " + " & ".join(col_names) + " & Avg \\\\",
        "\\midrule",
    ]

    for method in methods:
        row_values = []
        valid_scores = []

        for dataset in datasets:
            value = metrics[method].get(dataset)
            if value is not None:
                valid_scores.append(value)
                # Bold if best
                if abs(value - best_per_dataset[dataset]) < 0.0001:
                    row_values.append(f"\\textbf{{{value:.3f}}}")
                else:
                    row_values.append(f"{value:.3f}")
            else:
                row_values.append("-")

        # Compute average
        if valid_scores:
            avg = np.mean(valid_scores)
            # Check if this method has best average
            all_avgs = []
            for m in methods:
                m_scores = [metrics[m].get(d) for d in datasets if metrics[m].get(d) is not None]
                if m_scores:
                    all_avgs.append((m, np.mean(m_scores)))
            best_avg = max(a[1] for a in all_avgs) if all_avgs else 0

            if abs(avg - best_avg) < 0.0001:
                avg_str = f"\\textbf{{{avg:.3f}}}"
            else:
                avg_str = f"{avg:.3f}"
        else:
            avg_str = "-"

        lines.append(f"{method} & " + " & ".join(row_values) + f" & {avg_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    # Write to file
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated table: {output_path}")
    print("\nPreview:")
    print("-" * 60)
    print(latex)
    print("-" * 60)

    return latex


def generate_table_with_latency(metrics: Dict[str, Dict[str, float]],
                                latencies: Dict[str, float],
                                output_path: Path):
    """Generate table including latency column."""
    # Similar to above but with latency column
    all_datasets = set()
    for method_data in metrics.values():
        all_datasets.update(method_data.keys())

    datasets = [d for d in DATASET_ORDER if d in all_datasets][:4]  # Limit for space
    methods = [m for m in METHOD_ORDER if m in metrics]

    col_names = [DATASET_SHORT_NAMES.get(d, d) for d in datasets]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Main Results with Inference Latency}",
        "\\label{tab:main_with_latency}",
        "\\begin{tabular}{l" + "c" * len(datasets) + "cc}",
        "\\toprule",
        "Method & " + " & ".join(col_names) + " & Avg & Latency (ms) \\\\",
        "\\midrule",
    ]

    for method in methods:
        row_values = []
        valid_scores = []

        for dataset in datasets:
            value = metrics[method].get(dataset)
            if value is not None:
                valid_scores.append(value)
                row_values.append(f"{value:.3f}")
            else:
                row_values.append("-")

        avg = np.mean(valid_scores) if valid_scores else 0
        latency = latencies.get(method, 0)

        lines.append(f"{method} & " + " & ".join(row_values) +
                    f" & {avg:.3f} & {latency:.1f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated table with latency: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate main results table")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='paper/tables/main_results.tex',
                        help='Output file path')
    parser.add_argument('--caption', type=str,
                        default="Main Results: AUROC comparison across hallucination detection benchmarks. Best results in bold.",
                        help='Table caption')
    parser.add_argument('--label', type=str, default='tab:main_results',
                        help='Table label')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    results = load_results(results_dir)
    if not results:
        print("Error: No results found")
        return 1

    metrics = compute_metrics(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_table(metrics, output_path, caption=args.caption, label=args.label)

    return 0


if __name__ == '__main__':
    exit(main())
