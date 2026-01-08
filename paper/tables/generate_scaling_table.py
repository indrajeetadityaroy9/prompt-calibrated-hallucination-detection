#!/usr/bin/env python3
"""
Generate model scaling table for paper.

Shows AG-SAR performance across different model sizes.

Usage:
    python paper/tables/generate_scaling_table.py --results results/03_scaling/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score

# Model size ordering
MODEL_ORDER = [
    ('Llama-3.1-8B', '8B'),
    ('Llama-3.1-70B', '70B'),
    ('Llama-3.2-1B', '1B'),
    ('Llama-3.2-3B', '3B'),
    ('Mistral-7B', '7B'),
    ('Qwen2-7B', '7B'),
]

# Model families for grouping
MODEL_FAMILIES = {
    'llama-3.1': ['Llama-3.1-8B', 'Llama-3.1-70B'],
    'llama-3.2': ['Llama-3.2-1B', 'Llama-3.2-3B'],
    'mistral': ['Mistral-7B'],
    'qwen': ['Qwen2-7B'],
}


def load_scaling_results(results_dir: Path) -> Dict[str, Dict[str, List]]:
    """
    Load scaling experiment results.

    Expected format:
    {"model": "Llama-3.1-8B", "dataset": "HaluEval-QA", "score": 0.87, "label": 1,
     "latency_ms": 45.2, "memory_gb": 16.5}
    """
    results = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                model = record.get('model', 'unknown')
                dataset = record.get('dataset', 'unknown')

                if model not in results:
                    results[model] = {
                        'datasets': {},
                        'latencies': [],
                        'memory': [],
                    }
                if dataset not in results[model]['datasets']:
                    results[model]['datasets'][dataset] = {'scores': [], 'labels': []}

                results[model]['datasets'][dataset]['scores'].append(record.get('score', 0))
                results[model]['datasets'][dataset]['labels'].append(record.get('label', 0))

                if 'latency_ms' in record:
                    results[model]['latencies'].append(record['latency_ms'])
                if 'memory_gb' in record:
                    results[model]['memory'].append(record['memory_gb'])

    return results


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC, return 0 if only one class."""
    if len(set(labels)) < 2:
        return 0.0
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


def compute_scaling_metrics(results: Dict) -> Dict[str, Dict]:
    """Compute metrics for each model."""
    metrics = {}

    for model, data in results.items():
        metrics[model] = {
            'auroc_by_dataset': {},
            'auroc_avg': 0.0,
            'latency_ms': np.mean(data['latencies']) if data['latencies'] else 0,
            'memory_gb': np.mean(data['memory']) if data['memory'] else 0,
        }

        aurocs = []
        for dataset, ds_data in data['datasets'].items():
            auroc = compute_auroc(ds_data['scores'], ds_data['labels'])
            metrics[model]['auroc_by_dataset'][dataset] = auroc
            aurocs.append(auroc)

        metrics[model]['auroc_avg'] = np.mean(aurocs) if aurocs else 0.0

    return metrics


def generate_scaling_table(metrics: Dict[str, Dict],
                           output_path: Path,
                           include_latency: bool = True,
                           include_memory: bool = True):
    """Generate LaTeX scaling table."""
    # Determine model order
    models = []
    for model_name, _ in MODEL_ORDER:
        matching = [m for m in metrics.keys() if model_name.lower() in m.lower()]
        models.extend(matching)
    # Add any remaining models
    models.extend([m for m in metrics.keys() if m not in models])

    # Determine columns
    cols = ['AUROC']
    if include_latency:
        cols.append('Latency')
    if include_memory:
        cols.append('Memory')

    col_spec = "l" + "c" * len(cols)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Model Scaling: AG-SAR performance across model sizes.}",
        "\\label{tab:scaling}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Header
    header_parts = ["Model"]
    if 'AUROC' in cols:
        header_parts.append("AUROC (Avg)")
    if include_latency:
        header_parts.append("Latency (ms)")
    if include_memory:
        header_parts.append("Memory (GB)")

    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    # Find best AUROC for bolding
    best_auroc = max(m['auroc_avg'] for m in metrics.values())

    for model in models:
        if model not in metrics:
            continue

        data = metrics[model]
        row_parts = [model]

        auroc = data['auroc_avg']
        if abs(auroc - best_auroc) < 0.0001:
            row_parts.append(f"\\textbf{{{auroc:.3f}}}")
        else:
            row_parts.append(f"{auroc:.3f}")

        if include_latency:
            latency = data['latency_ms']
            row_parts.append(f"{latency:.1f}" if latency > 0 else "-")

        if include_memory:
            memory = data['memory_gb']
            row_parts.append(f"{memory:.1f}" if memory > 0 else "-")

        lines.append(" & ".join(row_parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated scaling table: {output_path}")
    print("\nPreview:")
    print("-" * 60)
    print(latex)
    print("-" * 60)

    return latex


def generate_family_comparison_table(metrics: Dict[str, Dict],
                                     output_path: Path):
    """Generate table comparing model families."""
    # Group by family
    family_metrics = {}

    for family, models in MODEL_FAMILIES.items():
        family_aurocs = []
        for model in models:
            matching = [m for m in metrics.keys() if model.lower() in m.lower()]
            for m in matching:
                family_aurocs.append(metrics[m]['auroc_avg'])

        if family_aurocs:
            family_metrics[family] = {
                'auroc_mean': np.mean(family_aurocs),
                'auroc_std': np.std(family_aurocs) if len(family_aurocs) > 1 else 0,
                'n_models': len(family_aurocs),
            }

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{AG-SAR Performance by Model Family}",
        "\\label{tab:family_comparison}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model Family & AUROC & \\# Models \\\\",
        "\\midrule",
    ]

    for family in ['llama-3.1', 'llama-3.2', 'mistral', 'qwen']:
        if family not in family_metrics:
            continue

        data = family_metrics[family]
        mean = data['auroc_mean']
        std = data['auroc_std']
        n = data['n_models']

        if std > 0:
            auroc_str = f"${mean:.3f} \\pm {std:.3f}$"
        else:
            auroc_str = f"{mean:.3f}"

        lines.append(f"{family.title()} & {auroc_str} & {n} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated family comparison table: {output_path}")
    return latex


def generate_detailed_scaling_table(metrics: Dict[str, Dict],
                                    output_path: Path,
                                    datasets: List[str] = None):
    """Generate detailed scaling table with per-dataset results."""
    models = []
    for model_name, _ in MODEL_ORDER:
        matching = [m for m in metrics.keys() if model_name.lower() in m.lower()]
        models.extend(matching)
    models.extend([m for m in metrics.keys() if m not in models])

    if datasets is None:
        # Get datasets from first model
        first_model = list(metrics.values())[0]
        datasets = list(first_model['auroc_by_dataset'].keys())[:4]

    short_names = {
        'HaluEval-QA': 'HE-QA',
        'HaluEval-Summ': 'HE-Sum',
        'RAGTruth': 'RAG',
        'TruthfulQA': 'TQA',
    }

    col_names = [short_names.get(d, d[:6]) for d in datasets]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Detailed Scaling: AG-SAR AUROC across models and datasets.}",
        "\\label{tab:scaling_detailed}",
        "\\begin{tabular}{l" + "c" * len(datasets) + "c}",
        "\\toprule",
        "Model & " + " & ".join(col_names) + " & Avg \\\\",
        "\\midrule",
    ]

    # Find best per dataset
    best_per_dataset = {}
    for dataset in datasets:
        values = [metrics[m]['auroc_by_dataset'].get(dataset, 0) for m in models if m in metrics]
        best_per_dataset[dataset] = max(values) if values else 0

    for model in models:
        if model not in metrics:
            continue

        data = metrics[model]
        row_values = []

        for dataset in datasets:
            value = data['auroc_by_dataset'].get(dataset, 0)
            if abs(value - best_per_dataset[dataset]) < 0.0001 and value > 0:
                row_values.append(f"\\textbf{{{value:.3f}}}")
            else:
                row_values.append(f"{value:.3f}" if value > 0 else "-")

        avg = data['auroc_avg']
        lines.append(f"{model} & " + " & ".join(row_values) + f" & {avg:.3f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated detailed scaling table: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate model scaling table")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to scaling results directory')
    parser.add_argument('--output', type=str, default='paper/tables/scaling.tex',
                        help='Output file path')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed per-dataset table')
    parser.add_argument('--family', action='store_true',
                        help='Generate model family comparison table')
    parser.add_argument('--no-latency', action='store_true',
                        help='Exclude latency column')
    parser.add_argument('--no-memory', action='store_true',
                        help='Exclude memory column')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    results = load_scaling_results(results_dir)
    if not results:
        print("Error: No scaling results found")
        return 1

    metrics = compute_scaling_metrics(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.detailed:
        generate_detailed_scaling_table(metrics, output_path)
    elif args.family:
        generate_family_comparison_table(metrics, output_path)
    else:
        generate_scaling_table(
            metrics, output_path,
            include_latency=not args.no_latency,
            include_memory=not args.no_memory
        )

    return 0


if __name__ == '__main__':
    exit(main())
