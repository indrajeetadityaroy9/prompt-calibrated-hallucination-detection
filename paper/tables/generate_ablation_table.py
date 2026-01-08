#!/usr/bin/env python3
"""
Generate ablation study table for paper.

Shows impact of removing each component of AG-SAR.

Usage:
    python paper/tables/generate_ablation_table.py --results results/02_ablation/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.metrics import roc_auc_score

# Component ablation configurations
ABLATION_CONFIGS = {
    'Full AG-SAR': 'full',
    'w/o Authority Flow': 'no_authority',
    'w/o Agreement Gate': 'no_gate',
    'w/o Semantic Dispersion': 'no_dispersion',
    'w/o Unified Gating': 'no_unified_gating',
    'w/o Residual Connection': 'no_residual',
    'Authority Flow Only': 'authority_only',
}

# Paper notation for components
COMPONENT_NOTATION = {
    'no_authority': r'$-\mathcal{A}$',
    'no_gate': r'$-\mathcal{G}$',
    'no_dispersion': r'$-\mathcal{D}$',
    'no_unified_gating': r'$-\mathcal{U}$',
    'no_residual': r'$-\beta$',
    'authority_only': r'$\mathcal{A}$ only',
}


def load_ablation_results(results_dir: Path) -> Dict[str, Dict[str, List]]:
    """
    Load ablation results from JSONL files.

    Expected format:
    {"ablation": "no_gate", "dataset": "HaluEval-QA", "score": 0.72, "label": 1}
    """
    results = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                ablation = record.get('ablation', 'full')
                dataset = record.get('dataset', 'unknown')

                if ablation not in results:
                    results[ablation] = {}
                if dataset not in results[ablation]:
                    results[ablation][dataset] = {'scores': [], 'labels': []}

                results[ablation][dataset]['scores'].append(record.get('score', 0))
                results[ablation][dataset]['labels'].append(record.get('label', 0))

    return results


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """Compute AUROC, return 0 if only one class."""
    if len(set(labels)) < 2:
        return 0.0
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.0


def compute_ablation_metrics(results: Dict) -> Dict[str, Dict[str, float]]:
    """Compute AUROC for each ablation-dataset pair."""
    metrics = {}

    for ablation, datasets in results.items():
        metrics[ablation] = {}
        for dataset, data in datasets.items():
            auroc = compute_auroc(data['scores'], data['labels'])
            metrics[ablation][dataset] = auroc

        # Compute average across datasets
        values = list(metrics[ablation].values())
        metrics[ablation]['Avg'] = np.mean(values) if values else 0.0

    return metrics


def generate_ablation_table(metrics: Dict[str, Dict[str, float]],
                            output_path: Path,
                            datasets: List[str] = None):
    """Generate LaTeX ablation table."""
    if datasets is None:
        # Get datasets from first ablation
        first_ablation = list(metrics.values())[0]
        datasets = [d for d in first_ablation.keys() if d != 'Avg']

    # Get full model baseline
    full_avg = metrics.get('full', {}).get('Avg', 0)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Ablation Study: Impact of removing AG-SAR components. $\\Delta$ shows change from full model.}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Configuration & AUROC (Avg) & $\\Delta$ \\\\",
        "\\midrule",
    ]

    # Full model first
    if 'full' in metrics:
        lines.append(f"Full AG-SAR & \\textbf{{{full_avg:.3f}}} & - \\\\")
        lines.append("\\midrule")

    # Ablations
    ablation_order = ['no_authority', 'no_gate', 'no_dispersion',
                      'no_unified_gating', 'no_residual']

    for ablation in ablation_order:
        if ablation not in metrics:
            continue

        avg = metrics[ablation].get('Avg', 0)
        delta = avg - full_avg
        delta_str = f"{delta:+.3f}"

        # Get display name
        display_name = None
        for name, key in ABLATION_CONFIGS.items():
            if key == ablation:
                display_name = name
                break
        if display_name is None:
            display_name = ablation

        lines.append(f"{display_name} & {avg:.3f} & {delta_str} \\\\")

    # Authority only baseline
    if 'authority_only' in metrics:
        lines.append("\\midrule")
        avg = metrics['authority_only'].get('Avg', 0)
        delta = avg - full_avg
        lines.append(f"Authority Flow Only & {avg:.3f} & {delta:+.3f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated ablation table: {output_path}")
    print("\nPreview:")
    print("-" * 60)
    print(latex)
    print("-" * 60)

    return latex


def generate_detailed_ablation_table(metrics: Dict[str, Dict[str, float]],
                                     output_path: Path,
                                     datasets: List[str] = None):
    """Generate detailed ablation table with per-dataset results."""
    if datasets is None:
        first_ablation = list(metrics.values())[0]
        datasets = [d for d in first_ablation.keys() if d != 'Avg'][:4]

    # Short names
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
        "\\caption{Detailed Ablation: Per-dataset AUROC for each configuration.}",
        "\\label{tab:ablation_detailed}",
        "\\begin{tabular}{l" + "c" * len(datasets) + "c}",
        "\\toprule",
        "Configuration & " + " & ".join(col_names) + " & Avg \\\\",
        "\\midrule",
    ]

    # Order: full first, then ablations
    config_order = ['full'] + ['no_authority', 'no_gate', 'no_dispersion',
                               'no_unified_gating', 'no_residual', 'authority_only']

    for ablation in config_order:
        if ablation not in metrics:
            continue

        # Get display name
        display_name = None
        for name, key in ABLATION_CONFIGS.items():
            if key == ablation:
                display_name = name
                break
        if display_name is None:
            display_name = ablation

        row_values = []
        for dataset in datasets:
            value = metrics[ablation].get(dataset, 0)
            row_values.append(f"{value:.3f}")

        avg = metrics[ablation].get('Avg', 0)

        if ablation == 'full':
            lines.append(f"\\textbf{{{display_name}}} & " +
                        " & ".join([f"\\textbf{{{v}}}" for v in row_values]) +
                        f" & \\textbf{{{avg:.3f}}} \\\\")
            lines.append("\\midrule")
        else:
            lines.append(f"{display_name} & " + " & ".join(row_values) +
                        f" & {avg:.3f} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    latex = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"Generated detailed ablation table: {output_path}")
    return latex


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study table")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to ablation results directory')
    parser.add_argument('--output', type=str, default='paper/tables/ablation.tex',
                        help='Output file path')
    parser.add_argument('--detailed', action='store_true',
                        help='Generate detailed per-dataset table')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    results = load_ablation_results(results_dir)
    if not results:
        print("Error: No ablation results found")
        return 1

    metrics = compute_ablation_metrics(results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.detailed:
        generate_detailed_ablation_table(metrics, output_path)
    else:
        generate_ablation_table(metrics, output_path)

    return 0


if __name__ == '__main__':
    exit(main())
