#!/usr/bin/env python3
"""
Generate ROC curves for main results figure.

Usage:
    python paper/figures/generate_auroc_curves.py --results results/01_main_sota/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
})

# Method colors for consistency across figures
METHOD_COLORS = {
    'AG-SAR': '#2E86AB',      # Blue
    'LogProb': '#A23B72',      # Pink
    'Entropy': '#F18F01',      # Orange
    'SelfCheck': '#C73E1D',    # Red
    'EigenScore': '#3B1F2B',   # Dark
    'SemanticEntropy': '#6B4226',  # Brown
}

METHOD_MARKERS = {
    'AG-SAR': 'o',
    'LogProb': 's',
    'Entropy': '^',
    'SelfCheck': 'D',
    'EigenScore': 'v',
    'SemanticEntropy': 'p',
}


def load_results(results_dir: Path) -> dict:
    """Load all JSONL results from directory."""
    results_by_method = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                method = record.get('method', 'unknown')
                if method not in results_by_method:
                    results_by_method[method] = {'scores': [], 'labels': []}
                results_by_method[method]['scores'].append(record.get('score', 0))
                results_by_method[method]['labels'].append(record.get('label', 0))

    return results_by_method


def plot_roc_curves(results: dict, output_path: Path, title: str = "ROC Curves"):
    """Generate ROC curve comparison figure."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for method, data in sorted(results.items()):
        scores = np.array(data['scores'])
        labels = np.array(data['labels'])

        if len(np.unique(labels)) < 2:
            print(f"Warning: {method} has only one class, skipping")
            continue

        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        color = METHOD_COLORS.get(method, '#666666')
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{method} (AUROC = {roc_auc:.3f})')

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ROC curves from results")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results directory')
    parser.add_argument('--output', type=str, default='paper/figures/roc_curves.pdf',
                        help='Output file path')
    parser.add_argument('--title', type=str, default='ROC Curves - Method Comparison',
                        help='Figure title')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    results = load_results(results_dir)
    if not results:
        print("Error: No results found")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(results, output_path, title=args.title)
    return 0


if __name__ == '__main__':
    exit(main())
