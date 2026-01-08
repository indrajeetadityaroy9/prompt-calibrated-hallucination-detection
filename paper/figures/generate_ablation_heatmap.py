#!/usr/bin/env python3
"""
Generate ablation study heatmap for paper figure.

Visualizes how AUROC changes when varying pairs of hyperparameters,
helping readers understand parameter sensitivity.

Usage:
    python paper/figures/generate_ablation_heatmap.py --results results/02_ablation/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300,
})

# Paper notation for parameters
PARAM_NOTATION = {
    'prompt_authority': r'$\alpha_p$ (Prompt Authority)',
    'memory_weight': r'$w_m$ (Memory Weight)',
    'gate_temperature': r'$\tau$ (Gate Temperature)',
    'dispersion_k': r'$k$ (Dispersion Top-k)',
    'semantic_layers': r'$L$ (Semantic Layers)',
    'residual_weight': r'$\beta$ (Residual Weight)',
    'dispersion_sensitivity': r'$\gamma$ (Dispersion Sensitivity)',
}


def load_ablation_results(results_dir: Path) -> dict:
    """
    Load ablation results from JSONL files.

    Expected format per line:
    {"param1": "prompt_authority", "value1": 0.3, "param2": "memory_weight",
     "value2": 0.5, "auroc": 0.87, "dataset": "halueval"}
    """
    results = []

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if 'param1' in record and 'param2' in record:
                    results.append(record)

    return results


def create_heatmap_data(results: list, param1: str, param2: str,
                        metric: str = 'auroc') -> tuple:
    """
    Extract heatmap data for a parameter pair.

    Returns:
        (values1, values2, heatmap_matrix)
    """
    # Filter for this parameter pair
    filtered = [r for r in results
                if r.get('param1') == param1 and r.get('param2') == param2]

    if not filtered:
        # Try reverse order
        filtered = [r for r in results
                    if r.get('param1') == param2 and r.get('param2') == param1]
        if filtered:
            param1, param2 = param2, param1

    if not filtered:
        return None, None, None

    # Get unique values
    values1 = sorted(set(r['value1'] for r in filtered))
    values2 = sorted(set(r['value2'] for r in filtered))

    # Build matrix
    matrix = np.full((len(values2), len(values1)), np.nan)
    for r in filtered:
        i = values2.index(r['value2'])
        j = values1.index(r['value1'])
        matrix[i, j] = r.get(metric, 0)

    return values1, values2, matrix


def plot_ablation_heatmap(results: list, param1: str, param2: str,
                          output_path: Path, metric: str = 'auroc',
                          title: str = None):
    """Generate a single ablation heatmap."""
    values1, values2, matrix = create_heatmap_data(results, param1, param2, metric)

    if matrix is None:
        print(f"Warning: No data for {param1} vs {param2}")
        return False

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap: red (low) -> white (mid) -> blue (high)
    cmap = LinearSegmentedColormap.from_list('auroc', ['#d73027', '#ffffff', '#4575b4'])

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                   vmin=0.5, vmax=1.0, origin='lower')

    # Labels
    ax.set_xticks(range(len(values1)))
    ax.set_xticklabels([f'{v:.2f}' if isinstance(v, float) else str(v)
                        for v in values1])
    ax.set_yticks(range(len(values2)))
    ax.set_yticklabels([f'{v:.2f}' if isinstance(v, float) else str(v)
                        for v in values2])

    ax.set_xlabel(PARAM_NOTATION.get(param1, param1))
    ax.set_ylabel(PARAM_NOTATION.get(param2, param2))

    if title:
        ax.set_title(title)

    # Annotate cells with values
    for i in range(len(values2)):
        for j in range(len(values1)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.7 or val > 0.9 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=color, fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('AUROC', rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation heatmap to {output_path}")
    return True


def plot_ablation_grid(results: list, output_path: Path,
                       params: list = None, metric: str = 'auroc'):
    """Generate a grid of ablation heatmaps for multiple parameter pairs."""
    if params is None:
        params = ['prompt_authority', 'memory_weight', 'gate_temperature',
                  'dispersion_k']

    n_params = len(params)
    # Only upper triangle of parameter pairs
    n_plots = n_params * (n_params - 1) // 2

    if n_plots == 0:
        print("Error: Need at least 2 parameters for comparison grid")
        return

    # Determine grid layout
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    cmap = LinearSegmentedColormap.from_list('auroc', ['#d73027', '#ffffff', '#4575b4'])

    plot_idx = 0
    for i in range(n_params):
        for j in range(i + 1, n_params):
            if plot_idx >= n_rows * n_cols:
                break

            row, col = plot_idx // n_cols, plot_idx % n_cols
            ax = axes[row, col]

            param1, param2 = params[i], params[j]
            values1, values2, matrix = create_heatmap_data(results, param1, param2, metric)

            if matrix is None:
                ax.set_visible(False)
                plot_idx += 1
                continue

            im = ax.imshow(matrix, cmap=cmap, aspect='auto',
                          vmin=0.5, vmax=1.0, origin='lower')

            ax.set_xticks(range(len(values1)))
            ax.set_xticklabels([f'{v:.1f}' if isinstance(v, float) else str(v)
                                for v in values1], fontsize=8)
            ax.set_yticks(range(len(values2)))
            ax.set_yticklabels([f'{v:.1f}' if isinstance(v, float) else str(v)
                                for v in values2], fontsize=8)

            # Short labels for grid
            ax.set_xlabel(PARAM_NOTATION.get(param1, param1).split('(')[0].strip(),
                         fontsize=10)
            ax.set_ylabel(PARAM_NOTATION.get(param2, param2).split('(')[0].strip(),
                         fontsize=10)

            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    # Add shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='AUROC')

    plt.suptitle('Parameter Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ablation grid to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate ablation heatmaps")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to ablation results directory')
    parser.add_argument('--output', type=str, default='paper/figures/ablation_heatmap.pdf',
                        help='Output file path')
    parser.add_argument('--param1', type=str, default=None,
                        help='First parameter for single heatmap')
    parser.add_argument('--param2', type=str, default=None,
                        help='Second parameter for single heatmap')
    parser.add_argument('--grid', action='store_true',
                        help='Generate grid of all parameter pairs')
    parser.add_argument('--metric', type=str, default='auroc',
                        help='Metric to plot (default: auroc)')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    results = load_ablation_results(results_dir)
    if not results:
        print("Error: No ablation results found")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.grid:
        plot_ablation_grid(results, output_path, metric=args.metric)
    elif args.param1 and args.param2:
        success = plot_ablation_heatmap(
            results, args.param1, args.param2, output_path,
            metric=args.metric,
            title=f'Ablation: {args.param1} vs {args.param2}'
        )
        if not success:
            return 1
    else:
        # Default: generate grid
        plot_ablation_grid(results, output_path, metric=args.metric)

    return 0


if __name__ == '__main__':
    exit(main())
