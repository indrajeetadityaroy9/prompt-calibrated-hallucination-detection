#!/usr/bin/env python3
"""
Generate latency comparison bar chart for paper figure.

Compares inference latency of AG-SAR vs baseline methods,
demonstrating computational efficiency.

Usage:
    python paper/figures/generate_latency_comparison.py --results results/latency/
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (10, 5),
    'figure.dpi': 300,
})

# Method colors consistent with other figures
METHOD_COLORS = {
    'AG-SAR': '#2E86AB',
    'LogProb': '#A23B72',
    'Entropy': '#F18F01',
    'SelfCheck': '#C73E1D',
    'EigenScore': '#3B1F2B',
    'SemanticEntropy': '#6B4226',
}

# Method display order (AG-SAR first, then baselines by speed)
METHOD_ORDER = [
    'AG-SAR',
    'LogProb',
    'Entropy',
    'EigenScore',
    'SelfCheck',
    'SemanticEntropy',
]


def load_latency_results(results_dir: Path) -> dict:
    """
    Load latency measurements from JSONL files.

    Expected format per line:
    {"method": "AG-SAR", "latency_ms": 45.2, "latency_std": 5.1,
     "sequence_length": 512, "batch_size": 1}
    """
    results = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                method = record.get('method', 'unknown')
                if method not in results:
                    results[method] = {'latencies': [], 'stds': []}
                results[method]['latencies'].append(record.get('latency_ms', 0))
                results[method]['stds'].append(record.get('latency_std', 0))

    # Aggregate
    aggregated = {}
    for method, data in results.items():
        aggregated[method] = {
            'mean': np.mean(data['latencies']),
            'std': np.sqrt(np.mean(np.array(data['stds'])**2)),  # Combined std
        }

    return aggregated


def plot_latency_bars(results: dict, output_path: Path,
                      title: str = "Inference Latency Comparison"):
    """Generate latency comparison bar chart."""
    # Order methods
    methods = [m for m in METHOD_ORDER if m in results]
    methods += [m for m in results if m not in methods]  # Add any extras

    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    colors = [METHOD_COLORS.get(m, '#666666') for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(methods))
    width = 0.6

    bars = ax.bar(x, means, width, yerr=stds, color=colors,
                  capsize=5, error_kw={'linewidth': 1.5})

    # Highlight AG-SAR bar
    if 'AG-SAR' in methods:
        agsar_idx = methods.index('AG-SAR')
        bars[agsar_idx].set_edgecolor('black')
        bars[agsar_idx].set_linewidth(2)

    ax.set_xlabel('Method')
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.annotate(f'{mean:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height + std + 1),
                   ha='center', va='bottom', fontsize=10)

    # Add speedup annotation for AG-SAR
    if 'AG-SAR' in methods and len(methods) > 1:
        agsar_latency = results['AG-SAR']['mean']
        # Find slowest multi-sample method
        slow_methods = ['SelfCheck', 'SemanticEntropy']
        slowest = max((results[m]['mean'] for m in slow_methods if m in results),
                     default=agsar_latency)
        if slowest > agsar_latency:
            speedup = slowest / agsar_latency
            ax.annotate(f'{speedup:.1f}x faster\nthan multi-sample',
                       xy=(0.02, 0.95), xycoords='axes fraction',
                       fontsize=11, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latency comparison to {output_path}")


def plot_latency_by_length(results_dir: Path, output_path: Path):
    """Generate latency vs sequence length plot."""
    # Load detailed results
    data_by_method = {}

    for jsonl_file in results_dir.glob("*.jsonl"):
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                method = record.get('method', 'unknown')
                seq_len = record.get('sequence_length', 512)
                latency = record.get('latency_ms', 0)

                if method not in data_by_method:
                    data_by_method[method] = {}
                if seq_len not in data_by_method[method]:
                    data_by_method[method][seq_len] = []
                data_by_method[method][seq_len].append(latency)

    if not data_by_method:
        print("No sequence length data found")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in METHOD_ORDER:
        if method not in data_by_method:
            continue

        seq_lens = sorted(data_by_method[method].keys())
        means = [np.mean(data_by_method[method][s]) for s in seq_lens]
        stds = [np.std(data_by_method[method][s]) for s in seq_lens]

        color = METHOD_COLORS.get(method, '#666666')
        ax.errorbar(seq_lens, means, yerr=stds, label=method,
                   color=color, marker='o', capsize=3, linewidth=2)

    ax.set_xlabel('Sequence Length (tokens)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency vs Sequence Length')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved latency vs length to {output_path}")


def plot_throughput_comparison(results: dict, output_path: Path):
    """Generate throughput (samples/sec) comparison."""
    methods = [m for m in METHOD_ORDER if m in results]
    methods += [m for m in results if m not in methods]

    # Convert latency to throughput (assuming batch_size=1)
    throughputs = [1000 / results[m]['mean'] for m in methods]  # samples/sec
    colors = [METHOD_COLORS.get(m, '#666666') for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(methods))
    width = 0.6

    bars = ax.bar(x, throughputs, width, color=colors)

    # Highlight AG-SAR
    if 'AG-SAR' in methods:
        agsar_idx = methods.index('AG-SAR')
        bars[agsar_idx].set_edgecolor('black')
        bars[agsar_idx].set_linewidth(2)

    ax.set_xlabel('Method')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Inference Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')

    # Add value labels
    for bar, tp in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(f'{tp:.1f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   ha='center', va='bottom', fontsize=10)

    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved throughput comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate latency comparison figures")
    parser.add_argument('--results', type=str, required=True,
                        help='Path to latency results directory')
    parser.add_argument('--output', type=str, default='paper/figures/latency_comparison.pdf',
                        help='Output file path')
    parser.add_argument('--by-length', action='store_true',
                        help='Generate latency vs sequence length plot')
    parser.add_argument('--throughput', action='store_true',
                        help='Generate throughput comparison')
    parser.add_argument('--title', type=str, default='Inference Latency Comparison',
                        help='Figure title')
    args = parser.parse_args()

    results_dir = Path(args.results)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.by_length:
        plot_latency_by_length(results_dir, output_path)
    elif args.throughput:
        results = load_latency_results(results_dir)
        if not results:
            print("Error: No latency results found")
            return 1
        plot_throughput_comparison(results, output_path)
    else:
        results = load_latency_results(results_dir)
        if not results:
            print("Error: No latency results found")
            return 1
        plot_latency_bars(results, output_path, title=args.title)

    return 0


if __name__ == '__main__':
    exit(main())
