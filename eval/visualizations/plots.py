"""
General plotting utilities for AG-SAR evaluation.

Includes ROC curves, calibration diagrams, bar charts, and more.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_experiment_summary(
    results: Dict,
    title: str = "AG-SAR Evaluation Summary",
    save_path: Optional[Path] = None
):
    """
    Create summary dashboard of all experiment results.

    Args:
        results: Dict with experiment results
        title: Dashboard title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return

    fig = plt.figure(figsize=(16, 12))

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. AUROC comparison (exp3)
    if 'exp3_auroc' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        _plot_auroc_bars(ax1, results['exp3_auroc'])

    # 2. ECE comparison (exp4)
    if 'exp4_calibration' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        _plot_ece_bars(ax2, results['exp4_calibration'])

    # 3. Latency breakdown (exp5)
    if 'exp5_latency' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        _plot_latency_breakdown(ax3, results['exp5_latency'])

    # 4. Throughput comparison (exp6)
    if 'exp6_throughput' in results:
        ax4 = fig.add_subplot(gs[1, 0])
        _plot_throughput_bars(ax4, results['exp6_throughput'])

    # 5. Ablation study (exp7)
    if 'exp7_ablation' in results:
        ax5 = fig.add_subplot(gs[1, 1])
        _plot_ablation_bars(ax5, results['exp7_ablation'])

    # 6. Summary metrics
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_success_summary(ax6, results)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_auroc_bars(ax, results: Dict):
    """Plot AUROC comparison bars."""
    methods = list(results.get('methods', {}).keys())
    aurocs = [results['methods'][m].get('auroc', 0) for m in methods]

    display_names = {
        'ag_sar': 'AG-SAR',
        'original_sar': 'Orig. SAR',
        'predictive_entropy': 'Pred. Entropy'
    }
    labels = [display_names.get(m, m) for m in methods]

    colors = ['#2ecc71' if m == 'ag_sar' else '#3498db' for m in methods]
    bars = ax.bar(labels, aurocs, color=colors, edgecolor='black')

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('AUROC')
    ax.set_title('Hallucination Detection')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)


def _plot_ece_bars(ax, results: Dict):
    """Plot ECE comparison bars."""
    methods = list(results.get('methods', {}).keys())
    eces = [results['methods'][m].get('ece', 0) for m in methods]

    display_names = {
        'ag_sar': 'AG-SAR',
        'predictive_entropy': 'Pred. Entropy'
    }
    labels = [display_names.get(m, m) for m in methods]

    colors = ['#2ecc71' if m == 'ag_sar' else '#3498db' for m in methods]
    bars = ax.bar(labels, eces, color=colors, edgecolor='black')

    for bar, val in zip(bars, eces):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('ECE')
    ax.set_title('Calibration Error (Lower = Better)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=7)


def _plot_latency_breakdown(ax, results: Dict):
    """Plot latency breakdown stacked bar."""
    seq_lengths = list(results.get('seq_lengths', {}).keys())

    if not seq_lengths:
        ax.text(0.5, 0.5, 'No latency data', ha='center', va='center')
        return

    components = ['1_forward_pass', '2_reconstruction', '3_centrality']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    labels = ['Forward', 'Reconstruct', 'Centrality']

    x = np.arange(len(seq_lengths))
    width = 0.6
    bottoms = np.zeros(len(seq_lengths))

    for comp, color, label in zip(components, colors, labels):
        values = [
            results['seq_lengths'][sl].get(comp, {}).get('mean_ms', 0)
            for sl in seq_lengths
        ]
        ax.bar(x, values, width, bottom=bottoms, label=label, color=color)
        bottoms += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Decomposition')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')


def _plot_throughput_bars(ax, results: Dict):
    """Plot throughput comparison bars."""
    methods = list(results.get('methods', {}).keys())
    tps = [results['methods'][m].get('tokens_per_second', 0) for m in methods]

    display_names = {
        'vanilla': 'Vanilla',
        'ag_sar': 'AG-SAR',
        'original_sar': 'Orig. SAR',
        'predictive_entropy': 'Pred. Entropy'
    }
    labels = [display_names.get(m, m) for m in methods]

    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#3498db']
    bars = ax.bar(labels, tps, color=colors[:len(methods)], edgecolor='black')

    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Tokens/Second')
    ax.set_title('Throughput Comparison')
    ax.grid(True, alpha=0.3, axis='y')


def _plot_ablation_bars(ax, results: Dict):
    """Plot ablation study bars."""
    configs = list(results.get('configurations', {}).keys())
    aurocs = [results['configurations'][c].get('auroc', 0) for c in configs]

    display_names = {
        'full': 'Full',
        'no_residual': 'No Resid.',
        'no_head_filter': 'No Filter',
        'no_value_norms': 'No Norms',
        'uniform_graph': 'Uniform'
    }
    labels = [display_names.get(c, c) for c in configs]

    colors = ['#2ecc71' if c == 'full' else '#f39c12' for c in configs]
    bars = ax.bar(labels, aurocs, color=colors, edgecolor='black')

    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_ylabel('AUROC')
    ax.set_title('Ablation Study')
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


def _plot_success_summary(ax, results: Dict):
    """Plot pass/fail summary for all experiments."""
    experiments = [
        ('Sink Awareness', results.get('exp1_sink_awareness', {})),
        ('POS Correlation', results.get('exp2_pos_correlation', {})),
        ('AUROC', results.get('exp3_auroc', {})),
        ('Calibration', results.get('exp4_calibration', {})),
        ('Latency', results.get('exp5_latency', {})),
        ('Throughput', results.get('exp6_throughput', {})),
        ('Ablation', results.get('exp7_ablation', {}))
    ]

    labels = []
    colors = []
    values = []

    for name, data in experiments:
        success = data.get('success', None)
        labels.append(name)
        if success is None:
            colors.append('#95a5a6')  # Gray for not run
            values.append(0)
        elif success:
            colors.append('#2ecc71')  # Green for pass
            values.append(1)
        else:
            colors.append('#e74c3c')  # Red for fail
            values.append(-1)

    # Horizontal bar chart
    y = np.arange(len(labels))
    bars = ax.barh(y, [1] * len(labels), color=colors, edgecolor='black')

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])

    # Add status text
    for i, (val, color) in enumerate(zip(values, colors)):
        if val == 1:
            text = 'PASS'
        elif val == -1:
            text = 'FAIL'
        else:
            text = 'N/A'
        ax.text(0.5, i, text, ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)

    ax.set_title('Experiment Results')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_method_comparison(
    methods: Dict[str, float],
    metric_name: str = "Performance",
    title: str = "Method Comparison",
    higher_is_better: bool = True,
    threshold: Optional[float] = None,
    save_path: Optional[Path] = None
):
    """
    Generic bar chart for comparing methods.

    Args:
        methods: Dict mapping method name to metric value
        metric_name: Y-axis label
        title: Plot title
        higher_is_better: Whether higher values are better
        threshold: Optional threshold line to draw
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(methods.keys())
    values = list(methods.values())

    # Color by rank
    if higher_is_better:
        best_idx = np.argmax(values)
    else:
        best_idx = np.argmin(values)

    colors = ['#2ecc71' if i == best_idx else '#3498db' for i in range(len(names))]

    bars = ax.bar(names, values, color=colors, edgecolor='black')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Threshold ({threshold})')
        ax.legend()

    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_results_table(
    results: Dict,
    save_path: Optional[Path] = None
) -> str:
    """
    Create formatted text table of results.

    Args:
        results: Dict with experiment results
        save_path: Path to save table as text

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("AG-SAR EVALUATION RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append("")

    # Header
    lines.append(f"{'Experiment':<25} {'Metric':<20} {'Value':>10} {'Status':>10}")
    lines.append("-" * 70)

    # Exp 1: Sink Awareness
    if 'exp1_sink_awareness' in results:
        data = results['exp1_sink_awareness']
        sink_mass = data.get('summary', {}).get('sink_aware', {}).get('special_token_mass_mean', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'1. Sink Awareness':<25} {'Special Token Mass':<20} {sink_mass:>9.1%} {status:>10}")

    # Exp 2: POS Correlation
    if 'exp2_pos_correlation' in results:
        data = results['exp2_pos_correlation']
        corr = data.get('summary', {}).get('mean_correlation', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'2. POS Correlation':<25} {'Spearman ρ':<20} {corr:>10.3f} {status:>10}")

    # Exp 3: AUROC
    if 'exp3_auroc' in results:
        data = results['exp3_auroc']
        auroc = data.get('methods', {}).get('ag_sar', {}).get('auroc', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'3. Hallucination AUROC':<25} {'AUROC':<20} {auroc:>10.3f} {status:>10}")

    # Exp 4: Calibration
    if 'exp4_calibration' in results:
        data = results['exp4_calibration']
        ece = data.get('methods', {}).get('ag_sar', {}).get('ece', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'4. Calibration':<25} {'ECE':<20} {ece:>10.4f} {status:>10}")

    # Exp 5: Latency
    if 'exp5_latency' in results:
        data = results['exp5_latency']
        overhead = data.get('summary', {}).get('overhead_pct', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'5. Latency Overhead':<25} {'Overhead %':<20} {overhead:>9.1f}% {status:>10}")

    # Exp 6: Throughput
    if 'exp6_throughput' in results:
        data = results['exp6_throughput']
        ratio = data.get('ag_sar_vanilla_ratio', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'6. Throughput':<25} {'AG-SAR/Vanilla':<20} {ratio:>10.2f} {status:>10}")

    # Exp 7: Ablation
    if 'exp7_ablation' in results:
        data = results['exp7_ablation']
        full_auroc = data.get('configurations', {}).get('full', {}).get('auroc', 0)
        status = "PASS" if data.get('success', False) else "FAIL"
        lines.append(f"{'7. Ablation':<25} {'Full AUROC':<20} {full_auroc:>10.3f} {status:>10}")

    lines.append("-" * 70)

    # Count passes
    total = sum(1 for k in results if k.startswith('exp'))
    passes = sum(1 for k, v in results.items() if k.startswith('exp') and v.get('success', False))
    lines.append(f"\nOverall: {passes}/{total} experiments passed")
    lines.append("=" * 70)

    table = "\n".join(lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(table)

    return table
