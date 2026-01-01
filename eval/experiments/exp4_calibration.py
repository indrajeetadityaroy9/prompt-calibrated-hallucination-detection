"""
Experiment 4: Calibration Error (ECE) Benchmark

Evaluates how well AG-SAR's confidence scores match actual accuracy.
A well-calibrated model should have confidence = accuracy for each bin.

Key Metric: Expected Calibration Error (ECE)
ECE = Σ (|bin| / n) × |accuracy(bin) - confidence(bin)|

Success Criteria: ECE < 0.15 (lower is better)
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalConfig
from ..datasets import load_truthfulqa
from ..metrics.calibration import (
    compute_ece,
    compute_mce,
    get_calibration_curve,
    plot_reliability_diagram,
    uncertainty_to_confidence
)
from .exp3_auroc import compute_rouge_l


def run_calibration_experiment(
    ag_sar,
    pe_baseline,
    config: EvalConfig,
    num_samples: int = 300,
    save_results: bool = True
) -> Dict:
    """
    Run calibration error benchmark.

    Measures Expected Calibration Error (ECE) and Maximum Calibration
    Error (MCE) for AG-SAR and baselines.

    Args:
        ag_sar: AGSAR instance
        pe_baseline: Predictive Entropy baseline
        config: Evaluation configuration
        num_samples: Number of TruthfulQA samples
        save_results: Whether to save results

    Returns:
        Dict with calibration results
    """
    print("=" * 60)
    print("Experiment 4: Calibration Error (ECE)")
    print("=" * 60)

    # Load TruthfulQA
    print(f"\nLoading TruthfulQA ({num_samples} samples)...")
    samples = load_truthfulqa(max_samples=num_samples)

    results = {
        'experiment': 'calibration_ece',
        'num_samples': len(samples),
        'num_bins': config.ece_num_bins,
        'rouge_threshold': config.rouge_threshold,
        'methods': {}
    }

    # Storage for each method
    method_data = {
        'ag_sar': {'uncertainties': [], 'accuracies': []},
        'predictive_entropy': {'uncertainties': [], 'accuracies': []}
    }

    print(f"\nProcessing {len(samples)} samples...")

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        prompt = sample.prompt
        response = sample.response or ""

        if not response or len(response.strip()) < 5:
            continue

        # Ground truth: use dataset label (True = factual/accurate)
        if sample.label is not None:
            is_accurate = sample.label  # True = factual, False = hallucination
        else:
            # Fallback if no label
            is_accurate = True

        try:
            # AG-SAR uncertainty
            ag_sar_score = ag_sar.compute_uncertainty(prompt, response)
            if hasattr(ag_sar_score, 'item'):
                ag_sar_score = ag_sar_score.item()
            method_data['ag_sar']['uncertainties'].append(ag_sar_score)
            method_data['ag_sar']['accuracies'].append(is_accurate)

            # Predictive Entropy
            pe_score = pe_baseline.compute_uncertainty(prompt, response)
            if hasattr(pe_score, 'item'):
                pe_score = pe_score.item()
            method_data['predictive_entropy']['uncertainties'].append(pe_score)
            method_data['predictive_entropy']['accuracies'].append(is_accurate)

        except Exception as e:
            print(f"  Warning: Failed on sample {i}: {e}")
            continue

    # Compute calibration metrics for each method
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"\n{'Method':<25} {'ECE':>10} {'MCE':>10} {'Samples':>10}")
    print("-" * 60)

    for method, data in method_data.items():
        if len(data['uncertainties']) < 20:
            print(f"{method:<25} {'N/A':>10} {'N/A':>10} {len(data['uncertainties']):>10}")
            continue

        # Convert uncertainties to confidences
        confidences = uncertainty_to_confidence(
            data['uncertainties'],
            method='normalize'
        )

        ece = compute_ece(
            confidences,
            data['accuracies'],
            num_bins=config.ece_num_bins
        )
        mce = compute_mce(
            confidences,
            data['accuracies'],
            num_bins=config.ece_num_bins
        )

        # Get calibration curve data
        bin_centers, bin_accs, bin_confs = get_calibration_curve(
            confidences,
            data['accuracies'],
            num_bins=config.ece_num_bins
        )

        results['methods'][method] = {
            'ece': float(ece),
            'mce': float(mce),
            'num_samples': len(data['uncertainties']),
            'accuracy_rate': sum(data['accuracies']) / len(data['accuracies']),
            'calibration_curve': {
                'bin_centers': bin_centers.tolist(),
                'bin_accuracies': bin_accs.tolist(),
                'bin_confidences': bin_confs.tolist()
            }
        }

        print(f"{method:<25} {ece:>10.4f} {mce:>10.4f} {len(data['uncertainties']):>10}")

    # Success check
    ag_sar_ece = results['methods'].get('ag_sar', {}).get('ece', 1.0)
    success = ag_sar_ece < 0.15

    results['success'] = success
    results['success_criteria'] = 'AG-SAR ECE < 0.15'

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"AG-SAR ECE: {ag_sar_ece:.4f} (threshold: 0.15)")
    print(f"{'=' * 60}")

    # Store raw data for plotting
    results['raw_data'] = {
        method: {
            'confidences': uncertainty_to_confidence(data['uncertainties'], 'normalize'),
            'accuracies': data['accuracies']
        }
        for method, data in method_data.items()
        if len(data['uncertainties']) >= 20
    }

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp4_calibration.json'

        # Remove raw data for JSON
        json_results = {k: v for k, v in results.items() if k != 'raw_data'}

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_calibration_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot reliability diagrams for all methods.
    """
    import matplotlib.pyplot as plt

    methods = list(results['methods'].keys())
    n_methods = len(methods)

    if n_methods == 0:
        print("No methods to plot")
        return

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    display_names = {
        'ag_sar': 'AG-SAR',
        'predictive_entropy': 'Predictive Entropy'
    }

    colors = ['#3498db', '#e74c3c']

    for i, method in enumerate(methods):
        ax = axes[i]
        data = results['methods'][method]
        curve = data['calibration_curve']

        bin_centers = np.array(curve['bin_centers'])
        bin_accs = np.array(curve['bin_accuracies'])
        bin_confs = np.array(curve['bin_confidences'])

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=1)

        # Calibration curve as bars
        num_bins = len(bin_centers)
        width = 1.0 / num_bins * 0.8

        bars = ax.bar(bin_centers, bin_accs, width=width, alpha=0.7,
                     color=colors[i % len(colors)], edgecolor='black',
                     label=f"ECE = {data['ece']:.3f}")

        # Gap visualization
        for bc, ba, bconf in zip(bin_centers, bin_accs, bin_confs):
            if ba < bconf:
                ax.fill_between([bc - width/2, bc + width/2],
                              ba, bconf, alpha=0.2, color='red')
            else:
                ax.fill_between([bc - width/2, bc + width/2],
                              bconf, ba, alpha=0.2, color='green')

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{display_names.get(method, method)}')
        ax.legend(loc='upper left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ece_bar_chart(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot bar chart comparing ECE across methods.
    """
    import matplotlib.pyplot as plt

    methods = list(results['methods'].keys())
    eces = [results['methods'][m]['ece'] for m in methods]
    mces = [results['methods'][m]['mce'] for m in methods]

    display_names = {
        'ag_sar': 'AG-SAR',
        'predictive_entropy': 'Pred. Entropy'
    }
    labels = [display_names.get(m, m) for m in methods]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, eces, width, label='ECE', color='#3498db')
    bars2 = ax.bar(x + width/2, mces, width, label='MCE', color='#e74c3c')

    ax.set_ylabel('Calibration Error')
    ax.set_title('Calibration Error Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add threshold line
    ax.axhline(y=0.15, color='green', linestyle='--', alpha=0.5,
               label='ECE Threshold (0.15)')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from ag_sar import AGSAR
    from eval.baselines import PredictiveEntropy

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    # Initialize methods
    print("Initializing AG-SAR...")
    ag_sar = AGSAR(model, tokenizer)

    print("Initializing Predictive Entropy...")
    pe_baseline = PredictiveEntropy(model, tokenizer)

    # Run experiment
    config = EvalConfig()
    results = run_calibration_experiment(
        ag_sar=ag_sar,
        pe_baseline=pe_baseline,
        config=config,
        num_samples=300
    )

    # Plot
    plot_calibration_comparison(results, config.results_dir / 'exp4_reliability.png')
    plot_ece_bar_chart(results, config.results_dir / 'exp4_ece_bars.png')
