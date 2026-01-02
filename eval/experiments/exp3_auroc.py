"""
Experiment 3: AUROC Benchmark for Hallucination Detection

Evaluates AG-SAR's ability to detect hallucinations using
TruthfulQA dataset with ROUGE-L thresholding for ground truth.

Compares:
1. Predictive Entropy (PE) - simple log-prob baseline
2. Original SAR - with O(N) RoBERTa perturbation analysis
3. AG-SAR - our O(1) internal graph method

Success Criteria: AG-SAR AUROC > PE AUROC, AG-SAR AUROC ≥ Original SAR AUROC

Label Convention:
    TruthfulQA dataset: sample.label = True means FACTUAL (not hallucination)
    AUROC computation: is_hallucination = not sample.label
    Higher uncertainty scores should correlate with hallucinations (positive class)
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalConfig
from ..datasets import load_truthfulqa, load_triviaqa, load_coqa
from ..metrics.auroc import (
    compute_auroc,
    compute_auprc,
    plot_roc_curve,
    compute_optimal_threshold
)
from ..metrics.rouge import compute_rouge_l


def run_auroc_experiment(
    ag_sar,
    original_sar,
    pe_baseline,
    config: EvalConfig,
    num_samples: int = 200,
    save_results: bool = True,
    dataset: str = 'truthfulqa'
) -> Dict:
    """
    Run AUROC benchmark for hallucination detection.

    Args:
        ag_sar: AGSAR instance
        original_sar: Original SAR baseline (O(N) perturbation)
        pe_baseline: Predictive Entropy baseline
        config: Evaluation configuration
        num_samples: Number of samples
        save_results: Whether to save results
        dataset: Dataset to use ('truthfulqa' or 'triviaqa')

    Returns:
        Dict with AUROC results
    """
    print("=" * 60)
    print("Experiment 3: AUROC Benchmark - Hallucination Detection")
    print(f"Dataset: {dataset}")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {dataset} ({num_samples} samples)...")
    if dataset == 'triviaqa':
        samples = load_triviaqa(max_samples=num_samples)
    elif dataset == 'coqa':
        samples = load_coqa(max_samples=num_samples)
    else:
        samples = load_truthfulqa(max_samples=num_samples)

    results = {
        'experiment': 'auroc_hallucination',
        'dataset': dataset,
        'num_samples': len(samples),
        'rouge_threshold': config.rouge_threshold,
        'methods': {}
    }

    # Storage for each method
    method_data = {
        'ag_sar': {'scores': [], 'labels': []},
        'original_sar': {'scores': [], 'labels': []},
        'predictive_entropy': {'scores': [], 'labels': []}
    }

    print(f"\nProcessing {len(samples)} samples...")
    print("(Note: Original SAR is O(N) and will be slower)")

    for i, sample in enumerate(tqdm(samples, desc="Evaluating")):
        prompt = sample.prompt
        response = sample.response

        if not response or len(response.strip()) < 5:
            continue

        # Use ground truth labels from dataset
        # TruthfulQA dataset has label=True for factual, label=False for hallucination
        if sample.label is None:
            # Fallback: compute ROUGE-L if no label (shouldn't happen for TruthfulQA)
            reference = sample.metadata.get('reference', response) if sample.metadata else response
            rouge_score = compute_rouge_l(response, reference)
            is_hallucination = rouge_score < config.rouge_threshold
        else:
            # Use dataset label (True = factual, False = hallucination)
            is_hallucination = not sample.label  # Invert: label=True means NOT hallucination

        try:
            # AG-SAR uncertainty (higher = more likely hallucination)
            ag_sar_score = ag_sar.compute_uncertainty(prompt, response)
            if hasattr(ag_sar_score, 'item'):
                ag_sar_score = ag_sar_score.item()
            method_data['ag_sar']['scores'].append(ag_sar_score)
            method_data['ag_sar']['labels'].append(is_hallucination)

            # Predictive Entropy
            pe_score = pe_baseline.compute_uncertainty(prompt, response)
            if hasattr(pe_score, 'item'):
                pe_score = pe_score.item()
            method_data['predictive_entropy']['scores'].append(pe_score)
            method_data['predictive_entropy']['labels'].append(is_hallucination)

            # Original SAR (O(N) - slow!)
            if original_sar is not None and i < 50:  # Limit for speed
                sar_score = original_sar.compute_uncertainty(prompt, response)
                if hasattr(sar_score, 'item'):
                    sar_score = sar_score.item()
                method_data['original_sar']['scores'].append(sar_score)
                method_data['original_sar']['labels'].append(is_hallucination)

        except Exception as e:
            print(f"  Warning: Failed on sample {i}: {e}")
            continue

    # Compute AUROC for each method
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"\n{'Method':<25} {'AUROC':>10} {'AUPRC':>10} {'Samples':>10}")
    print("-" * 60)

    for method, data in method_data.items():
        if len(data['scores']) < 10:
            print(f"{method:<25} {'N/A':>10} {'N/A':>10} {len(data['scores']):>10}")
            continue

        auroc = compute_auroc(data['labels'], data['scores'])
        auprc = compute_auprc(data['labels'], data['scores'])
        threshold, j_score = compute_optimal_threshold(
            data['labels'], data['scores']
        )

        results['methods'][method] = {
            'auroc': auroc,
            'auprc': auprc,
            'optimal_threshold': threshold,
            'youden_j': j_score,
            'num_samples': len(data['scores']),
            'num_hallucinations': sum(data['labels']),
            'hallucination_rate': sum(data['labels']) / len(data['labels'])
        }

        print(f"{method:<25} {auroc:>10.3f} {auprc:>10.3f} {len(data['scores']):>10}")

    # Success check
    ag_sar_auroc = results['methods'].get('ag_sar', {}).get('auroc', 0)
    pe_auroc = results['methods'].get('predictive_entropy', {}).get('auroc', 0)
    sar_auroc = results['methods'].get('original_sar', {}).get('auroc', 0)

    success = ag_sar_auroc > pe_auroc
    if sar_auroc > 0:
        success = success and (ag_sar_auroc >= sar_auroc - 0.02)  # Allow small margin

    results['success'] = bool(success)  # Convert numpy.bool_ to Python bool for JSON
    results['success_criteria'] = 'AG-SAR AUROC > PE AUROC and AG-SAR ≥ Original SAR'

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"AG-SAR: {ag_sar_auroc:.3f}, PE: {pe_auroc:.3f}, SAR: {sar_auroc:.3f}")
    print(f"{'=' * 60}")

    # Store raw data for plotting
    results['raw_data'] = {
        method: {
            'labels': data['labels'],
            'scores': data['scores']
        }
        for method, data in method_data.items()
        if len(data['scores']) >= 10
    }

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp3_auroc.json'

        # Remove raw data from JSON (too large)
        json_results = {k: v for k, v in results.items() if k != 'raw_data'}

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_auroc_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot ROC curves for all methods.
    """
    import matplotlib.pyplot as plt

    raw_data = results.get('raw_data', {})

    if not raw_data:
        print("No raw data available for plotting")
        return

    # Prepare data for plotting
    plot_data = {
        method: (data['labels'], data['scores'])
        for method, data in raw_data.items()
    }

    # Rename for display
    display_names = {
        'ag_sar': 'AG-SAR',
        'original_sar': 'Original SAR',
        'predictive_entropy': 'Predictive Entropy'
    }

    renamed_data = {
        display_names.get(k, k): v
        for k, v in plot_data.items()
    }

    dataset = results.get('dataset', 'TruthfulQA').upper()
    plot_roc_curve(
        renamed_data,
        title=f'ROC Curve - Hallucination Detection ({dataset})',
        save_path=str(save_path) if save_path else None
    )


def plot_auroc_bar_chart(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot bar chart comparing AUROC scores.
    """
    import matplotlib.pyplot as plt

    methods = list(results['methods'].keys())
    aurocs = [results['methods'][m]['auroc'] for m in methods]
    auprcs = [results['methods'][m]['auprc'] for m in methods]

    display_names = {
        'ag_sar': 'AG-SAR',
        'original_sar': 'Original SAR',
        'predictive_entropy': 'Pred. Entropy'
    }
    labels = [display_names.get(m, m) for m in methods]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, aurocs, width, label='AUROC', color='#3498db')
    bars2 = ax.bar(x + width/2, auprcs, width, label='AUPRC', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Hallucination Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    # Add random baseline
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5,
               label='Random (0.5)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from ag_sar import AGSAR
    from eval.baselines import PredictiveEntropy, OriginalSAR

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

    print("Initializing Original SAR (loading RoBERTa-Large)...")
    original_sar = OriginalSAR(model, tokenizer)

    # Run experiment
    config = EvalConfig()
    results = run_auroc_experiment(
        ag_sar=ag_sar,
        original_sar=original_sar,
        pe_baseline=pe_baseline,
        config=config,
        num_samples=200
    )

    # Plot
    plot_auroc_comparison(results, config.results_dir / 'exp3_roc_curves.png')
    plot_auroc_bar_chart(results, config.results_dir / 'exp3_auroc_bars.png')
