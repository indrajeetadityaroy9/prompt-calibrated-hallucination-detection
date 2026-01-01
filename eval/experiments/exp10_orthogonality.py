"""
Experiment 10: The Orthogonality Test

Goal: Prove that the Attention Graph (R) provides information that Logits (S) do not contain.

Method:
1. Run on TruthfulQA (Adversarial) and TriviaQA (Factual)
2. For each sample, compute:
   - Pure Surprisal (NLL) - Logits only
   - Pure Centrality (Gini) - Graph only
   - AG-SAR (TWS) - Combined
3. Generate 2D Scatter Plot: X=Gini (Sharpness), Y=Surprisal (Confidence)
4. Color code by Fact vs Hallucination

Hypothesis: Hallucinations cluster in "High Surprisal / Low Gini" quadrant
(Confident Rambling) - a region neither metric detects alone.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.uncertainty import compute_token_surprisal, compute_graph_shifted_surprisal
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.datasets import load_truthfulqa, load_triviaqa
from eval.config import EvalConfig


def gini_coefficient(x: np.ndarray) -> float:
    """Compute Gini coefficient - measure of inequality/sharpness."""
    x = np.abs(x).flatten()
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def run_orthogonality_test(
    model_id: str = "meta-llama/Llama-3.2-3B",
    num_samples: int = 200,
    save_results: bool = True,
    save_plots: bool = True
) -> Dict:
    """
    Run the Orthogonality Test: Compare Structure vs Statistics.
    """
    print("="*70)
    print("EXPERIMENT 10: ORTHOGONALITY TEST")
    print("Structure (Gini) vs Statistics (Surprisal)")
    print("="*70)

    # Load model
    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, config)

    results = {
        'model': model_id,
        'datasets': {}
    }

    for dataset_name in ['truthfulqa', 'triviaqa']:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*60}")

        # Load dataset
        if dataset_name == 'truthfulqa':
            samples = load_truthfulqa(max_samples=num_samples)
        else:
            samples = load_triviaqa(max_samples=num_samples)

        data_points = []

        for sample in tqdm(samples, desc=f"Processing {dataset_name}"):
            if not sample.response or len(sample.response.strip()) < 3:
                continue

            try:
                # Get AG-SAR details
                details = ag_sar.compute_uncertainty(
                    sample.prompt, sample.response, return_details=True
                )

                relevance = details['relevance']
                response_start = details['response_start']
                input_ids = details['input_ids']

                # Get logits for surprisal
                text = sample.prompt + sample.response
                inputs = tokenizer(text, return_tensors="pt").to("cuda")

                with torch.no_grad():
                    outputs = model(input_ids=inputs.input_ids)
                    logits = outputs.logits.float()

                # Compute surprisal
                surprisal = compute_token_surprisal(logits, inputs.input_ids)

                # Extract response-only metrics
                response_relevance = relevance[:, response_start:]
                response_surprisal = surprisal[:, response_start:]

                # Create response mask
                response_mask = torch.zeros_like(inputs.attention_mask)
                response_mask[:, response_start:] = 1

                # === Metric 1: Pure Surprisal (avg NLL) ===
                avg_surprisal = response_surprisal.mean().item()

                # === Metric 2: Pure Gini (graph sharpness) ===
                rel_np = response_relevance[0].cpu().numpy()
                gini = gini_coefficient(rel_np)

                # === Metric 3: TWS (combined) ===
                tws = compute_graph_shifted_surprisal(
                    surprisal, relevance, attention_mask=response_mask
                ).item()

                # Ground truth label
                is_factual = sample.label if sample.label is not None else True

                data_points.append({
                    'surprisal': avg_surprisal,
                    'gini': gini,
                    'tws': tws,
                    'is_factual': is_factual,
                    'response_len': len(sample.response)
                })

            except Exception as e:
                continue

        results['datasets'][dataset_name] = {
            'num_samples': len(data_points),
            'data_points': data_points
        }

        # Compute separation metrics
        factual = [p for p in data_points if p['is_factual']]
        halluc = [p for p in data_points if not p['is_factual']]

        if factual and halluc:
            print(f"\nFactual samples: {len(factual)}")
            print(f"Hallucination samples: {len(halluc)}")

            # Surprisal separation
            fact_surp = np.mean([p['surprisal'] for p in factual])
            hall_surp = np.mean([p['surprisal'] for p in halluc])
            print(f"\nSurprisal - Factual: {fact_surp:.3f}, Halluc: {hall_surp:.3f}")

            # Gini separation
            fact_gini = np.mean([p['gini'] for p in factual])
            hall_gini = np.mean([p['gini'] for p in halluc])
            print(f"Gini - Factual: {fact_gini:.4f}, Halluc: {hall_gini:.4f}")

            # TWS separation
            fact_tws = np.mean([p['tws'] for p in factual])
            hall_tws = np.mean([p['tws'] for p in halluc])
            print(f"TWS - Factual: {fact_tws:.3f}, Halluc: {hall_tws:.3f}")

            results['datasets'][dataset_name]['stats'] = {
                'factual_surprisal': fact_surp,
                'halluc_surprisal': hall_surp,
                'factual_gini': fact_gini,
                'halluc_gini': hall_gini,
                'factual_tws': fact_tws,
                'halluc_tws': hall_tws
            }

    ag_sar.cleanup()

    # Save results
    if save_results:
        results_dir = Path(__file__).parent.parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        # Convert for JSON
        json_results = {
            'model': results['model'],
            'datasets': {}
        }
        for ds_name, ds_data in results['datasets'].items():
            json_results['datasets'][ds_name] = {
                'num_samples': ds_data['num_samples'],
                'stats': ds_data.get('stats', {}),
                'data_points': ds_data['data_points']
            }

        with open(results_dir / 'exp10_orthogonality.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_dir / 'exp10_orthogonality.json'}")

    # Generate plots
    if save_plots:
        plot_orthogonality(results, results_dir)

    return results


def plot_orthogonality(results: Dict, save_dir: Path):
    """Generate the 2D scatter plot (Gini vs Surprisal)."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (dataset_name, ds_data) in enumerate(results['datasets'].items()):
        ax = axes[idx]
        data_points = ds_data['data_points']

        factual = [p for p in data_points if p['is_factual']]
        halluc = [p for p in data_points if not p['is_factual']]

        # Plot factual (blue)
        if factual:
            ax.scatter(
                [p['gini'] for p in factual],
                [p['surprisal'] for p in factual],
                c='#3498db', alpha=0.6, s=50, label='Factual',
                edgecolors='white', linewidth=0.5
            )

        # Plot hallucinations (red)
        if halluc:
            ax.scatter(
                [p['gini'] for p in halluc],
                [p['surprisal'] for p in halluc],
                c='#e74c3c', alpha=0.6, s=50, label='Hallucination',
                edgecolors='white', linewidth=0.5
            )

        ax.set_xlabel('Gini Coefficient (Graph Sharpness) →', fontsize=11)
        ax.set_ylabel('Surprisal (NLL) →', fontsize=11)
        ax.set_title(f'{dataset_name.upper()}: Structure vs Statistics', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Add quadrant annotations
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmid = (xlim[0] + xlim[1]) / 2
        ymid = (ylim[0] + ylim[1]) / 2

        # Annotate quadrants
        ax.annotate('Grounded\nConfident', xy=(xlim[1]*0.85, ylim[0]*1.1),
                   fontsize=9, ha='center', color='green', alpha=0.7)
        ax.annotate('Ungrounded\nUncertain', xy=(xlim[0]*1.1, ylim[1]*0.9),
                   fontsize=9, ha='center', color='red', alpha=0.7)

    plt.suptitle('Orthogonality Test: Do Structure and Statistics Provide Independent Signals?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'exp10_orthogonality_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {save_dir / 'exp10_orthogonality_scatter.png'}")

    # Also plot AUROC comparison
    plot_auroc_comparison(results, save_dir)


def plot_auroc_comparison(results: Dict, save_dir: Path):
    """Plot AUROC comparison across metrics."""
    from sklearn.metrics import roc_auc_score

    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['surprisal', 'gini', 'tws']
    metric_names = ['Pure Surprisal\n(Logits Only)', 'Pure Gini\n(Graph Only)', 'TWS\n(Combined)']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    x = np.arange(len(results['datasets']))
    width = 0.25

    for m_idx, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
        aurocs = []
        for ds_name, ds_data in results['datasets'].items():
            data_points = ds_data['data_points']
            if len(data_points) < 10:
                aurocs.append(0.5)
                continue

            labels = [0 if p['is_factual'] else 1 for p in data_points]
            scores = [p[metric] for p in data_points]

            if len(set(labels)) < 2:
                aurocs.append(0.5)
            else:
                auroc = roc_auc_score(labels, scores)
                aurocs.append(auroc)

        bars = ax.bar(x + m_idx * width, aurocs, width, label=name, color=color, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, aurocs):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=9)

    ax.set_ylabel('AUROC')
    ax.set_title('Hallucination Detection: Structure vs Statistics vs Combined')
    ax.set_xticks(x + width)
    ax.set_xticklabels(list(results['datasets'].keys()))
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_dir / 'exp10_auroc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"AUROC comparison saved to: {save_dir / 'exp10_auroc_comparison.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Orthogonality Test")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--no-plots', action='store_true')

    args = parser.parse_args()

    results = run_orthogonality_test(
        model_id=args.model,
        num_samples=args.samples,
        save_results=True,
        save_plots=not args.no_plots
    )
