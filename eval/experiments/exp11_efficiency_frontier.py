"""
Experiment 11: The Efficiency Frontier (Cost vs. Capability)

Goal: Compare AG-SAR against current SOTA 'Semantic' methods.

Baselines:
1. Predictive Entropy (PE): Single forward pass O(1N) - fast but naive
2. Semantic Entropy (SE): K=5 generations + clustering O(5N) - expensive
3. EigenScore: K=5 generations + covariance O(5N) - expensive
4. AG-SAR (TWS): Single forward pass O(1N) - fast and semantic

Metrics:
- AUROC: Hallucination detection accuracy
- TPS: Tokens per second (throughput)

Output: Pareto frontier plot (AUROC vs TPS)

Hypothesis: AG-SAR appears in Top-Left quadrant (High Accuracy, Max Speed)
"""

import torch
import numpy as np
from pathlib import Path
import json
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.uncertainty import compute_token_surprisal, compute_graph_shifted_surprisal
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.baselines.predictive_entropy import PredictiveEntropy
from eval.baselines.semantic_entropy import SemanticEntropy
from eval.baselines.eigenscore import EigenScore
from eval.datasets import load_triviaqa


def run_efficiency_frontier(
    model_id: str = "meta-llama/Llama-3.2-3B",
    num_samples: int = 100,
    k_samples: int = 5,  # For SE and EigenScore
    max_new_tokens: int = 15,
    warmup_samples: int = 5,
    save_results: bool = True,
    save_plots: bool = True
) -> Dict:
    """
    Run the Efficiency Frontier benchmark.

    Compares AUROC and Throughput (TPS) across methods.
    """
    print("=" * 70)
    print("EXPERIMENT 11: EFFICIENCY FRONTIER")
    print("Cost vs Capability Analysis")
    print("=" * 70)

    # Load model
    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Initialize methods
    print("Initializing methods...")
    config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, config)
    pe = PredictiveEntropy(model, tokenizer)
    se = SemanticEntropy(model, tokenizer, k_samples=k_samples, temperature=0.7)
    es = EigenScore(model, tokenizer, k_samples=k_samples, temperature=0.7)

    # Load dataset
    print(f"\nLoading TriviaQA ({num_samples} samples)...")
    samples = load_triviaqa(max_samples=num_samples)

    # Prepare prompts and responses
    prompts = [s.prompt for s in samples]
    responses = [s.response for s in samples]
    labels = [1 if s.label else 0 for s in samples]  # 1 = factual, 0 = hallucination

    results = {
        'model': model_id,
        'num_samples': len(samples),
        'k_samples': k_samples,
        'methods': {}
    }

    # === BENCHMARK EACH METHOD ===

    # Method 1: Predictive Entropy (PE)
    print("\n" + "=" * 60)
    print("Benchmarking: Predictive Entropy (PE)")
    print("=" * 60)
    pe_scores, pe_time = benchmark_predictive_entropy(
        pe, prompts, responses, warmup_samples
    )
    pe_auroc = compute_auroc(pe_scores, labels)
    pe_tps = compute_tps(prompts, responses, pe_time, tokenizer)

    results['methods']['pe'] = {
        'name': 'Predictive Entropy',
        'auroc': pe_auroc,
        'tps': pe_tps,
        'total_time': pe_time,
        'complexity': 'O(1N)'
    }
    print(f"  AUROC: {pe_auroc:.4f}")
    print(f"  TPS: {pe_tps:.1f}")

    # Method 2: AG-SAR (TWS)
    print("\n" + "=" * 60)
    print("Benchmarking: AG-SAR (TWS)")
    print("=" * 60)
    agsar_scores, agsar_time = benchmark_agsar(
        ag_sar, model, tokenizer, prompts, responses, warmup_samples
    )
    agsar_auroc = compute_auroc(agsar_scores, labels)
    agsar_tps = compute_tps(prompts, responses, agsar_time, tokenizer)

    results['methods']['agsar'] = {
        'name': 'AG-SAR (TWS)',
        'auroc': agsar_auroc,
        'tps': agsar_tps,
        'total_time': agsar_time,
        'complexity': 'O(1N)'
    }
    print(f"  AUROC: {agsar_auroc:.4f}")
    print(f"  TPS: {agsar_tps:.1f}")

    # Method 3: Semantic Entropy (SE) - expensive
    print("\n" + "=" * 60)
    print(f"Benchmarking: Semantic Entropy (K={k_samples})")
    print("=" * 60)
    se_scores, se_time = benchmark_semantic_entropy(
        se, prompts, max_new_tokens, warmup_samples
    )
    se_auroc = compute_auroc(se_scores, labels)
    se_tps = compute_tps(prompts, responses, se_time, tokenizer)

    results['methods']['se'] = {
        'name': f'Semantic Entropy (K={k_samples})',
        'auroc': se_auroc,
        'tps': se_tps,
        'total_time': se_time,
        'complexity': f'O({k_samples}N)'
    }
    print(f"  AUROC: {se_auroc:.4f}")
    print(f"  TPS: {se_tps:.1f}")

    # Method 4: EigenScore - expensive
    print("\n" + "=" * 60)
    print(f"Benchmarking: EigenScore (K={k_samples})")
    print("=" * 60)
    es_scores, es_time = benchmark_eigenscore(
        es, prompts, max_new_tokens, warmup_samples
    )
    es_auroc = compute_auroc(es_scores, labels)
    es_tps = compute_tps(prompts, responses, es_time, tokenizer)

    results['methods']['eigenscore'] = {
        'name': f'EigenScore (K={k_samples})',
        'auroc': es_auroc,
        'tps': es_tps,
        'total_time': es_time,
        'complexity': f'O({k_samples}N)'
    }
    print(f"  AUROC: {es_auroc:.4f}")
    print(f"  TPS: {es_tps:.1f}")

    # Cleanup
    ag_sar.cleanup()

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("EFFICIENCY FRONTIER SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<30} | {'AUROC':>8} | {'TPS':>10} | {'Complexity':>10}")
    print("-" * 70)
    for method_id, method_data in results['methods'].items():
        print(f"{method_data['name']:<30} | {method_data['auroc']:>8.4f} | "
              f"{method_data['tps']:>10.1f} | {method_data['complexity']:>10}")

    # Speedup relative to Semantic Entropy
    if results['methods']['se']['tps'] > 0:
        agsar_speedup = results['methods']['agsar']['tps'] / results['methods']['se']['tps']
        pe_speedup = results['methods']['pe']['tps'] / results['methods']['se']['tps']
        print(f"\nSpeedup vs Semantic Entropy:")
        print(f"  AG-SAR: {agsar_speedup:.1f}x faster")
        print(f"  PE: {pe_speedup:.1f}x faster")

    # Save results
    if save_results:
        results_dir = Path(__file__).parent.parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / 'exp11_efficiency_frontier.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_dir / 'exp11_efficiency_frontier.json'}")

    # Generate plots
    if save_plots:
        plot_efficiency_frontier(results, results_dir)

    return results


def benchmark_predictive_entropy(
    pe: PredictiveEntropy,
    prompts: List[str],
    responses: List[str],
    warmup_samples: int = 5
) -> Tuple[List[float], float]:
    """Benchmark Predictive Entropy."""
    # Warmup
    for i in range(min(warmup_samples, len(prompts))):
        pe.compute_uncertainty(prompts[i], responses[i])

    torch.cuda.synchronize()

    # Benchmark
    scores = []
    start = time.perf_counter()

    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts), desc="PE"):
        score = pe.compute_uncertainty(prompt, response)
        scores.append(score)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return scores, total_time


def benchmark_agsar(
    ag_sar: AGSAR,
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    warmup_samples: int = 5
) -> Tuple[List[float], float]:
    """Benchmark AG-SAR with TWS (Graph-Shifted Surprisal)."""
    # Warmup
    for i in range(min(warmup_samples, len(prompts))):
        ag_sar.compute_uncertainty(prompts[i], responses[i])

    torch.cuda.synchronize()

    # Benchmark
    scores = []
    start = time.perf_counter()

    for prompt, response in tqdm(zip(prompts, responses), total=len(prompts), desc="AG-SAR"):
        # Get relevance from AG-SAR
        details = ag_sar.compute_uncertainty(prompt, response, return_details=True)
        relevance = details['relevance']
        response_start = details['response_start']

        # Get logits for surprisal
        text = prompt + response
        inputs = tokenizer(text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids)
            logits = outputs.logits.float()

        # Compute surprisal
        surprisal = compute_token_surprisal(logits, inputs.input_ids)

        # Create response mask
        response_mask = torch.zeros_like(inputs.attention_mask)
        response_mask[:, response_start:] = 1

        # Compute TWS (Graph-Shifted Surprisal)
        tws = compute_graph_shifted_surprisal(
            surprisal, relevance, attention_mask=response_mask
        ).item()

        scores.append(tws)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return scores, total_time


def benchmark_semantic_entropy(
    se: SemanticEntropy,
    prompts: List[str],
    max_new_tokens: int,
    warmup_samples: int = 5
) -> Tuple[List[float], float]:
    """Benchmark Semantic Entropy (K generations)."""
    # Warmup
    for i in range(min(warmup_samples, len(prompts))):
        se.compute_uncertainty(prompts[i], max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()

    # Benchmark
    scores = []
    start = time.perf_counter()

    for prompt in tqdm(prompts, desc="SE"):
        score = se.compute_uncertainty(prompt, max_new_tokens=max_new_tokens)
        scores.append(score)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return scores, total_time


def benchmark_eigenscore(
    es: EigenScore,
    prompts: List[str],
    max_new_tokens: int,
    warmup_samples: int = 5
) -> Tuple[List[float], float]:
    """Benchmark EigenScore (K generations)."""
    # Warmup
    for i in range(min(warmup_samples, len(prompts))):
        es.compute_uncertainty(prompts[i], max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()

    # Benchmark
    scores = []
    start = time.perf_counter()

    for prompt in tqdm(prompts, desc="EigenScore"):
        score = es.compute_uncertainty(prompt, max_new_tokens=max_new_tokens)
        scores.append(score)

    torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    return scores, total_time


def compute_tps(
    prompts: List[str],
    responses: List[str],
    total_time: float,
    tokenizer
) -> float:
    """Compute tokens per second."""
    total_tokens = sum(
        len(tokenizer.encode(p + r))
        for p, r in zip(prompts, responses)
    )
    return total_tokens / total_time


def compute_auroc(scores: List[float], labels: List[int]) -> float:
    """
    Compute AUROC for hallucination detection.

    Higher score should indicate hallucination (label=0).
    """
    if len(set(labels)) < 2:
        return 0.5

    # Invert labels: we want high score = hallucination
    # labels: 1=factual, 0=hallucination
    # For AUROC, we predict "hallucination" class
    inverted_labels = [1 - l for l in labels]

    try:
        return roc_auc_score(inverted_labels, scores)
    except Exception:
        return 0.5


def plot_efficiency_frontier(results: Dict, save_dir: Path):
    """Generate the AUROC vs TPS Pareto frontier plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define method colors and markers
    method_styles = {
        'pe': {'color': '#3498db', 'marker': 's', 'label': 'Predictive Entropy'},
        'agsar': {'color': '#e74c3c', 'marker': '*', 'label': 'AG-SAR (TWS)'},
        'se': {'color': '#2ecc71', 'marker': 'o', 'label': 'Semantic Entropy'},
        'eigenscore': {'color': '#9b59b6', 'marker': '^', 'label': 'EigenScore'}
    }

    # Plot each method
    for method_id, method_data in results['methods'].items():
        style = method_styles.get(method_id, {'color': 'gray', 'marker': 'x'})
        ax.scatter(
            method_data['tps'],
            method_data['auroc'],
            c=style['color'],
            marker=style['marker'],
            s=300,
            label=f"{style['label']} ({method_data['complexity']})",
            edgecolors='white',
            linewidth=2,
            zorder=5
        )

        # Add annotation
        ax.annotate(
            f"  {method_data['auroc']:.3f}",
            xy=(method_data['tps'], method_data['auroc']),
            fontsize=10,
            va='center'
        )

    # Add quadrant shading
    ax.axhline(y=0.75, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5)

    # Shade "ideal" quadrant (top-right: high AUROC, high TPS)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between(
        [500, xlim[1]], 0.75, ylim[1],
        color='green', alpha=0.1,
        label='Ideal Region'
    )

    ax.set_xlabel('Throughput (Tokens Per Second) →', fontsize=12)
    ax.set_ylabel('AUROC (Detection Accuracy) →', fontsize=12)
    ax.set_title('Efficiency Frontier: Cost vs Capability', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Add annotations for quadrants
    ax.text(0.95, 0.95, 'IDEAL:\nFast + Accurate',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=10, color='green', alpha=0.7)
    ax.text(0.05, 0.05, 'SLOW:\nExpensive Methods',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, color='gray', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_dir / 'exp11_efficiency_frontier.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to: {save_dir / 'exp11_efficiency_frontier.png'}")

    # Also create a bar chart comparison
    plot_comparison_bars(results, save_dir)


def plot_comparison_bars(results: Dict, save_dir: Path):
    """Create side-by-side bar chart for AUROC and TPS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    methods = list(results['methods'].keys())
    method_names = [results['methods'][m]['name'] for m in methods]
    aurocs = [results['methods'][m]['auroc'] for m in methods]
    tps_values = [results['methods'][m]['tps'] for m in methods]

    # Define colors
    colors = {
        'pe': '#3498db',
        'agsar': '#e74c3c',
        'se': '#2ecc71',
        'eigenscore': '#9b59b6'
    }
    bar_colors = [colors.get(m, 'gray') for m in methods]

    # AUROC bar chart
    ax = axes[0]
    x = np.arange(len(methods))
    bars = ax.bar(x, aurocs, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, aurocs):
        ax.annotate(f'{val:.3f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Detection Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, alpha=0.3, axis='y')

    # TPS bar chart
    ax = axes[1]
    bars = ax.bar(x, tps_values, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, tps_values):
        ax.annotate(f'{val:.0f}',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', fontsize=11, fontweight='bold')

    ax.set_ylabel('Tokens Per Second', fontsize=12)
    ax.set_title('Throughput (Speed)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=15, ha='right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Efficiency Frontier: {results["model"]}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / 'exp11_efficiency_bars.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Bar chart saved to: {save_dir / 'exp11_efficiency_bars.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Efficiency Frontier Benchmark")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B')
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--k-samples', type=int, default=5,
                        help='K generations for SE and EigenScore')
    parser.add_argument('--max-new-tokens', type=int, default=15)
    parser.add_argument('--no-plots', action='store_true')

    args = parser.parse_args()

    results = run_efficiency_frontier(
        model_id=args.model,
        num_samples=args.samples,
        k_samples=args.k_samples,
        max_new_tokens=args.max_new_tokens,
        save_results=True,
        save_plots=not args.no_plots
    )
