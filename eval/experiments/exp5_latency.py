"""
Experiment 5: Latency Decomposition

Validates the "zero-latency" claim by decomposing AG-SAR pipeline
into components and measuring each with microsecond precision.

Success Criteria: Reconstruction + Centrality < 10% of Forward Pass
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
import torch

from ..config import EvalConfig, ExperimentResult
from ..profiling.latency import LatencyProfiler, profile_ag_sar_components


def run_latency_experiment(
    ag_sar,
    config: EvalConfig,
    save_results: bool = True
) -> Dict:
    """
    Run latency decomposition experiment.

    Measures time spent on:
    1. Forward Pass (Flash Attention/SDPA)
    2. Hook Overhead (Storing Q/K)
    3. Reconstruction (matmul(Q, K.T))
    4. Graph Centrality (Power Iteration)

    Args:
        ag_sar: AGSAR instance
        config: Evaluation configuration
        save_results: Whether to save results to disk

    Returns:
        Dict with latency breakdown
    """
    print("=" * 60)
    print("Experiment 5: Latency Decomposition")
    print("=" * 60)

    results = {
        'experiment': 'latency_decomposition',
        'seq_lengths': {},
        'summary': {}
    }

    # Test prompts of varying lengths
    for seq_len in config.seq_lengths:
        print(f"\nSequence length: {seq_len}")

        # Generate prompt/response to match target length
        prompt = "The quick brown fox " * (seq_len // 10)
        response = "jumps over the lazy dog. " * (seq_len // 20)

        # Profile components
        component_results = profile_ag_sar_components(
            ag_sar,
            prompt=prompt,
            response=response,
            num_runs=config.benchmark_runs,
            warmup_runs=config.warmup_runs
        )

        # Store results
        seq_results = {
            name: {
                'mean_ms': r.mean_ms,
                'std_ms': r.std_ms,
                'min_ms': r.min_ms,
                'max_ms': r.max_ms
            }
            for name, r in component_results.items()
        }

        # Calculate percentages
        total_time = sum(r.mean_ms for r in component_results.values())
        percentages = {
            name: (r.mean_ms / total_time * 100) if total_time > 0 else 0
            for name, r in component_results.items()
        }

        seq_results['percentages'] = percentages
        seq_results['total_ms'] = total_time

        results['seq_lengths'][seq_len] = seq_results

        # Print results
        print(f"  Total: {total_time:.3f} ms")
        for name, r in component_results.items():
            pct = percentages[name]
            print(f"  {name}: {r.mean_ms:.3f} ms ({pct:.1f}%)")

    # Calculate summary statistics
    forward_pcts = []
    reconstruction_pcts = []
    centrality_pcts = []

    for seq_len, seq_data in results['seq_lengths'].items():
        pcts = seq_data['percentages']
        forward_pcts.append(pcts.get('1_forward_pass', 0))
        reconstruction_pcts.append(pcts.get('2_reconstruction', 0))
        centrality_pcts.append(pcts.get('3_centrality', 0))

    results['summary'] = {
        'avg_forward_pct': sum(forward_pcts) / len(forward_pcts) if forward_pcts else 0,
        'avg_reconstruction_pct': sum(reconstruction_pcts) / len(reconstruction_pcts) if reconstruction_pcts else 0,
        'avg_centrality_pct': sum(centrality_pcts) / len(centrality_pcts) if centrality_pcts else 0,
        'overhead_pct': (
            sum(reconstruction_pcts) / len(reconstruction_pcts) +
            sum(centrality_pcts) / len(centrality_pcts)
        ) if reconstruction_pcts else 0
    }

    # Success check
    overhead = results['summary']['overhead_pct']
    success = overhead < 10.0

    results['success'] = success
    results['success_criteria'] = 'Reconstruction + Centrality < 10% of total'

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"Overhead: {overhead:.1f}% (threshold: 10%)")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp5_latency.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_latency_breakdown(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot stacked bar chart of latency breakdown.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    seq_lengths = list(results['seq_lengths'].keys())
    components = ['1_forward_pass', '2_reconstruction', '3_centrality', '4_gse_computation']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    labels = ['Forward Pass', 'Reconstruction', 'Centrality', 'GSE']

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(seq_lengths))
    width = 0.6

    bottoms = np.zeros(len(seq_lengths))

    for i, (component, color, label) in enumerate(zip(components, colors, labels)):
        values = [
            results['seq_lengths'][sl].get(component, {}).get('mean_ms', 0)
            for sl in seq_lengths
        ]
        ax.bar(x, values, width, bottom=bottoms, label=label, color=color)
        bottoms += values

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('AG-SAR Latency Decomposition by Component')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage annotations
    for i, sl in enumerate(seq_lengths):
        total = bottoms[i]
        overhead = (
            results['seq_lengths'][sl].get('2_reconstruction', {}).get('mean_ms', 0) +
            results['seq_lengths'][sl].get('3_centrality', {}).get('mean_ms', 0)
        )
        pct = (overhead / total * 100) if total > 0 else 0
        ax.annotate(f'{pct:.1f}% overhead', xy=(i, total), ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from ag_sar import AGSAR

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    # Initialize AG-SAR
    ag_sar = AGSAR(model, tokenizer)

    # Run experiment
    config = EvalConfig(
        seq_lengths=[128, 512, 1024],
        warmup_runs=10,
        benchmark_runs=100
    )

    results = run_latency_experiment(ag_sar, config)

    # Plot
    plot_latency_breakdown(results, config.results_dir / 'exp5_latency.png')
