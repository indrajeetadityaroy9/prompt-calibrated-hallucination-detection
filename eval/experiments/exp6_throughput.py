"""
Experiment 6: Throughput Benchmarking

Compares Tokens Per Second (TPS) across:
1. Vanilla GPT-2 (no uncertainty)
2. AG-SAR (internal graph)
3. Original SAR (GPT-2 + RoBERTa perturbation - O(N) passes)
4. Multi-GPU AG-SAR (linear scaling validation)

Expected: AG-SAR TPS ≈ Vanilla TPS >> Original SAR TPS
Multi-GPU scaling efficiency should be >80% for 2 GPUs.
"""

from typing import Dict, Optional
from pathlib import Path
import json
import torch

from ..config import EvalConfig
from ..profiling.throughput import ThroughputBenchmark, MultiGPUBenchmark, compute_speedup


def run_throughput_experiment(
    ag_sar,
    original_sar,
    pe_baseline,
    model,
    tokenizer,
    config: EvalConfig,
    num_samples: int = 100,
    save_results: bool = True
) -> Dict:
    """
    Run throughput benchmarking experiment.

    Args:
        ag_sar: AGSAR instance
        original_sar: Original SAR baseline
        pe_baseline: Predictive Entropy baseline
        model: Base language model
        tokenizer: Tokenizer
        config: Evaluation configuration
        num_samples: Number of samples for benchmarking
        save_results: Whether to save results

    Returns:
        Dict with throughput results
    """
    print("=" * 60)
    print("Experiment 6: Throughput Benchmarking")
    print("=" * 60)

    # Generate test samples
    prompts = [f"Question {i}: What is the meaning of " for i in range(num_samples)]
    responses = ["life, the universe, and everything." for _ in range(num_samples)]

    # Initialize benchmark
    benchmark = ThroughputBenchmark(
        model=model,
        tokenizer=tokenizer,
        warmup_samples=config.warmup_runs
    )

    # Run benchmarks
    print("\nRunning benchmarks...")
    throughput_results = benchmark.run_all(
        ag_sar=ag_sar,
        original_sar=original_sar,
        pe_baseline=pe_baseline,
        prompts=prompts,
        responses=responses
    )

    # Compute speedups
    speedups = compute_speedup(throughput_results)

    # Format results
    results = {
        'experiment': 'throughput_benchmark',
        'num_samples': num_samples,
        'methods': {}
    }

    for method, result in throughput_results.items():
        results['methods'][method] = {
            'tokens_per_second': result.tokens_per_second,
            'samples_per_second': result.samples_per_second,
            'total_time_seconds': result.total_time_seconds,
            'avg_tokens_per_sample': result.avg_tokens_per_sample,
            'speedup_vs_original_sar': speedups[method]
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"{'Method':<25} {'TPS':>10} {'Samples/s':>12} {'Speedup':>10}")
    print("-" * 60)

    for method, data in results['methods'].items():
        print(f"{method:<25} {data['tokens_per_second']:>10.1f} "
              f"{data['samples_per_second']:>12.2f} "
              f"{data['speedup_vs_original_sar']:>10.1f}x")

    # Success check - focus on speedup vs Original SAR (the key claim)
    ag_sar_tps = results['methods']['ag_sar']['tokens_per_second']
    vanilla_tps = results['methods']['vanilla']['tokens_per_second']
    original_sar_tps = results['methods'].get('original_sar', {}).get('tokens_per_second', 1)

    vanilla_ratio = ag_sar_tps / vanilla_tps if vanilla_tps > 0 else 0
    speedup_vs_sar = ag_sar_tps / original_sar_tps if original_sar_tps > 0 else 0

    # Key claim: AG-SAR should be significantly faster than Original SAR (>5x)
    # and have reasonable overhead vs vanilla (<60% overhead, i.e., ratio > 0.4)
    success = speedup_vs_sar > 5.0 and vanilla_ratio > 0.4
    results['success'] = success
    results['success_criteria'] = 'AG-SAR > 5x Original SAR AND AG-SAR/Vanilla > 0.4'
    results['ag_sar_vanilla_ratio'] = vanilla_ratio
    results['speedup_vs_original_sar'] = speedup_vs_sar

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"AG-SAR/Vanilla ratio: {vanilla_ratio:.2f} (threshold: 0.4)")
    print(f"Speedup vs Original SAR: {speedup_vs_sar:.1f}x (threshold: 5x)")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp6_throughput.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def run_multi_gpu_scaling_experiment(
    single_ag_sar,
    multi_ag_sar,
    tokenizer,
    config: EvalConfig,
    num_samples: int = 100,
    save_results: bool = True
) -> Dict:
    """
    Benchmark multi-GPU scaling efficiency.

    Tests how well throughput scales from 1 GPU to N GPUs.
    Target: >80% scaling efficiency for 2 GPUs.

    Args:
        single_ag_sar: Single-GPU AGSAR instance
        multi_ag_sar: MultiGPUAGSAR instance
        tokenizer: Tokenizer for token counting
        config: Evaluation configuration
        num_samples: Number of samples for benchmarking
        save_results: Whether to save results

    Returns:
        Dict with scaling results
    """
    print("=" * 60)
    print("Experiment 6b: Multi-GPU Scaling Benchmark")
    print("=" * 60)
    print(f"Number of GPUs: {multi_ag_sar.num_gpus}")

    # Generate test samples
    prompts = [f"Question {i}: What is the capital of country number " for i in range(num_samples)]
    responses = ["The capital is a beautiful city with rich history and culture." for _ in range(num_samples)]

    # Initialize benchmark
    benchmark = MultiGPUBenchmark(
        tokenizer=tokenizer,
        warmup_samples=config.warmup_runs
    )

    # Run scaling test
    print("\nRunning scaling benchmark...")
    scaling_results = benchmark.run_scaling_test(
        single_ag_sar=single_ag_sar,
        multi_ag_sar=multi_ag_sar,
        prompts=prompts,
        responses=responses
    )

    # Format results
    results = {
        'experiment': 'multi_gpu_scaling',
        'num_samples': num_samples,
        'num_gpus': scaling_results['num_gpus'],
        'single_gpu': {
            'tokens_per_second': scaling_results['single_gpu'].tokens_per_second,
            'samples_per_second': scaling_results['single_gpu'].samples_per_second,
            'total_time_seconds': scaling_results['single_gpu'].total_time_seconds,
        },
        'multi_gpu': {
            'tokens_per_second': scaling_results['multi_gpu'].tokens_per_second,
            'samples_per_second': scaling_results['multi_gpu'].samples_per_second,
            'total_time_seconds': scaling_results['multi_gpu'].total_time_seconds,
        },
        'ideal_speedup': scaling_results['ideal_speedup'],
        'actual_speedup': scaling_results['actual_speedup'],
        'scaling_efficiency': scaling_results['scaling_efficiency'],
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"{'Configuration':<25} {'TPS':>12} {'Samples/s':>12}")
    print("-" * 60)
    print(f"{'Single GPU':<25} {results['single_gpu']['tokens_per_second']:>12.1f} "
          f"{results['single_gpu']['samples_per_second']:>12.2f}")
    num_gpus = results['num_gpus']
    multi_gpu_label = f'Multi-GPU ({num_gpus}x)'
    print(f"{multi_gpu_label:<25} {results['multi_gpu']['tokens_per_second']:>12.1f} "
          f"{results['multi_gpu']['samples_per_second']:>12.2f}")
    print("-" * 60)
    print(f"{'Ideal Speedup':<25} {results['ideal_speedup']:>12.1f}x")
    print(f"{'Actual Speedup':<25} {results['actual_speedup']:>12.2f}x")
    print(f"{'Scaling Efficiency':<25} {results['scaling_efficiency']:>12.1%}")

    # Success criteria: >80% scaling efficiency
    success = results['scaling_efficiency'] > 0.80
    results['success'] = success
    results['success_criteria'] = 'Scaling efficiency > 80%'

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"Scaling efficiency: {results['scaling_efficiency']:.1%} (threshold: 80%)")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp6b_multi_gpu_scaling.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_throughput_comparison(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot bar chart comparing throughput across methods.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    methods = list(results['methods'].keys())
    tps = [results['methods'][m]['tokens_per_second'] for m in methods]
    speedups = [results['methods'][m]['speedup_vs_original_sar'] for m in methods]

    # Rename for display
    display_names = {
        'vanilla': 'Vanilla GPT-2',
        'ag_sar': 'AG-SAR',
        'predictive_entropy': 'Pred. Entropy',
        'original_sar': 'Original SAR'
    }
    labels = [display_names.get(m, m) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TPS plot
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
    bars = ax1.bar(labels, tps, color=colors)
    ax1.set_ylabel('Tokens Per Second')
    ax1.set_title('Throughput Comparison')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, tps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.0f}', ha='center', va='bottom')

    # Speedup plot
    bars = ax2.bar(labels, speedups, color=colors)
    ax2.set_ylabel('Speedup vs Original SAR')
    ax2.set_title('Speedup Factor')
    ax2.axhline(y=1, color='red', linestyle='--', label='Original SAR baseline')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Add value labels
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}x', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ag_sar import AGSAR
    from eval.baselines import PredictiveEntropy, OriginalSAR

    # Load model
    print("Loading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained('gpt2')
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

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
    results = run_throughput_experiment(
        ag_sar=ag_sar,
        original_sar=original_sar,
        pe_baseline=pe_baseline,
        model=model,
        tokenizer=tokenizer,
        config=config,
        num_samples=50  # Reduced for Original SAR speed
    )

    # Plot
    plot_throughput_comparison(results, config.results_dir / 'exp6_throughput.png')
