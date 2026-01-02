#!/usr/bin/env python3
"""
AG-SAR Evaluation Runner

Main script to run all experiments and generate results.

Usage:
    python scripts/run_eval.py --chapter 3     # Run profiling only
    python scripts/run_eval.py --chapter 1 2   # Run mechanistic + predictive
    python scripts/run_eval.py --all           # Run all experiments
    python scripts/run_eval.py --quick         # Quick run with reduced samples
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU/CUDA).
    This ensures deterministic behavior across runs.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def setup_device():
    """Setup compute device and dtype."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def load_model_and_tokenizer(model_name: str = 'gpt2', device=None):
    """Load the base language model.

    Supports:
        - 'gpt2': GPT-2 base model
        - 'llama3': Meta-Llama-3-8B-Instruct (requires HF token)
        - Any HuggingFace model ID
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    # Map shorthand names to full model IDs
    # NOTE: Use BASE models for uncertainty estimation, NOT Instruct versions
    # Instruct models are RLHF-tuned to be overconfident, destroying calibration
    model_map = {
        'gpt2': 'gpt2',
        # Llama-3 family (use BASE models for uncertainty - Instruct is RLHF-overconfident)
        'llama3': 'meta-llama/Meta-Llama-3-8B',  # Base model
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B',
        'llama3-instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'llama3.1': 'meta-llama/Llama-3.1-8B',
        'llama3.1-8b': 'meta-llama/Llama-3.1-8B',
        'llama3.2': 'meta-llama/Llama-3.2-3B',
        'llama3.2-3b': 'meta-llama/Llama-3.2-3B',
        # Non-gated alternatives
        'qwen2.5': 'Qwen/Qwen2.5-7B',
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B',
        'qwen': 'Qwen/Qwen2.5-7B',
        'mistral': 'mistralai/Mistral-7B-v0.3',
        'mistral-7b': 'mistralai/Mistral-7B-v0.3',
    }
    model_id = model_map.get(model_name, model_name)

    print(f"\nLoading {model_id}...")

    # Check for HF token for gated models
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    token_kwargs = {'token': hf_token} if hf_token else {}

    # Detect architecture type for RoPE-based models
    is_rope_model = any(arch in model_id.lower() for arch in ['llama', 'qwen', 'mistral'])

    # Load model with appropriate settings
    if is_rope_model:
        # RoPE-based models (Llama, Qwen, Mistral) - use Flash Attention
        arch_name = 'Llama' if 'llama' in model_id.lower() else \
                    'Qwen' if 'qwen' in model_id.lower() else 'Mistral'
        print(f"  Detected {arch_name} architecture (RoPE + GQA)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map='auto' if device is None else None,
            attn_implementation='sdpa',  # Use SDPA for Flash Attention
            trust_remote_code=True,  # Required for Qwen
            **token_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,  # Required for Qwen
            **token_kwargs
        )

        # RoPE models need pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if device and not hasattr(model, 'hf_device_map'):
            model = model.to(device)

        print("  Using bfloat16 precision")
    else:
        # GPT-2 and other models
        model = AutoModelForCausalLM.from_pretrained(model_id, **token_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **token_kwargs)

        if device:
            model = model.to(device)

        # Use bfloat16 if available for faster inference
        if device and device.type == 'cuda' and torch.cuda.is_bf16_supported():
            model = model.to(torch.bfloat16)
            print("  Using bfloat16 precision")

    model.eval()
    return model, tokenizer


def initialize_ag_sar(model, tokenizer, use_torch_compile: bool = True, head_weights_path: str = None):
    """Initialize AG-SAR."""
    from ag_sar import AGSAR
    from ag_sar.config import AGSARConfig

    print("Initializing AG-SAR...")
    config = AGSARConfig(
        use_torch_compile=use_torch_compile,
        use_head_weighting=head_weights_path is not None,
        head_weights_path=head_weights_path,
    )
    if head_weights_path:
        print(f"  Using Truth-Head weights from: {head_weights_path}")
    ag_sar = AGSAR(model, tokenizer, config=config)
    return ag_sar


def initialize_baselines(model, tokenizer):
    """Initialize baseline methods."""
    from eval.baselines import PredictiveEntropy, OriginalSAR

    print("Initializing Predictive Entropy baseline...")
    pe_baseline = PredictiveEntropy(model, tokenizer)

    print("Initializing Original SAR baseline (loading RoBERTa-Large)...")
    try:
        original_sar = OriginalSAR(model, tokenizer)
    except Exception as e:
        print(f"Warning: Failed to load Original SAR: {e}")
        original_sar = None

    return pe_baseline, original_sar


def run_chapter_3(ag_sar, original_sar, pe_baseline, model, tokenizer, config, quick=False):
    """Run Chapter 3: Computational Profiling experiments."""
    from eval.experiments.exp5_latency import run_latency_experiment, plot_latency_breakdown
    from eval.experiments.exp6_throughput import run_throughput_experiment, plot_throughput_comparison

    results = {}

    print("\n" + "=" * 70)
    print("CHAPTER 3: COMPUTATIONAL PROFILING")
    print("=" * 70)

    # Experiment 5: Latency Decomposition
    print("\n--- Experiment 5: Latency Decomposition ---")
    latency_results = run_latency_experiment(ag_sar, config)
    results['exp5_latency'] = latency_results
    plot_latency_breakdown(latency_results, config.results_dir / 'exp5_latency.png')

    # Experiment 6: Throughput Benchmarking
    print("\n--- Experiment 6: Throughput Benchmarking ---")
    num_samples = 20 if quick else 100
    throughput_results = run_throughput_experiment(
        ag_sar, original_sar, pe_baseline, model, tokenizer,
        config, num_samples=num_samples
    )
    results['exp6_throughput'] = throughput_results
    plot_throughput_comparison(throughput_results, config.results_dir / 'exp6_throughput.png')

    return results


def run_chapter_1(ag_sar, config, quick=False):
    """Run Chapter 1: Mechanistic Verification experiments."""
    from eval.experiments.exp1_sink_awareness import run_sink_awareness_experiment, plot_sink_awareness_comparison
    from eval.experiments.exp2_pos_correlation import run_pos_correlation_experiment, plot_correlation_distribution

    results = {}

    print("\n" + "=" * 70)
    print("CHAPTER 1: MECHANISTIC VERIFICATION")
    print("=" * 70)

    # Experiment 1: Sink-Awareness Test
    print("\n--- Experiment 1: Sink-Awareness Test ---")
    num_samples = 30 if quick else 100
    sink_results = run_sink_awareness_experiment(ag_sar, config, num_samples=num_samples)
    results['exp1_sink_awareness'] = sink_results
    plot_sink_awareness_comparison(sink_results, config.results_dir / 'exp1_sink_awareness.png')

    # Experiment 2: POS-Tag Correlation
    print("\n--- Experiment 2: POS-Tag Correlation ---")
    num_samples = 50 if quick else 200
    pos_results = run_pos_correlation_experiment(ag_sar, config, num_samples=num_samples)
    results['exp2_pos_correlation'] = pos_results
    plot_correlation_distribution(pos_results, config.results_dir / 'exp2_pos_correlation.png')

    return results


def run_chapter_2(ag_sar, original_sar, pe_baseline, config, quick=False, dataset='truthfulqa'):
    """Run Chapter 2: Hallucination Detection experiments."""
    from eval.experiments.exp3_auroc import run_auroc_experiment, plot_auroc_comparison, plot_auroc_bar_chart
    from eval.experiments.exp4_calibration import run_calibration_experiment, plot_calibration_comparison

    results = {}

    print("\n" + "=" * 70)
    print("CHAPTER 2: HALLUCINATION DETECTION")
    print(f"Dataset: {dataset}")
    print("=" * 70)

    # Experiment 3: AUROC Benchmark
    print("\n--- Experiment 3: AUROC Benchmark ---")
    num_samples = 50 if quick else 200
    auroc_results = run_auroc_experiment(
        ag_sar, original_sar, pe_baseline, config, num_samples=num_samples, dataset=dataset
    )
    results['exp3_auroc'] = auroc_results
    plot_auroc_comparison(auroc_results, config.results_dir / 'exp3_roc_curves.png')
    plot_auroc_bar_chart(auroc_results, config.results_dir / 'exp3_auroc_bars.png')

    # Experiment 4: Calibration Error
    print("\n--- Experiment 4: Calibration Error ---")
    num_samples = 80 if quick else 300
    calibration_results = run_calibration_experiment(
        ag_sar, pe_baseline, config, num_samples=num_samples, dataset=dataset
    )
    results['exp4_calibration'] = calibration_results
    plot_calibration_comparison(calibration_results, config.results_dir / 'exp4_reliability.png')

    return results


def run_chapter_5(ag_sar, config, quick=False):
    """Run Chapter 5: Ablation Studies."""
    from eval.experiments.exp7_ablation import run_ablation_experiment, plot_ablation_comparison, plot_ablation_heatmap

    results = {}

    print("\n" + "=" * 70)
    print("CHAPTER 5: ABLATION STUDIES")
    print("=" * 70)

    # Experiment 7: Component Ablation
    print("\n--- Experiment 7: Component Ablation ---")
    num_samples = 30 if quick else 100
    ablation_results = run_ablation_experiment(ag_sar, config, num_samples=num_samples)
    results['exp7_ablation'] = ablation_results
    plot_ablation_comparison(ablation_results, config.results_dir / 'exp7_ablation_bars.png')
    plot_ablation_heatmap(ablation_results, config.results_dir / 'exp7_ablation_heatmap.png')

    return results


def run_chapter_4(ag_sar, config, quick=False):
    """Run Chapter 4: Head Specialization Analysis."""
    from eval.experiments.exp8_head_specialization import (
        run_head_specialization_experiment,
        plot_head_specialization_heatmap
    )

    results = {}

    print("\n" + "=" * 70)
    print("CHAPTER 4: HEAD SPECIALIZATION ANALYSIS")
    print("=" * 70)

    # Experiment 8: Head Specialization
    print("\n--- Experiment 8: Head Specialization ---")
    num_samples = 30 if quick else 100
    head_results = run_head_specialization_experiment(ag_sar, config, num_samples=num_samples)
    results['exp8_head_specialization'] = head_results
    plot_head_specialization_heatmap(head_results, config.results_dir / 'exp8_head_heatmap.png')

    return results


def print_final_summary(all_results: dict):
    """Print final summary of all experiments."""
    from eval.visualizations.plots import create_results_table

    print("\n" + "=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)

    table = create_results_table(all_results)
    print(table)


def main():
    parser = argparse.ArgumentParser(description='AG-SAR Evaluation Runner')
    parser.add_argument(
        '--chapter', nargs='+', type=int,
        choices=[1, 2, 3, 4, 5],
        help='Chapters to run (1=Mechanistic, 2=Predictive, 3=Profiling, 4=Head Specialization, 5=Ablation)'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Run all experiments'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick run with reduced samples'
    )
    parser.add_argument(
        '--model', type=str, default='gpt2',
        help='Model name (default: gpt2)'
    )
    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Results directory (default: results/)'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int, default=[42],
        help='Random seeds for statistical significance (default: [42], use 42 101 999 for 3-seed)'
    )
    parser.add_argument(
        '--nli', action='store_true',
        help='Use NLI-based ground truth labeling (DeBERTa entailment)'
    )
    parser.add_argument(
        '--no-compile', action='store_true',
        help='Disable torch.compile (fixes variable sequence length issues)'
    )
    parser.add_argument(
        '--dataset', type=str, default='truthfulqa',
        choices=['truthfulqa', 'triviaqa', 'coqa'],
        help='Dataset for AUROC/calibration experiments (truthfulqa=adversarial, triviaqa=fact retrieval, coqa=conversational)'
    )
    parser.add_argument(
        '--head-weights', type=str, default=None,
        help='Path to calibrated head weights JSON (enables Truth-Head weighting)'
    )

    args = parser.parse_args()

    # Determine which chapters to run
    if args.all:
        chapters = [3, 1, 2, 4, 5]  # Profiling first, then mechanistic, predictive, head spec, ablation
    elif args.chapter:
        chapters = args.chapter
    else:
        print("Please specify --chapter or --all")
        parser.print_help()
        return 1

    print("=" * 70)
    print("AG-SAR EVALUATION FRAMEWORK")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {args.model}")
    print(f"Chapters: {chapters}")
    print(f"Quick mode: {args.quick}")
    print(f"Seeds: {args.seeds}")
    print(f"NLI labeling: {args.nli}")
    print(f"torch.compile: {not getattr(args, 'no_compile', False)}")
    print(f"Dataset: {args.dataset}")
    head_weights = getattr(args, 'head_weights', None)
    print(f"Head weights: {head_weights if head_weights else 'disabled'}")
    print("=" * 70)

    # Setup
    device = setup_device()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Initialize AG-SAR
    use_compile = not getattr(args, 'no_compile', False)
    ag_sar = initialize_ag_sar(
        model, tokenizer,
        use_torch_compile=use_compile,
        head_weights_path=head_weights
    )

    # Initialize baselines (only if needed)
    pe_baseline, original_sar = None, None
    if 2 in chapters or 3 in chapters:
        pe_baseline, original_sar = initialize_baselines(model, tokenizer)

    # Setup config
    from eval.config import EvalConfig
    config = EvalConfig(model_name=args.model)
    if args.results_dir:
        config.results_dir = Path(args.results_dir)
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Apply seed for reproducibility
    # Use first seed from list (multi-seed support for future statistical significance)
    seed = args.seeds[0]
    set_seed(seed)
    print(f"\nRandom seed set to: {seed}")

    # Run experiments
    all_results = {}

    if 3 in chapters:
        results = run_chapter_3(
            ag_sar, original_sar, pe_baseline, model, tokenizer, config, args.quick
        )
        all_results.update(results)

    if 1 in chapters:
        results = run_chapter_1(ag_sar, config, args.quick)
        all_results.update(results)

    if 2 in chapters:
        results = run_chapter_2(ag_sar, original_sar, pe_baseline, config, args.quick, args.dataset)
        all_results.update(results)

    if 4 in chapters:
        results = run_chapter_4(ag_sar, config, args.quick)
        all_results.update(results)

    if 5 in chapters:
        results = run_chapter_5(ag_sar, config, args.quick)
        all_results.update(results)

    # Save all results
    all_results_path = config.results_dir / 'all_results.json'
    with open(all_results_path, 'w') as f:
        # Filter out non-serializable data
        serializable = {}
        for k, v in all_results.items():
            if isinstance(v, dict):
                filtered = {
                    kk: vv for kk, vv in v.items()
                    if kk != 'raw_data'
                }
                serializable[k] = filtered
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    print(f"\nAll results saved to: {all_results_path}")

    # Create summary dashboard
    try:
        from eval.visualizations.plots import plot_experiment_summary
        plot_experiment_summary(all_results, save_path=config.results_dir / 'summary_dashboard.png')
        print(f"Summary dashboard saved to: {config.results_dir / 'summary_dashboard.png'}")
    except Exception as e:
        print(f"Warning: Could not create summary dashboard: {e}")

    # Print final summary
    print_final_summary(all_results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return exit code based on success
    total_experiments = len([k for k in all_results if k.startswith('exp')])
    passed = sum(1 for k, v in all_results.items() if k.startswith('exp') and v.get('success', False))

    if passed == total_experiments:
        print("\n✓ All experiments passed!")
        return 0
    else:
        print(f"\n✗ {total_experiments - passed}/{total_experiments} experiments failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
