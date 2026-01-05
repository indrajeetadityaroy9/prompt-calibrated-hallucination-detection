#!/usr/bin/env python3
"""
Verify v12.2 Adaptive Normalization (Model-Agnostic).

Tests the three adaptive mechanisms:
1. Adaptive Hysteresis: Threshold = α × (1 - baseline_score)
2. Batch-Wise Outlier Selection: Only switch if winner is >σ above batch mean
3. Gate Sharpening (optional): Force gate to extremes if outside [0.2, 0.8]

Usage:
    python scripts/verify_stability_fixes.py --dataset truthfulqa --samples 30
    python scripts/verify_stability_fixes.py --dataset ragtruth --samples 30
    python scripts/verify_stability_fixes.py --dataset all --samples 30
"""

import argparse
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.generation import AGSARGuidedGenerator

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_dataset_samples(dataset_name: str, num_samples: int, seed: int = 42):
    """Load samples from the specified dataset."""

    if dataset_name == "ragtruth":
        from experiments.data.ragtruth import RAGTruthDataset
        dataset = RAGTruthDataset(num_samples=num_samples, seed=seed, task_type="QA")
        dataset.load()

        samples = []
        for sample in dataset:
            samples.append({
                "id": sample.id,
                "prompt": sample.prompt,
                "reference": sample.response,
                "label": sample.label,
                "dataset": "ragtruth",
            })
        return samples

    elif dataset_name == "truthfulqa":
        from experiments.data.truthfulqa import TruthfulQADataset
        dataset = TruthfulQADataset(num_samples=num_samples, seed=seed)
        dataset.load()

        samples = []
        for sample in dataset:
            samples.append({
                "id": sample.id,
                "prompt": sample.prompt,
                "reference": sample.response,
                "label": sample.label,
                "dataset": "truthfulqa",
            })
        return samples

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def calculate_metrics(results):
    """Calculate reliability metrics from results."""
    trust_scores = [r['trust_score'] for r in results if r['trust_score'] is not None]
    labels = [r['label'] for r in results if r['trust_score'] is not None]

    if not trust_scores or not SKLEARN_AVAILABLE:
        return {"auroc": 0.5, "auprc": 0.5}

    safe_labels = [1 - label for label in labels]

    try:
        auroc = roc_auc_score(safe_labels, trust_scores)
    except Exception:
        auroc = 0.5

    try:
        auprc = average_precision_score(safe_labels, trust_scores)
    except Exception:
        auprc = sum(safe_labels) / len(safe_labels) if safe_labels else 0.5

    return {"auroc": auroc, "auprc": auprc}


def run_evaluation(
    generator,
    dataset_name: str,
    num_samples: int,
    step_size: int = 10,
    num_candidates: int = 3,
    max_tokens: int = 80,
    hysteresis_alpha: float = 0.10,
    outlier_sigma: float = 1.0,
):
    """Run evaluation on a dataset with stability fixes enabled."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    samples = load_dataset_samples(dataset_name, num_samples)
    print(f"Loaded {len(samples)} samples")

    results = []
    total_interventions = 0
    total_steps = 0
    total_latency_ms = 0

    for sample in tqdm(samples, desc=f"{dataset_name}"):
        prompt = sample["prompt"]

        t0 = time.perf_counter()
        try:
            generated_text, avg_trust, interventions, steps = generator.generate_with_stats(
                prompt,
                max_new_tokens=max_tokens,
                step_size=step_size,
                num_candidates=num_candidates,
                hysteresis_alpha=hysteresis_alpha,
                outlier_sigma=outlier_sigma,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            total_interventions += interventions
            total_steps += steps
            total_latency_ms += latency_ms

            results.append({
                "id": sample["id"],
                "dataset": dataset_name,
                "label": sample["label"],
                "trust_score": avg_trust,
                "interventions": interventions,
                "steps": steps,
                "latency_ms": latency_ms,
            })

        except Exception as e:
            print(f"  Error on sample {sample['id']}: {e}")
            results.append({
                "id": sample["id"],
                "dataset": dataset_name,
                "label": sample["label"],
                "trust_score": None,
                "error": str(e),
            })

    # Calculate metrics
    metrics = calculate_metrics(results)

    # Calculate rejection rate
    valid_results = [r for r in results if r.get("steps", 0) > 0]
    if valid_results:
        rejection_rate = total_interventions / total_steps if total_steps > 0 else 0
        avg_latency = total_latency_ms / len(valid_results)
    else:
        rejection_rate = 0
        avg_latency = 0

    metrics["rejection_rate"] = rejection_rate
    metrics["avg_latency_ms"] = avg_latency

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Verify v12.2 Adaptive Normalization")
    parser.add_argument("--dataset", type=str, default="truthfulqa",
                        choices=["ragtruth", "truthfulqa", "all"])
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--hysteresis-alpha", type=float, default=0.05)
    parser.add_argument("--outlier-sigma", type=float, default=0.0,
                        help="Set to 0 to disable outlier check (v12.3)")
    parser.add_argument("--gate-sharpening", action="store_true",
                        help="Enable hard gate sharpening (optional)")
    args = parser.parse_args()

    print("=" * 60)
    print("v12.3 RELAXED ADAPTIVE SELECTION VERIFICATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples per dataset: {args.samples}")
    print()
    print("v12.3 MECHANISMS:")
    print(f"  1. Adaptive Hysteresis: threshold = {args.hysteresis_alpha} × (1 - baseline)")
    if args.outlier_sigma > 0:
        print(f"  2. Outlier Selection: candidate must be >{args.outlier_sigma}σ above batch mean")
    else:
        print("  2. Outlier Selection: DISABLED (prevents deadlock for N=3)")
    if args.gate_sharpening:
        print("  3. Gate Sharpening: ENABLED (if gate < 0.2 -> 0.0, if gate > 0.8 -> 1.0)")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    )

    TRUTH_VECTOR_PATH = "data/truth_vectors/llama_3.1_8b_instruct.pt"

    # Initialize AG-SAR with optional gate sharpening
    config = AGSARConfig(
        enable_unified_gating=True,
        enable_semantic_dispersion=True,
        enable_intrinsic_detection=os.path.exists(TRUTH_VECTOR_PATH),
        truth_vector_path=TRUTH_VECTOR_PATH if os.path.exists(TRUTH_VECTOR_PATH) else None,
        # v12.1 Gate Sharpening (optional, for comparison)
        enable_gate_sharpening=args.gate_sharpening,
        gate_sharpen_low=0.2,
        gate_sharpen_high=0.8,
    )

    print("\nInitializing AG-SAR Engine...")
    print(f"  Truth Vector: {'ENABLED' if config.enable_intrinsic_detection else 'DISABLED'}")
    print(f"  Gate Sharpening: {'ENABLED' if config.enable_gate_sharpening else 'DISABLED (using adaptive)'}")

    engine = AGSAR(model, tokenizer, config)

    print("Initializing Guided Generator...")
    generator = AGSARGuidedGenerator(model, tokenizer, engine)

    # Determine datasets
    if args.dataset == "all":
        datasets = ["truthfulqa", "ragtruth"]
    else:
        datasets = [args.dataset]

    # Run evaluations
    all_results = {}
    for dataset_name in datasets:
        metrics = run_evaluation(
            generator=generator,
            dataset_name=dataset_name,
            num_samples=args.samples,
            step_size=args.step_size,
            num_candidates=args.candidates,
            hysteresis_alpha=args.hysteresis_alpha,
            outlier_sigma=args.outlier_sigma,
        )
        all_results[dataset_name] = metrics

        print(f"\n{dataset_name.upper()} Results:")
        print(f"  AUROC:          {metrics['auroc']:.4f}")
        print(f"  AUPRC:          {metrics['auprc']:.4f}")
        print(f"  Rejection Rate: {metrics['rejection_rate']:.1%}")
        print(f"  Avg Latency:    {metrics['avg_latency_ms']:.1f} ms")

    # Final Summary
    print("\n" + "=" * 60)
    print("v12.3 RELAXED ADAPTIVE SELECTION SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'AUROC':<10} {'Rejection':<12} {'Target AUROC':<14} {'Target Rej':<12}")
    print("-" * 60)

    targets = {
        "truthfulqa": {"auroc": 0.60, "rejection": 0.30},
        "ragtruth": {"auroc": 0.70, "rejection": 0.25},
    }

    for dataset_name, metrics in all_results.items():
        target = targets.get(dataset_name, {"auroc": 0.60, "rejection": 0.30})
        auroc_ok = "OK" if metrics['auroc'] >= target['auroc'] else "FAIL"
        rej_ok = "OK" if metrics['rejection_rate'] <= target['rejection'] else "HIGH"

        print(
            f"{dataset_name:<15} "
            f"{metrics['auroc']:.4f}    "
            f"{metrics['rejection_rate']:.1%}        "
            f">={target['auroc']:.2f} [{auroc_ok}]   "
            f"<={target['rejection']:.0%} [{rej_ok}]"
        )

    print("=" * 60)

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
