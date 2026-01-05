#!/usr/bin/env python3
"""
Phase 3 Generative Evaluation: Universal Hallucination Controller.

Evaluates the RL-at-Depth controller across three pillars:
1. RAGTruth (Extrinsic) - Context adherence via JEPA
2. TruthfulQA (Intrinsic/General) - Common misconceptions via Truth Vector
3. WikiBio (Intrinsic/Specific) - Entity-specific facts via Truth Vector

Metrics:
- Group A (Quality): Accuracy, truthfulness
- Group B (Reliability): AUROC, AUPRC, ECE on trust scores
- Group C (Viability): Latency, Rejection Rate

Usage:
    python scripts/evaluate_generative.py --dataset ragtruth --samples 100
    python scripts/evaluate_generative.py --dataset truthfulqa --samples 100
    python scripts/evaluate_generative.py --dataset wikibio --samples 100
    python scripts/evaluate_generative.py --dataset all --samples 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.generation import AGSARGuidedGenerator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Metrics
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def load_dataset_samples(dataset_name: str, num_samples: int, seed: int = 42) -> List[Dict]:
    """Load samples from the specified dataset."""

    if dataset_name == "ragtruth":
        from experiments.data.ragtruth import RAGTruthDataset
        dataset = RAGTruthDataset(num_samples=num_samples, seed=seed, task_type="QA")
        dataset.load()

        samples = []
        for sample in dataset:
            # RAGTruth has context in the prompt
            samples.append({
                "id": sample.id,
                "prompt": sample.prompt,
                "reference": sample.response,  # The actual response
                "label": sample.label,  # 0=Faithful, 1=Hallucination
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
                "label": sample.label,  # 0=Fact (correct), 1=Misconception
                "dataset": "truthfulqa",
            })
        return samples

    elif dataset_name == "wikibio":
        from experiments.data.wikibio import load_wikibio_eval_set
        return load_wikibio_eval_set(limit=num_samples, seed=seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def calculate_metrics(results: List[Dict]) -> Dict[str, float]:
    """Calculate reliability metrics from results."""
    trust_scores = [r['trust_score'] for r in results if r['trust_score'] is not None]
    labels = [r['label'] for r in results if r['trust_score'] is not None]

    if not trust_scores or not SKLEARN_AVAILABLE:
        return {"auroc": 0.5, "auprc": 0.5}

    # For AUROC: High trust should correlate with label=0 (Safe/Faithful)
    # We want: Faithful (label=0) -> High Trust, Hallucination (label=1) -> Low Trust
    # So we use 1-label as the positive class (Safe=1, Unsafe=0)
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


def run_baseline_generation(model, tokenizer, prompt: str, max_tokens: int = 80) -> str:
    """Run baseline (greedy/standard) generation."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_dataset(
    generator: AGSARGuidedGenerator,
    model,
    tokenizer,
    engine: AGSAR,
    dataset_name: str,
    num_samples: int,
    output_dir: str,
    step_size: int = 10,
    num_candidates: int = 3,
    max_tokens: int = 80,
) -> Dict[str, Any]:
    """
    Evaluate on a single dataset.

    Returns dict with metrics and sample results.
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {dataset_name.upper()}")
    print(f"{'=' * 60}")

    # Load samples
    samples = load_dataset_samples(dataset_name, num_samples)
    print(f"Loaded {len(samples)} samples")

    results = []
    total_interventions = 0
    total_steps = 0
    total_latency_ms = 0

    for sample in tqdm(samples, desc=f"{dataset_name}"):
        prompt = sample["prompt"]

        # Run guided generation with stats
        t0 = time.perf_counter()
        try:
            generated_text, avg_trust, interventions, steps = generator.generate_with_stats(
                prompt,
                max_new_tokens=max_tokens,
                step_size=step_size,
                num_candidates=num_candidates,
                verbose=False,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            # Extract response (everything after prompt)
            response = generated_text[len(prompt):].strip()

            total_interventions += interventions
            total_steps += steps
            total_latency_ms += latency_ms

            results.append({
                "id": sample["id"],
                "dataset": dataset_name,
                "prompt": prompt[:200],  # Truncate for storage
                "generated": response,
                "reference": sample.get("reference", "")[:200],
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
    metrics["total_samples"] = len(samples)
    metrics["valid_samples"] = len(valid_results)

    # Print summary
    print(f"\n{dataset_name.upper()} Results:")
    print(f"  Reliability (AUROC):  {metrics['auroc']:.4f}")
    print(f"  Reliability (AUPRC):  {metrics['auprc']:.4f}")
    print(f"  Rejection Rate:       {metrics['rejection_rate']:.1%}")
    print(f"  Avg Latency:          {metrics['avg_latency_ms']:.1f} ms")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"phase3_{dataset_name}_{timestamp}.json")

    with open(out_path, 'w') as f:
        json.dump({
            "dataset": dataset_name,
            "metrics": metrics,
            "results": results,
        }, f, indent=2)

    print(f"  Saved to: {out_path}")

    return {"metrics": metrics, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Generative Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ragtruth", "truthfulqa", "wikibio", "all"],
        help="Dataset to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to evaluate"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per dataset"
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Tokens per evaluation step"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=3,
        help="Number of candidate paths"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=80,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase3_generative",
        help="Output directory"
    )
    parser.add_argument(
        "--truth-vector",
        type=str,
        default="data/truth_vectors/llama_3.1_8b_instruct.pt",
        help="Path to Truth Vector"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 3: Universal Hallucination Controller Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples per dataset: {args.samples}")
    print(f"Step size: {args.step_size}")
    print(f"Candidates: {args.candidates}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="eager",
    )

    # Initialize AG-SAR with Universal mode + Gate Sharpening (v12.3)
    config = AGSARConfig(
        enable_unified_gating=True,
        enable_semantic_dispersion=True,
        enable_intrinsic_detection=os.path.exists(args.truth_vector),
        truth_vector_path=args.truth_vector if os.path.exists(args.truth_vector) else None,
        # Gate Sharpening (v12.1) - proven robust for Truth Vector integration
        enable_gate_sharpening=True,
        gate_sharpen_low=0.2,
        gate_sharpen_high=0.8,
    )

    print("Initializing AG-SAR Engine...")
    engine = AGSAR(model, tokenizer, config)

    print(f"  Truth Vector: {'ENABLED' if config.enable_intrinsic_detection else 'DISABLED'}")
    print(f"  Gate Sharpening: ENABLED (low={config.gate_sharpen_low}, high={config.gate_sharpen_high})")

    print("Initializing Guided Generator...")
    generator = AGSARGuidedGenerator(model, tokenizer, engine)

    # Determine datasets to evaluate
    if args.dataset == "all":
        datasets = ["ragtruth", "truthfulqa", "wikibio"]
    else:
        datasets = [args.dataset]

    # Run evaluations
    all_results = {}
    for dataset_name in datasets:
        result = evaluate_dataset(
            generator=generator,
            model=model,
            tokenizer=tokenizer,
            engine=engine,
            dataset_name=dataset_name,
            num_samples=args.samples,
            output_dir=args.output_dir,
            step_size=args.step_size,
            num_candidates=args.candidates,
            max_tokens=args.max_tokens,
        )
        all_results[dataset_name] = result["metrics"]

    # Final Summary
    print("\n" + "=" * 60)
    print("PHASE 3 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'AUROC':<10} {'AUPRC':<10} {'Rejection':<12} {'Latency':<10}")
    print("-" * 60)

    for dataset_name, metrics in all_results.items():
        print(
            f"{dataset_name:<15} "
            f"{metrics['auroc']:.4f}    "
            f"{metrics['auprc']:.4f}    "
            f"{metrics['rejection_rate']:.1%}        "
            f"{metrics['avg_latency_ms']:.0f} ms"
        )

    print("=" * 60)

    # Interpretation
    print("\nInterpretation Guide:")
    print("  - AUROC > 0.75: System reliably knows when it's uncertain")
    print("  - Rejection Rate 15-40%: Healthy intervention level")
    print("  - Low Rejection on RAGTruth: JEPA agrees with model (good)")
    print("  - High Rejection on WikiBio: Controller steering away from lies")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
