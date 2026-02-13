"""
TruthfulQA Evaluation Script with MC Scoring Protocol.

Evaluates JSD(max) on TruthfulQA using:
1. MC scoring: compute log-likelihood for each answer option
2. Identify model's chosen answer (highest LL)
3. Compute JSD(max) over tokens of the CHOSEN ANSWER
4. Report AUROC + risk-coverage metrics with bootstrap CIs

This is the primary deterministic-label benchmark for the paper.

Usage:
    python scripts/eval_truthfulqa.py --model meta-llama/Llama-3.1-8B-Instruct --num-examples 200
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar.evaluation.data.truthfulqa_loader import (
    TruthfulQALoader,
    TruthfulQAEvaluator,
    compute_truthfulqa_metrics,
)
from ag_sar.evaluation.metrics import (
    compute_auroc,
    bootstrap_auroc_ci,
    compute_risk_coverage,
    compute_aurc,
    compute_e_aurc,
)
from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig



def main():
    parser = argparse.ArgumentParser(description="Evaluate on TruthfulQA")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=200,
        help="Number of examples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--no-jsd",
        action="store_true",
        help="Skip JSD computation (only MC scoring)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific categories",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("TruthfulQA Evaluation")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Num examples: {args.num_examples}")
    print(f"Compute JSD: {not args.no_jsd}")

    # Load TruthfulQA dataset
    print("\nLoading TruthfulQA dataset...")
    loader = TruthfulQALoader(
        max_examples=args.num_examples,
        seed=args.seed,
        categories=args.categories,
    )
    examples = loader.load()
    stats = loader.get_statistics(examples)
    print(f"Loaded {stats['total_examples']} examples")
    print(f"Categories: {stats['categories']}")
    print(f"Avg MC1 options: {stats['avg_mc1_options']:.1f}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    # Initialize detector (only needed if computing JSD)
    detector = None
    if not args.no_jsd:
        print("Initializing hallucination detector...")
        config = DetectorConfig(
            layer_subset="last_third",
            candidate_topk=128,
            lci_topk=10,
        )
        detector = AGSAR(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

    # Initialize evaluator
    evaluator = TruthfulQAEvaluator(
        model=model,
        tokenizer=tokenizer,
        detector=detector,
        max_answer_tokens=64,
    )

    # Run evaluation
    print(f"\nEvaluating {len(examples)} examples...")
    results = evaluator.evaluate_batch(
        examples,
        compute_jsd_on_answer=not args.no_jsd,
        progress=True,
    )

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Basic accuracy
    valid_results = [r for r in results if "error" not in r]
    correct = sum(1 for r in valid_results if r.get("is_correct", False))
    accuracy = correct / len(valid_results) if valid_results else 0

    print(f"\nMC1 Accuracy: {accuracy:.3f} ({correct}/{len(valid_results)})")

    if not args.no_jsd:
        # Compute AUROC metrics
        metrics = compute_truthfulqa_metrics(results)

        print(f"\n--- AUROC (predicting incorrect from JSD) ---")
        print(f"JSD(max) AUROC: {metrics['jsd_auroc']:.3f} [{metrics['jsd_ci_low']:.3f}, {metrics['jsd_ci_high']:.3f}]")
        print(f"Entropy(max) AUROC: {metrics['entropy_auroc']:.3f} [{metrics['entropy_ci_low']:.3f}, {metrics['entropy_ci_high']:.3f}]")
        print(f"Length AUROC: {metrics['length_auroc']:.3f} [{metrics['length_ci_low']:.3f}, {metrics['length_ci_high']:.3f}]")

        # Delta vs length
        delta_jsd_length = metrics['jsd_auroc'] - metrics['length_auroc']
        print(f"\nΔ JSD vs Length: {delta_jsd_length:+.3f}")

        # Risk-coverage analysis
        jsd_scores = [r.get("jsd_max", 0) for r in valid_results if r.get("jsd_max") is not None]
        is_incorrect = [0 if r.get("is_correct", False) else 1 for r in valid_results if r.get("jsd_max") is not None]

        if jsd_scores and is_incorrect:
            print(f"\n--- Risk-Coverage Analysis ---")
            risk_coverage = compute_risk_coverage(jsd_scores, is_incorrect)

            for rc in risk_coverage:
                print(
                    f"Coverage {rc['coverage']:.0%}: "
                    f"Error rate {rc['error_rate']:.3f} "
                    f"(reduction: {rc['error_reduction_pct']:+.1f}%)"
                )

            # AURC
            aurc = compute_aurc(jsd_scores, is_incorrect)
            e_aurc = compute_e_aurc(jsd_scores, is_incorrect)
            print(f"\nAURC: {aurc:.4f}")
            print(f"E-AURC: {e_aurc:.4f}")

            metrics['risk_coverage'] = risk_coverage
            metrics['aurc'] = aurc
            metrics['e_aurc'] = e_aurc

        # Category breakdown
        if metrics.get('by_category'):
            print(f"\n--- By Category ---")
            for cat, data in sorted(metrics['by_category'].items()):
                jsd_str = f"JSD AUROC: {data.get('jsd_auroc', 'N/A'):.3f}" if 'jsd_auroc' in data else ""
                print(f"  {cat}: Acc={data['accuracy']:.3f} {jsd_str}")

    # Save results
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "num_examples": len(valid_results),
            "accuracy": accuracy,
            "results": valid_results,
        }
        if not args.no_jsd:
            output_data["metrics"] = metrics

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
