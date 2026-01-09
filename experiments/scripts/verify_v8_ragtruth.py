#!/usr/bin/env python3
"""
Verify v8 Residual Stream Contrast on RAGTruth dataset.

RAGTruth tests grounding failures where the model generates text
that isn't supported by the provided context - exactly the case
where v8's FFN interference detection should shine.

v8 Hypothesis:
- RAGTruth hallucinations involve FFN overriding context signal (CHOKE)
- Previous versions miss cases where attention looks normal but FFN substitutes
- v8's JSD(P_attn || P_final) directly measures this override
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig


def test_ragtruth(agsar, task_type="QA", num_samples=50):
    """Test on RAGTruth dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: RAGTruth {task_type}")
    print("=" * 60)

    dataset = load_dataset("flowaicom/RAGTruth_test", split=task_type.lower())

    uncertainties = []
    labels = []
    details_list = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        prompt = sample["prompt"]
        response = sample["response"]
        is_hall = sample["score"] == 1  # 1 = hallucinated, 0 = faithful

        try:
            # Calibrate on prompt (for versions that need it)
            agsar.calibrate_on_prompt(prompt)

            # Get uncertainty with details
            result = agsar.compute_uncertainty(prompt, response, return_details=True)
            if isinstance(result, dict):
                uncertainty = result.get("uncertainty", result.get("score", 0.5))
                details_list.append(result)
            else:
                uncertainty = result
                details_list.append({})

            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)

            if i < 5:
                print(f"  Sample {i}: hall={is_hall}, uncertainty={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")

    return auroc, {"facts_mean": np.mean(facts), "halls_mean": np.mean(halls)}


def main():
    print("=" * 70)
    print("AG-SAR v8 Residual Stream Contrast - RAGTruth Evaluation")
    print("=" * 70)

    print("\nLoading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Required for attention hooks
    )

    # Test versions to compare
    versions = [
        ("v8 (Residual Stream)", 8, "mean"),
        ("v5 (Dual-Path)", 5, "mean"),
        ("v4 (Semantic Shield)", 4, "min"),
    ]

    # RAGTruth task types (QA is the main one)
    task_types = ["QA"]
    num_samples = 100  # Use 100 samples for more reliable AUROC

    all_results = {}

    for name, version, agg_method in versions:
        print(f"\n{'#' * 70}")
        print(f"# VERSION: {name}")
        print(f"{'#' * 70}")

        config = AGSARConfig(version=version, aggregation_method=agg_method)
        agsar = AGSAR(model, tokenizer, config)

        results = {}
        for task_type in task_types:
            auroc, stats = test_ragtruth(agsar, task_type, num_samples=num_samples)
            results[f"ragtruth_{task_type.lower()}"] = auroc

        # Summary
        print("\n" + "=" * 60)
        print(f"SUMMARY ({name})")
        print("=" * 60)
        for dataset_name, auroc in results.items():
            status = "PASS" if auroc > 0.55 else "FAIL"
            print(f"  {dataset_name:<25}: AUROC = {auroc:.4f} [{status}]")

        avg_auroc = np.mean(list(results.values()))
        print(f"\n  Average AUROC: {avg_auroc:.4f}")
        all_results[name] = results

        agsar.cleanup()

    # Final comparison
    print("\n" + "=" * 70)
    print("VERSION COMPARISON (RAGTruth)")
    print("=" * 70)
    header = f"{'Dataset':<25}"
    for name, _, _ in versions:
        header += f" {name:>18}"
    header += f" {'Best':>10}"
    print(header)
    print("-" * 70)

    for dataset in [f"ragtruth_{t.lower()}" for t in task_types]:
        row = f"{dataset:<25}"
        scores = []
        for name, _, _ in versions:
            score = all_results[name][dataset]
            scores.append((name, score))
            row += f" {score:>18.4f}"
        best = max(scores, key=lambda x: x[1])[0].split()[0]
        row += f" {best:>10}"
        print(row)

    # Average comparison
    print("-" * 70)
    row = f"{'Average':<25}"
    avgs = []
    for name, _, _ in versions:
        avg = np.mean(list(all_results[name].values()))
        avgs.append((name, avg))
        row += f" {avg:>18.4f}"
    best = max(avgs, key=lambda x: x[1])[0].split()[0]
    row += f" {best:>10}"
    print(row)

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    v8_auroc = list(all_results["v8 (Residual Stream)"].values())[0]
    v5_auroc = list(all_results["v5 (Dual-Path)"].values())[0]
    if v8_auroc > v5_auroc:
        print(f"v8 outperforms v5 by {(v8_auroc - v5_auroc)*100:.1f}pp on RAGTruth")
        print("This confirms the FFN interference hypothesis for CHOKE detection.")
    elif v8_auroc < v5_auroc:
        print(f"v5 outperforms v8 by {(v5_auroc - v8_auroc)*100:.1f}pp on RAGTruth")
        print("Authority flow may be more important than FFN interference for this task.")
    else:
        print("v8 and v5 perform equally on RAGTruth.")


if __name__ == "__main__":
    main()
