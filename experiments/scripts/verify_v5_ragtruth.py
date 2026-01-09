#!/usr/bin/env python3
"""
Verify v5 Dual-Path Aggregation on RAGTruth dataset.

RAGTruth tests grounding failures where the model generates text
that isn't supported by the provided context - exactly the case
where v5's Mean(A) aggregation should shine.

v5 Hypothesis:
- RAGTruth hallucinations have LOW Authority (not grounded in context)
- v4's min(A×T) might miss this if T is high (consistent but ungrounded)
- v5's Mean(A) × Min(T) should catch ungrounded generation via A_mean
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
    thresholds_used = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        prompt = sample["prompt"]
        response = sample["response"]
        is_hall = sample["score"] == 1  # 1 = hallucinated, 0 = faithful

        try:
            # Calibrate on prompt
            cal = agsar.calibrate_on_prompt(prompt)
            threshold = min(max(cal['dispersion_mu'] + cal['dispersion_sigma'], 0.10), 0.40)
            thresholds_used.append(threshold)

            uncertainty = agsar.compute_uncertainty(prompt, response)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} ± {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} ± {np.std(halls):.3f}")
    print(f"  Dynamic τ: {np.mean(thresholds_used):.3f} ± {np.std(thresholds_used):.3f}")

    return auroc


def main():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Test both versions
    versions = [
        ("v5 (Dual-Path)", 5, "mean"),
        ("v4 (min)", 4, "min"),
    ]

    # RAGTruth task types
    task_types = ["QA", "summarization"]

    all_results = {}

    for name, version, agg_method in versions:
        print(f"\n{'#' * 70}")
        print(f"# VERSION: {name}")
        print(f"{'#' * 70}")

        config = AGSARConfig(version=version, aggregation_method=agg_method)
        agsar = AGSAR(model, tokenizer, config)

        results = {}
        for task_type in task_types:
            results[f"ragtruth_{task_type.lower()}"] = test_ragtruth(agsar, task_type, num_samples=50)

        # Summary
        print("\n" + "=" * 60)
        print(f"SUMMARY ({name})")
        print("=" * 60)
        for dataset_name, auroc in results.items():
            status = "✓" if auroc > 0.55 else "✗"
            print(f"  {dataset_name:<25}: AUROC = {auroc:.4f} {status}")

        avg_auroc = np.mean(list(results.values()))
        print(f"\n  Average AUROC: {avg_auroc:.4f}")
        all_results[name] = results

        agsar.cleanup()

    # Final comparison
    print("\n" + "=" * 70)
    print("VERSION COMPARISON (RAGTruth)")
    print("=" * 70)
    print(f"{'Dataset':<25} {'v5 (Dual-Path)':>15} {'v4 (min)':>15} {'Winner':>15}")
    print("-" * 70)
    for dataset in [f"ragtruth_{t.lower()}" for t in task_types]:
        v5 = all_results["v5 (Dual-Path)"][dataset]
        v4 = all_results["v4 (min)"][dataset]
        winner = "v5" if v5 > v4 else "v4" if v4 > v5 else "tie"
        print(f"{dataset:<25} {v5:>15.4f} {v4:>15.4f} {winner:>15}")

    v5_avg = np.mean(list(all_results["v5 (Dual-Path)"].values()))
    v4_avg = np.mean(list(all_results["v4 (min)"].values()))
    winner = "v5" if v5_avg > v4_avg else "v4"
    print("-" * 70)
    print(f"{'Average':<25} {v5_avg:>15.4f} {v4_avg:>15.4f} {winner:>15}")


if __name__ == "__main__":
    main()
