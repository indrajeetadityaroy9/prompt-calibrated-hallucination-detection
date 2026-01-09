#!/usr/bin/env python3
"""
Verify AG-SAR data-agnostic behavior across multiple datasets.

Tests multiple versions:
- v4 (min): Semantic Shielding with min aggregation
- v5 (dual-path): Heterogeneous aggregation (Mean(A) × Min(T))

Datasets:
1. HaluEval QA - Factual question answering
2. HaluEval Summarization - Document-grounded summarization
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


def test_halueval_qa(agsar, num_samples=50):
    """Test on HaluEval QA dataset."""
    print("\n" + "=" * 60)
    print("Dataset: HaluEval QA")
    print("=" * 60)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    uncertainties = []
    labels = []
    thresholds_used = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        question = sample["question"]
        answer = sample["answer"]
        is_hall = sample["hallucination"] == "yes"

        prompt = f"Question: {question}\nAnswer:"

        try:
            # Calibrate and compute
            cal = agsar.calibrate_on_prompt(prompt)
            threshold = min(max(cal['dispersion_mu'] + cal['dispersion_sigma'], 0.05), 0.20)
            thresholds_used.append(threshold)

            uncertainty = agsar.compute_uncertainty(prompt, answer)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

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


def test_halueval_summarization(agsar, num_samples=50):
    """Test on HaluEval Summarization dataset."""
    print("\n" + "=" * 60)
    print("Dataset: HaluEval Summarization")
    print("=" * 60)

    dataset = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")

    uncertainties = []
    labels = []
    thresholds_used = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        # Use first 500 chars of document as prompt context
        doc = sample["document"][:500]
        summary = sample["summary"]
        is_hall = sample["hallucination"] == "yes"

        prompt = f"Document: {doc}...\n\nSummary:"

        try:
            cal = agsar.calibrate_on_prompt(prompt)
            threshold = min(max(cal['dispersion_mu'] + cal['dispersion_sigma'], 0.05), 0.20)
            thresholds_used.append(threshold)

            uncertainty = agsar.compute_uncertainty(prompt, summary)
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
    all_results = {}

    for name, version, agg_method in versions:
        print(f"\n{'#' * 70}")
        print(f"# VERSION: {name}")
        print(f"{'#' * 70}")

        # Initialize with specified version
        config = AGSARConfig(version=version, aggregation_method=agg_method)
        agsar = AGSAR(model, tokenizer, config)

        # Test on multiple datasets
        results = {}
        results["halueval_qa"] = test_halueval_qa(agsar, num_samples=50)
        results["halueval_summarization"] = test_halueval_summarization(agsar, num_samples=50)

        # Summary for this version
        print("\n" + "=" * 60)
        print(f"SUMMARY ({name})")
        print("=" * 60)
        for dataset_name, auroc in results.items():
            status = "✓" if auroc > 0.6 else "✗"
            print(f"  {dataset_name:<25}: AUROC = {auroc:.4f} {status}")

        avg_auroc = np.mean(list(results.values()))
        print(f"\n  Average AUROC: {avg_auroc:.4f}")
        all_results[name] = results

        # Cleanup
        agsar.cleanup()

    # Final comparison
    print("\n" + "=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print(f"{'Dataset':<25} {'v5 (Dual-Path)':>15} {'v4 (min)':>15} {'Winner':>15}")
    print("-" * 70)
    for dataset in ["halueval_qa", "halueval_summarization"]:
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
