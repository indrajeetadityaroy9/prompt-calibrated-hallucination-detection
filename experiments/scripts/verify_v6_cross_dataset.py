#!/usr/bin/env python3
"""
Verify AG-SAR v6 Gaussian Complexity Matching across multiple datasets.

Tests:
- HaluEval QA (Confusion regime: V_hall >> V_prompt)
- RAGTruth QA (Simplification regime: V_hall << V_prompt)

v6 should handle BOTH regimes via bidirectional Gaussian penalty:
    G_complexity = exp(-(R_t - 1.0)² / 2σ²)
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
    """Test on HaluEval QA dataset (Confusion regime)."""
    print("\n" + "=" * 60)
    print("Dataset: HaluEval QA (Confusion Regime)")
    print("=" * 60)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    uncertainties = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        question = sample["question"]
        answer = sample["answer"]
        is_hall = sample["hallucination"] == "yes"

        prompt = f"Question: {question}\nAnswer:"

        try:
            agsar.calibrate_on_prompt(prompt)
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

    return auroc


def test_ragtruth_qa(agsar, num_samples=50):
    """Test on RAGTruth QA dataset (Simplification regime)."""
    print("\n" + "=" * 60)
    print("Dataset: RAGTruth QA (Simplification Regime)")
    print("=" * 60)

    dataset = load_dataset("flowaicom/RAGTruth_test", split="qa")

    uncertainties = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        prompt = sample["prompt"]
        response = sample["response"]
        is_hall = sample["score"] == 1  # 1 = hallucinated, 0 = faithful

        try:
            agsar.calibrate_on_prompt(prompt)
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

    # Test versions
    versions = [
        ("v6 (Gaussian)", 6, "mean"),
        ("v5 (Dual-Path)", 5, "mean"),
        ("v4 (Shielding)", 4, "min"),
    ]

    all_results = {}

    for name, version, agg_method in versions:
        print(f"\n{'#' * 70}")
        print(f"# VERSION: {name}")
        print(f"{'#' * 70}")

        config = AGSARConfig(version=version, aggregation_method=agg_method)
        agsar = AGSAR(model, tokenizer, config)

        results = {}
        results["halueval_qa"] = test_halueval_qa(agsar, num_samples=50)
        results["ragtruth_qa"] = test_ragtruth_qa(agsar, num_samples=50)

        # Summary for this version
        print("\n" + "=" * 60)
        print(f"SUMMARY ({name})")
        print("=" * 60)
        for dataset_name, auroc in results.items():
            status = "pass" if auroc > 0.55 else "FAIL"
            print(f"  {dataset_name:<20}: AUROC = {auroc:.4f} [{status}]")

        avg_auroc = np.mean(list(results.values()))
        print(f"\n  Average AUROC: {avg_auroc:.4f}")
        all_results[name] = results

        agsar.cleanup()

    # Final comparison
    print("\n" + "=" * 70)
    print("VERSION COMPARISON")
    print("=" * 70)
    print(f"{'Dataset':<20} {'v6 (Gaussian)':>15} {'v5 (Dual-Path)':>15} {'v4 (Shielding)':>15}")
    print("-" * 70)

    for dataset in ["halueval_qa", "ragtruth_qa"]:
        v6 = all_results["v6 (Gaussian)"][dataset]
        v5 = all_results["v5 (Dual-Path)"][dataset]
        v4 = all_results["v4 (Shielding)"][dataset]
        print(f"{dataset:<20} {v6:>15.4f} {v5:>15.4f} {v4:>15.4f}")

    print("-" * 70)
    v6_avg = np.mean(list(all_results["v6 (Gaussian)"].values()))
    v5_avg = np.mean(list(all_results["v5 (Dual-Path)"].values()))
    v4_avg = np.mean(list(all_results["v4 (Shielding)"].values()))
    print(f"{'Average':<20} {v6_avg:>15.4f} {v5_avg:>15.4f} {v4_avg:>15.4f}")

    # Winner determination
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    # HaluEval (Confusion regime) - v4/v5 should work
    print("\nHaluEval (Confusion Regime - V_hall >> V_prompt):")
    halueval_winner = max(
        [("v6", all_results["v6 (Gaussian)"]["halueval_qa"]),
         ("v5", all_results["v5 (Dual-Path)"]["halueval_qa"]),
         ("v4", all_results["v4 (Shielding)"]["halueval_qa"])],
        key=lambda x: x[1]
    )
    print(f"  Winner: {halueval_winner[0]} ({halueval_winner[1]:.4f})")

    # RAGTruth (Simplification regime) - v6 should excel
    print("\nRAGTruth (Simplification Regime - V_hall << V_prompt):")
    ragtruth_winner = max(
        [("v6", all_results["v6 (Gaussian)"]["ragtruth_qa"]),
         ("v5", all_results["v5 (Dual-Path)"]["ragtruth_qa"]),
         ("v4", all_results["v4 (Shielding)"]["ragtruth_qa"])],
        key=lambda x: x[1]
    )
    print(f"  Winner: {ragtruth_winner[0]} ({ragtruth_winner[1]:.4f})")

    # Overall
    overall_winner = max(
        [("v6", v6_avg), ("v5", v5_avg), ("v4", v4_avg)],
        key=lambda x: x[1]
    )
    print(f"\nOverall Winner: {overall_winner[0]} ({overall_winner[1]:.4f})")

    # Goldilocks test
    print("\n" + "=" * 70)
    print("GOLDILOCKS TEST")
    print("=" * 70)
    v6_halueval = all_results["v6 (Gaussian)"]["halueval_qa"]
    v6_ragtruth = all_results["v6 (Gaussian)"]["ragtruth_qa"]

    if v6_halueval > 0.60 and v6_ragtruth > 0.50:
        print("v6 PASSES the Goldilocks test!")
        print(f"  HaluEval (Confusion): {v6_halueval:.4f} > 0.60")
        print(f"  RAGTruth (Simplification): {v6_ragtruth:.4f} > 0.50")
    else:
        print("v6 FAILS the Goldilocks test")
        print(f"  HaluEval (Confusion): {v6_halueval:.4f} (need > 0.60)")
        print(f"  RAGTruth (Simplification): {v6_ragtruth:.4f} (need > 0.50)")


if __name__ == "__main__":
    main()
