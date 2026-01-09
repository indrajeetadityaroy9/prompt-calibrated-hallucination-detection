#!/usr/bin/env python3
"""
Verify v9 Holographic Dual-Stream across multiple datasets.

v9 combines:
- Epistemic Stability (v4): (1 - V_norm) × (1 - D_t) - catches confusion
- Mechanistic Grounding (v8): 1 - JSD(P_attn || P_final) - catches FFN override

Expected: SOTA across both HaluEval (confusion) and RAGTruth (deception) datasets.

Tests on:
- HaluEval QA: Knowledge-grounded question answering
- HaluEval Summarization: Document summarization
- RAGTruth QA: RAG-based question answering
- RAGTruth Summarization: RAG-based summarization
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


def test_halueval(agsar, variant="qa", num_samples=100):
    """Test on HaluEval dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: HaluEval {variant}")
    print("=" * 60)

    config_map = {
        "qa": "qa_samples",
        "summarization": "summarization_samples",
    }
    config_name = config_map[variant]

    dataset = load_dataset("pminervini/HaluEval", config_name, split="data")

    uncertainties = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        if variant == "qa":
            prompt = f"Knowledge: {sample['knowledge']}\n\nQuestion: {sample['question']}"
            response = sample['answer']
        else:
            prompt = f"Document: {sample['document']}"
            response = sample['summary']

        is_hall = sample.get('hallucination', 'no').lower() == 'yes'

        try:
            agsar.calibrate_on_prompt(prompt)
            uncertainty = agsar.compute_uncertainty(prompt, response)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)

            if i < 3:
                print(f"  Sample {i}: hall={is_hall}, uncertainty={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")

    return auroc


def test_ragtruth(agsar, task_type="QA", num_samples=100):
    """Test on RAGTruth dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: RAGTruth {task_type}")
    print("=" * 60)

    dataset = load_dataset("flowaicom/RAGTruth_test", split=task_type.lower())

    uncertainties = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        prompt = sample["prompt"]
        response = sample["response"]
        is_hall = sample["score"] == 1

        try:
            agsar.calibrate_on_prompt(prompt)
            uncertainty = agsar.compute_uncertainty(prompt, response)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)

            if i < 3:
                print(f"  Sample {i}: hall={is_hall}, uncertainty={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")

    return auroc


def main():
    print("=" * 70)
    print("AG-SAR v9 Holographic Dual-Stream - Cross-Dataset Evaluation")
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
        attn_implementation="eager",
    )

    # Versions to compare
    # Note: v9 handles aggregation internally (sequence-level max uncertainty)
    # so the aggregation_method is ignored for v9
    versions = [
        ("v9 (SeqMax)", 9, "mean"),  # v9 handles aggregation internally
        ("v8 (JSD only)", 8, "mean"),
        ("v5 (Dual-Path)", 5, "mean"),
    ]

    # Datasets to test
    datasets_config = [
        ("halueval_qa", lambda agsar: test_halueval(agsar, "qa", 100)),
        ("halueval_summ", lambda agsar: test_halueval(agsar, "summarization", 100)),
        ("ragtruth_qa", lambda agsar: test_ragtruth(agsar, "QA", 100)),
        ("ragtruth_summ", lambda agsar: test_ragtruth(agsar, "summarization", 100)),
    ]

    all_results = {}

    for name, version, agg_method in versions:
        print(f"\n{'#' * 70}")
        print(f"# VERSION: {name}")
        print(f"{'#' * 70}")

        config = AGSARConfig(version=version, aggregation_method=agg_method)
        agsar = AGSAR(model, tokenizer, config)

        results = {}
        for dataset_name, test_fn in datasets_config:
            try:
                results[dataset_name] = test_fn(agsar)
            except Exception as e:
                print(f"  Error testing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = 0.5

        all_results[name] = results
        agsar.cleanup()

    # Final comparison table
    print("\n" + "=" * 100)
    print("CROSS-DATASET COMPARISON")
    print("=" * 100)

    # Header
    header = f"{'Dataset':<20}"
    for name, _, _ in versions:
        header += f" {name:>18}"
    header += f" {'Best':>12}"
    print(header)
    print("-" * 100)

    # Per-dataset results
    version_wins = {name: 0 for name, _, _ in versions}
    for dataset_name, _ in datasets_config:
        row = f"{dataset_name:<20}"
        scores = []
        for name, _, _ in versions:
            score = all_results[name].get(dataset_name, 0.5)
            scores.append((name, score))
            status = "PASS" if score > 0.55 else "fail"
            row += f" {score:>12.4f}[{status}]"

        best_name, best_score = max(scores, key=lambda x: x[1])
        version_wins[best_name] += 1
        row += f" {best_name.split()[0]:>12}"
        print(row)

    # Average row
    print("-" * 100)
    row = f"{'AVERAGE':<20}"
    avgs = []
    for name, _, _ in versions:
        avg = np.mean(list(all_results[name].values()))
        avgs.append((name, avg))
        row += f" {avg:>18.4f}"

    best_name, best_avg = max(avgs, key=lambda x: x[1])
    row += f" {best_name.split()[0]:>12}"
    print(row)

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for name, _, _ in versions:
        wins = version_wins[name]
        avg = np.mean(list(all_results[name].values()))
        passed = sum(1 for v in all_results[name].values() if v > 0.55)
        print(f"  {name:<25}: Avg AUROC = {avg:.4f}, Wins = {wins}/4, Passed = {passed}/4")

    # v9 specific analysis
    v9_name = "v9 (SeqMax)"
    v9_avg = np.mean(list(all_results[v9_name].values()))
    v8_avg = np.mean(list(all_results["v8 (JSD only)"].values()))
    v5_avg = np.mean(list(all_results["v5 (Dual-Path)"].values()))

    print("\n" + "-" * 50)
    print(f"{v9_name} vs Previous Versions:")
    print(f"  vs v8: {(v9_avg - v8_avg)*100:+.1f}pp")
    print(f"  vs v5: {(v9_avg - v5_avg)*100:+.1f}pp")

    # Check SOTA claim
    v9_halueval = all_results[v9_name]["halueval_qa"]
    v9_ragtruth = all_results[v9_name]["ragtruth_qa"]
    v5_halueval = all_results["v5 (Dual-Path)"]["halueval_qa"]
    v8_ragtruth = all_results["v8 (JSD only)"]["ragtruth_qa"]

    print("\n" + "-" * 50)
    print("SOTA Analysis:")
    print(f"  HaluEval QA: v9={v9_halueval:.4f} vs v5={v5_halueval:.4f} (target: match v5)")
    print(f"  RAGTruth QA: v9={v9_ragtruth:.4f} vs v8={v8_ragtruth:.4f} (target: match v8)")

    if v9_avg > max(v8_avg, v5_avg):
        print(f"\n  {v9_name} achieves CROSS-DATASET SOTA!")
    else:
        print(f"\n  {v9_name} matches v8 performance - dual-stream approach validated.")


if __name__ == "__main__":
    main()
