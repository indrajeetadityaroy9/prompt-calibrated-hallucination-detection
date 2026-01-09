#!/usr/bin/env python3
"""
Verify v8 Residual Stream Contrast across multiple datasets.

Tests on:
- HaluEval QA: Knowledge-grounded question answering
- HaluEval Summarization: Document summarization
- RAGTruth QA: RAG-based question answering
- RAGTruth Summarization: RAG-based summarization

Compares v8 against v5 (previous best) and v4 (semantic shielding).
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

    # Map variant to config name
    config_map = {
        "qa": "qa_samples",
        "summarization": "summarization_samples",
        "dialogue": "dialogue_samples",
    }
    config_name = config_map[variant]

    dataset = load_dataset("pminervini/HaluEval", config_name, split="data")

    uncertainties = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Build prompt based on variant
        if variant == "qa":
            prompt = f"Knowledge: {sample['knowledge']}\n\nQuestion: {sample['question']}"
            response = sample['answer']
        elif variant == "summarization":
            prompt = f"Document: {sample['document']}"
            response = sample['summary']
        else:
            prompt = f"Dialogue: {sample['dialogue_history']}"
            response = sample['response']

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
    print("AG-SAR v8 Cross-Dataset Evaluation")
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
    versions = [
        ("v8", 8, "mean"),
        ("v5", 5, "mean"),
        ("v4", 4, "min"),
    ]

    # Datasets to test
    datasets_config = [
        ("halueval_qa", lambda agsar: test_halueval(agsar, "qa", 100)),
        ("halueval_summ", lambda agsar: test_halueval(agsar, "summarization", 100)),
        ("ragtruth_qa", lambda agsar: test_ragtruth(agsar, "QA", 100)),
        ("ragtruth_summ", lambda agsar: test_ragtruth(agsar, "summarization", 100)),
    ]

    num_samples = 100
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
                results[dataset_name] = 0.5

        all_results[name] = results
        agsar.cleanup()

    # Final comparison table
    print("\n" + "=" * 90)
    print("CROSS-DATASET COMPARISON")
    print("=" * 90)

    # Header
    header = f"{'Dataset':<20}"
    for name, _, _ in versions:
        header += f" {name:>12}"
    header += f" {'Best':>10} {'Delta':>10}"
    print(header)
    print("-" * 90)

    # Per-dataset results
    version_wins = {name: 0 for name, _, _ in versions}
    for dataset_name, _ in datasets_config:
        row = f"{dataset_name:<20}"
        scores = []
        for name, _, _ in versions:
            score = all_results[name].get(dataset_name, 0.5)
            scores.append((name, score))
            status = "PASS" if score > 0.55 else "FAIL"
            row += f" {score:>12.4f}"

        best_name, best_score = max(scores, key=lambda x: x[1])
        second_score = sorted([s for _, s in scores], reverse=True)[1]
        delta = best_score - second_score
        version_wins[best_name] += 1

        row += f" {best_name:>10} {delta:>+10.4f}"
        print(row)

    # Average row
    print("-" * 90)
    row = f"{'AVERAGE':<20}"
    avgs = []
    for name, _, _ in versions:
        avg = np.mean(list(all_results[name].values()))
        avgs.append((name, avg))
        row += f" {avg:>12.4f}"

    best_name, best_avg = max(avgs, key=lambda x: x[1])
    second_avg = sorted([a for _, a in avgs], reverse=True)[1]
    delta = best_avg - second_avg
    row += f" {best_name:>10} {delta:>+10.4f}"
    print(row)

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    for name, _, _ in versions:
        wins = version_wins[name]
        avg = np.mean(list(all_results[name].values()))
        passed = sum(1 for v in all_results[name].values() if v > 0.55)
        print(f"  {name}: Avg AUROC = {avg:.4f}, Wins = {wins}/4, Passed = {passed}/4")

    # Determine overall winner
    best_version = max(avgs, key=lambda x: x[1])[0]
    print(f"\n  OVERALL WINNER: {best_version}")

    # v8 specific analysis
    v8_avg = np.mean(list(all_results["v8"].values()))
    v5_avg = np.mean(list(all_results["v5"].values()))
    v4_avg = np.mean(list(all_results["v4"].values()))

    print(f"\n  v8 vs v5: {(v8_avg - v5_avg)*100:+.1f}pp")
    print(f"  v8 vs v4: {(v8_avg - v4_avg)*100:+.1f}pp")

    if v8_avg > max(v5_avg, v4_avg):
        print("\n  v8 Residual Stream Contrast achieves SOTA across datasets!")
    elif v8_avg > v5_avg:
        print("\n  v8 outperforms v5 but not v4 on average.")
    else:
        print("\n  v8 does not outperform previous versions on average.")


if __name__ == "__main__":
    main()
