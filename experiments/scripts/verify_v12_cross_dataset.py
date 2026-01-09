#!/usr/bin/env python3
"""
Verify v12 Dual-Stream Risk across multiple datasets.

v12 combines:
- Internal Risk: tanh(V/τ) × Shield(D) - catches HaluEval confusion (burst)
- External Risk: JSD - catches RAGTruth deception (systemic)

With HETEROGENEOUS aggregation:
- R_internal = Percentile_90(risk_int)    [burst detection]
- R_external = Mean(risk_ext)             [systemic detection]
- Risk_seq = max(R_internal, R_external)  [MAX fusion]

Expected: Cross-dataset SOTA by running parallel specialist detectors.

Tests on:
- HaluEval QA: Knowledge-grounded question answering
- RAGTruth QA: RAG-based question answering
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
    details_list = []

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
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            # Handle different versions' return formats
            if 'uncertainty' in details:
                uncertainty = details['uncertainty']
            elif 'score' in details:
                uncertainty = details['score']
            else:
                uncertainty = agsar.compute_uncertainty(prompt, response)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
            details_list.append(details)

            if i < 3:
                r_int = details.get('R_internal', None)
                r_ext = details.get('R_external', None)
                if r_int is not None and r_ext is not None:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_int={r_int:.4f}, R_ext={r_ext:.4f}")
                else:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    # Analyze which detector is firing
    fact_details = [d for d, l in zip(details_list, labels) if l == 0]
    hall_details = [d for d, l in zip(details_list, labels) if l == 1]

    if fact_details and 'R_internal' in fact_details[0]:
        fact_r_int = np.mean([d['R_internal'] for d in fact_details])
        fact_r_ext = np.mean([d['R_external'] for d in fact_details])
        hall_r_int = np.mean([d['R_internal'] for d in hall_details]) if hall_details else 0
        hall_r_ext = np.mean([d['R_external'] for d in hall_details]) if hall_details else 0
    else:
        fact_r_int = fact_r_ext = hall_r_int = hall_r_ext = 0

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")
    print(f"  --- Detector Analysis ---")
    print(f"  Facts R_internal: {fact_r_int:.3f}, R_external: {fact_r_ext:.3f}")
    print(f"  Halls R_internal: {hall_r_int:.3f}, R_external: {hall_r_ext:.3f}")

    return auroc, {'fact_r_int': fact_r_int, 'fact_r_ext': fact_r_ext,
                   'hall_r_int': hall_r_int, 'hall_r_ext': hall_r_ext}


def test_ragtruth(agsar, task_type="QA", num_samples=100):
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
        is_hall = sample["score"] == 1

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            # Handle different versions' return formats
            if 'uncertainty' in details:
                uncertainty = details['uncertainty']
            elif 'score' in details:
                uncertainty = details['score']
            else:
                uncertainty = agsar.compute_uncertainty(prompt, response)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
            details_list.append(details)

            if i < 3:
                r_int = details.get('R_internal', None)
                r_ext = details.get('R_external', None)
                if r_int is not None and r_ext is not None:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_int={r_int:.4f}, R_ext={r_ext:.4f}")
                else:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    # Analyze which detector is firing
    fact_details = [d for d, l in zip(details_list, labels) if l == 0]
    hall_details = [d for d, l in zip(details_list, labels) if l == 1]

    if fact_details and 'R_internal' in fact_details[0]:
        fact_r_int = np.mean([d['R_internal'] for d in fact_details])
        fact_r_ext = np.mean([d['R_external'] for d in fact_details])
        hall_r_int = np.mean([d['R_internal'] for d in hall_details]) if hall_details else 0
        hall_r_ext = np.mean([d['R_external'] for d in hall_details]) if hall_details else 0
    else:
        fact_r_int = fact_r_ext = hall_r_int = hall_r_ext = 0

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")
    print(f"  --- Detector Analysis ---")
    print(f"  Facts R_internal: {fact_r_int:.3f}, R_external: {fact_r_ext:.3f}")
    print(f"  Halls R_internal: {hall_r_int:.3f}, R_external: {hall_r_ext:.3f}")

    return auroc, {'fact_r_int': fact_r_int, 'fact_r_ext': fact_r_ext,
                   'hall_r_int': hall_r_int, 'hall_r_ext': hall_r_ext}


def main():
    print("=" * 70)
    print("AG-SAR v12 Dual-Stream Risk - Cross-Dataset Evaluation")
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
        ("v12 (DualRisk)", 12, "mean"),  # v12 handles aggregation internally
        ("v8 (JSD only)", 8, "mean"),
        ("v5 (Dual-Path)", 5, "mean"),
    ]

    # Datasets to test (QA only for faster verification)
    datasets_config = [
        ("halueval_qa", lambda agsar: test_halueval(agsar, "qa", 100)),
        ("ragtruth_qa", lambda agsar: test_ragtruth(agsar, "QA", 100)),
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
                auroc, analysis = test_fn(agsar)
                results[dataset_name] = {'auroc': auroc, 'analysis': analysis}
            except Exception as e:
                print(f"  Error testing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                results[dataset_name] = {'auroc': 0.5, 'analysis': {}}

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
            score = all_results[name].get(dataset_name, {}).get('auroc', 0.5)
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
        aurocs = [all_results[name].get(ds, {}).get('auroc', 0.5) for ds, _ in datasets_config]
        avg = np.mean(aurocs)
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
        aurocs = [all_results[name].get(ds, {}).get('auroc', 0.5) for ds, _ in datasets_config]
        avg = np.mean(aurocs)
        passed = sum(1 for v in aurocs if v > 0.55)
        print(f"  {name:<25}: Avg AUROC = {avg:.4f}, Wins = {wins}/2, Passed = {passed}/2")

    # v12 analysis
    v12_name = "v12 (DualRisk)"
    v12_halueval = all_results[v12_name]["halueval_qa"]['auroc']
    v12_ragtruth = all_results[v12_name]["ragtruth_qa"]['auroc']
    v5_halueval = all_results["v5 (Dual-Path)"]["halueval_qa"]['auroc']
    v8_ragtruth = all_results["v8 (JSD only)"]["ragtruth_qa"]['auroc']

    print("\n" + "-" * 50)
    print("v12 SOTA Analysis:")
    print(f"  HaluEval QA: v12={v12_halueval:.4f} vs v5={v5_halueval:.4f} (target: match v5)")
    print(f"  RAGTruth QA: v12={v12_ragtruth:.4f} vs v8={v8_ragtruth:.4f} (target: match v8)")

    # Print v12 design rationale
    print("\n" + "=" * 100)
    print("V12 DESIGN RATIONALE")
    print("=" * 100)
    print("""
    v12 solves the "Multi-Objective Detection Problem":

    WHY PREVIOUS VERSIONS FAILED:
    - v10 (Multiplicative): S × G - when one signal is high, masks other's low
    - v11 (Additive): E_int + E_ext - different topologies need different aggregations

    V12 SOLUTION - Dual-Stream Risk with MAX Fusion:
    1. Internal Risk (Confusion Detector - HaluEval)
       - Signal: tanh(V/τ) × Shield(D)
       - Aggregation: Percentile_90 (catches burst confusion)

    2. External Risk (Deception Detector - RAGTruth)
       - Signal: JSD
       - Aggregation: Mean (captures systemic FFN override)

    3. MAX Fusion: Risk_seq = max(R_internal, R_external)
       - If either detector flags, we flag (OR logic)
       - Each detector uses its optimal aggregation
    """)


if __name__ == "__main__":
    main()
