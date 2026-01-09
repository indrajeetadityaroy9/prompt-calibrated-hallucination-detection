#!/usr/bin/env python3
"""
Verify v13.1 Adaptive Regime Switching across multiple datasets.

v13.1 uses sequence length to select the optimal detector:
- Short sequences (QA): Stability = (1-V) × (1-D) - catches confusion
- Long sequences (RAG): Grounding = 1 - JSD - catches FFN override

With ADAPTIVE aggregation:
- S_seq = SlidingWindowMin(Stability_t)   [burst detection]
- G_seq = SlidingWindowMin(Grounding_t)   [systemic detection]
- w = sigmoid((Length - 30) × 0.2)        [regime selector]
- Score_seq = (1-w) × S_seq + w × G_seq   [adaptive fusion]

Key Innovation: Use sequence length as automatic regime selector
- HaluEval (short ~10 tokens): w ≈ 0.0 → Stability dominates
- RAGTruth (long ~100 tokens): w ≈ 1.0 → Grounding dominates

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
                max_risk = details.get('max_risk', None)
                mean_risk = details.get('mean_risk', None)
                r_dec = details.get('R_deception', None)
                r_conf = details.get('R_confusion', None)
                v_mean = details.get('V_mean', None)
                gate = details.get('gate', None)
                mech = details.get('mech_integrity', None)
                epist = details.get('epist_coherence', None)
                s_v5 = details.get('S_v5', details.get('S_seq', None))
                g_v8 = details.get('G_v8', details.get('G_seq', None))
                w = details.get('regime_weight', None)
                if max_risk is not None and mean_risk is not None:
                    # v19 output (old)
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, max_risk={max_risk:.4f}, mean_risk={mean_risk:.4f}")
                elif r_dec is not None and r_conf is not None and v_mean is None:
                    # v19 output (new - topology-aware)
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")
                elif v_mean is not None and gate is not None:
                    # v17 output
                    print(f"  Sample {i}: hall={is_hall}, risk={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}, V_mean={v_mean:.2f}, gate={gate:.2f}")
                elif r_dec is not None and r_conf is not None:
                    # v16 output
                    print(f"  Sample {i}: hall={is_hall}, risk={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")
                elif mech is not None and epist is not None:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, mech={mech:.4f}, epist={epist:.4f}")
                elif s_v5 is not None and g_v8 is not None:
                    w_str = f"{w:.2f}" if w is not None else "N/A"
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, S_v5={s_v5:.4f}, G_v8={g_v8:.4f}, w={w_str}")
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

    if fact_details and ('S_v5' in fact_details[0] or 'S_seq' in fact_details[0]):
        key_s = 'S_v5' if 'S_v5' in fact_details[0] else 'S_seq'
        key_g = 'G_v8' if 'G_v8' in fact_details[0] else 'G_seq'
        fact_s = np.mean([d[key_s] for d in fact_details])
        fact_g = np.mean([d.get(key_g, 0) for d in fact_details])
        hall_s = np.mean([d[key_s] for d in hall_details]) if hall_details else 0
        hall_g = np.mean([d.get(key_g, 0) for d in hall_details]) if hall_details else 0
        avg_regime_weight = np.mean([d.get('regime_weight', 0) for d in details_list])
    else:
        fact_s = fact_g = hall_s = hall_g = avg_regime_weight = 0

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")
    print(f"  --- Detector Analysis ---")
    print(f"  Facts S(v5): {fact_s:.3f}, G(v8): {fact_g:.3f}")
    print(f"  Halls S(v5): {hall_s:.3f}, G(v8): {hall_g:.3f}")
    print(f"  Avg regime_weight: {avg_regime_weight:.3f}")

    return auroc, {'fact_s': fact_s, 'fact_g': fact_g,
                   'hall_s': hall_s, 'hall_g': hall_g,
                   'avg_regime_weight': avg_regime_weight}


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
                max_risk = details.get('max_risk', None)
                mean_risk = details.get('mean_risk', None)
                r_dec = details.get('R_deception', None)
                r_conf = details.get('R_confusion', None)
                v_mean = details.get('V_mean', None)
                gate = details.get('gate', None)
                mech = details.get('mech_integrity', None)
                epist = details.get('epist_coherence', None)
                s_v5 = details.get('S_v5', details.get('S_seq', None))
                g_v8 = details.get('G_v8', details.get('G_seq', None))
                w = details.get('regime_weight', None)
                if max_risk is not None and mean_risk is not None:
                    # v19 output (old)
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, max_risk={max_risk:.4f}, mean_risk={mean_risk:.4f}")
                elif r_dec is not None and r_conf is not None and v_mean is None:
                    # v19 output (new - topology-aware)
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")
                elif v_mean is not None and gate is not None:
                    # v17 output
                    print(f"  Sample {i}: hall={is_hall}, risk={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}, V_mean={v_mean:.2f}, gate={gate:.2f}")
                elif r_dec is not None and r_conf is not None:
                    # v16 output
                    print(f"  Sample {i}: hall={is_hall}, risk={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")
                elif mech is not None and epist is not None:
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, mech={mech:.4f}, epist={epist:.4f}")
                elif s_v5 is not None and g_v8 is not None:
                    w_str = f"{w:.2f}" if w is not None else "N/A"
                    print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, S_v5={s_v5:.4f}, G_v8={g_v8:.4f}, w={w_str}")
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

    if fact_details and ('S_v5' in fact_details[0] or 'S_seq' in fact_details[0]):
        key_s = 'S_v5' if 'S_v5' in fact_details[0] else 'S_seq'
        key_g = 'G_v8' if 'G_v8' in fact_details[0] else 'G_seq'
        fact_s = np.mean([d[key_s] for d in fact_details])
        fact_g = np.mean([d.get(key_g, 0) for d in fact_details])
        hall_s = np.mean([d[key_s] for d in hall_details]) if hall_details else 0
        hall_g = np.mean([d.get(key_g, 0) for d in hall_details]) if hall_details else 0
        avg_regime_weight = np.mean([d.get('regime_weight', 0) for d in details_list])
    else:
        fact_s = fact_g = hall_s = hall_g = avg_regime_weight = 0

    print(f"\n  Samples: {len(uncertainties)} ({sum(labels)} halls, {len(labels) - sum(labels)} facts)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Gap: {np.mean(halls) - np.mean(facts):.4f}")
    print(f"  Facts: {np.mean(facts):.3f} +/- {np.std(facts):.3f}")
    print(f"  Halls: {np.mean(halls):.3f} +/- {np.std(halls):.3f}")
    print(f"  --- Detector Analysis ---")
    print(f"  Facts S(v5): {fact_s:.3f}, G(v8): {fact_g:.3f}")
    print(f"  Halls S(v5): {hall_s:.3f}, G(v8): {hall_g:.3f}")
    print(f"  Avg regime_weight: {avg_regime_weight:.3f}")

    return auroc, {'fact_s': fact_s, 'fact_g': fact_g,
                   'hall_s': hall_s, 'hall_g': hall_g,
                   'avg_regime_weight': avg_regime_weight}


def main():
    print("=" * 70)
    print("AG-SAR v17 Thermodynamic Gating - Cross-Dataset Evaluation")
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
        ("v19 (HingeRisk)", 19, "mean"),     # v19: Hinge-Risk Architecture (SOTA)
        ("v17 (ThermoGate)", 17, "mean"),    # v17: Thermodynamic Gating
        ("v8 (JSD only)", 8, "mean"),
        ("v5 (Dual-Path)", 5, "mean"),
    ]

    # Datasets to test (QA + Summarization for comprehensive evaluation)
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

    n_datasets = len(datasets_config)
    for name, _, _ in versions:
        wins = version_wins[name]
        aurocs = [all_results[name].get(ds, {}).get('auroc', 0.5) for ds, _ in datasets_config]
        avg = np.mean(aurocs)
        passed = sum(1 for v in aurocs if v > 0.55)
        print(f"  {name:<25}: Avg AUROC = {avg:.4f}, Wins = {wins}/{n_datasets}, Passed = {passed}/{n_datasets}")

    # v17 analysis
    v17_name = "v17 (ThermoGate)"
    v16_name = "v16 (GroundRisk)"
    print("\n" + "-" * 50)
    print("v17 SOTA Analysis (Thermodynamic Gating):")

    # Get v17 results
    v17_halueval_qa = all_results[v17_name].get("halueval_qa", {}).get('auroc', 0.5)
    v17_halueval_summ = all_results[v17_name].get("halueval_summ", {}).get('auroc', 0.5)
    v17_ragtruth_qa = all_results[v17_name].get("ragtruth_qa", {}).get('auroc', 0.5)
    v17_ragtruth_summ = all_results[v17_name].get("ragtruth_summ", {}).get('auroc', 0.5)

    # Get specialist results for comparison
    v5_halueval_qa = all_results["v5 (Dual-Path)"].get("halueval_qa", {}).get('auroc', 0.5)
    v8_ragtruth_qa = all_results["v8 (JSD only)"].get("ragtruth_qa", {}).get('auroc', 0.5)

    print(f"  HaluEval QA:   v17={v17_halueval_qa:.4f} vs v5={v5_halueval_qa:.4f}")
    print(f"  HaluEval Summ: v17={v17_halueval_summ:.4f}")
    print(f"  RAGTruth QA:   v17={v17_ragtruth_qa:.4f} vs v8={v8_ragtruth_qa:.4f}")
    print(f"  RAGTruth Summ: v17={v17_ragtruth_summ:.4f}")

    # Print v17 design rationale
    print("\n" + "=" * 100)
    print("V17 DESIGN RATIONALE - THERMODYNAMIC GATING (MAXWELL'S DEMON)")
    print("=" * 100)
    print("""
    v17 uses ENTROPY to sort signals into the correct buckets:

    KEY INSIGHT - v16's MAX fusion failed because:
    - The "Noise Floor" of the idle detector exceeds the "Signal" of the active one
    - HaluEval: R_deception (JSD) is noise (~0.28), R_confusion is signal
    - RAGTruth: R_confusion is noise, R_deception is signal
    - MAX selects noise when it's larger → FAILED

    V17 SOLUTION - Thermodynamic Gating (State-of-Matter Heuristic):
    Use sequence-level varentropy to determine which detector to trust.

    1. Regime Detection:
       - V_mean > 1.75 (High Energy): Confusion Regime (HaluEval)
       - V_mean < 1.75 (Low Energy): Deception Regime (RAGTruth)

    2. Adaptive Fusion:
       gate = sigmoid((V_mean - 1.75) × 2.0)
       Risk = gate × R_confusion + (1-gate) × R_deception

    WHY THIS WORKS:
    | Dataset      | V_mean | Gate  | Trusted Signal | Expected |
    |--------------|--------|-------|----------------|----------|
    | HaluEval QA  | ~3.0   | ~0.93 | R_confusion    | ~0.97    |
    | HaluEval Sum | ~2.5   | ~0.82 | R_confusion    | ~0.58    |
    | RAGTruth QA  | ~1.0   | ~0.18 | R_deception    | ~0.79    |
    | RAGTruth Sum | ~1.2   | ~0.25 | R_deception    | ~0.54    |

    This is "Maxwell's Demon" - using entropy to sort particles (signals) into
    the correct chambers (regimes), achieving thermodynamic efficiency.
    """)


if __name__ == "__main__":
    main()
