#!/usr/bin/env python3
"""
AG-SAR V×D-Only Validation - "Safe Core" Hypothesis Test

This script confirms that V×D (Confusion) alone provides robust detection
on faithfulness tasks where JSD is inverted.

Previous Results (v19):
| Dataset                  | Full v19 | JSD Alone | V×D Alone |
|--------------------------|----------|-----------|-----------|
| FaithEval Counterfactual | 0.53     | 0.43      | **0.71**  |
| FaithEval Unanswerable   | 0.14     | 0.14      | **0.88**  |
| RAGBench                 | 0.33     | 0.20      | **0.75**  |

Hypothesis: V×D is the "Safe Core" - it works correctly on ALL task types.
JSD is context-sensitive - it inverts on counterfactual/unanswerable tasks.

If confirmed, v21 will build upon V×D as the primary signal with gated JSD.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ag_sar import AGSAR, AGSARConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_vxd_only(agsar, prompt, response):
    """
    Compute V×D signal only, bypassing full v19 pipeline.
    Returns R_confusion (Varentropy × Dispersion) without JSD fusion.
    """
    details = agsar.compute_uncertainty(prompt, response, return_details=True)
    return details.get('R_confusion', 0), details


def test_faitheval_counterfactual_vxd(agsar, tokenizer, model, num_samples=100):
    """
    FaithEval Counterfactual with V×D-only scoring.

    Expected: V×D AUROC ~0.71 (matching previous isolated stream analysis)
    """
    print(f"\n{'=' * 70}")
    print("FaithEval Counterfactual: V×D-Only Test")
    print("=" * 70)
    print("Testing: Does V×D alone detect context overrides?")
    print("")

    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")

    vxd_scores = []
    full_scores = []
    jsd_scores = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        context = sample['context']
        question = sample['question']
        choices = sample['choices']
        correct_key = sample['answerKey']

        choice_text = "\n".join([f"{l}: {t}" for l, t in zip(choices['label'], choices['text'])])

        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Choices:
{choice_text}

Answer with the letter of the correct choice:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_letter = response.strip().upper()[:1]
        is_faithful = response_letter == correct_key
        label = 0 if is_faithful else 1

        try:
            agsar.calibrate_on_prompt(prompt)
            vxd_score, details = compute_vxd_only(agsar, prompt, response)

            vxd_scores.append(vxd_score)
            full_scores.append(details.get('uncertainty', 0))
            jsd_scores.append(details.get('R_deception', 0))
            labels.append(label)

            if i < 3:
                print(f"  Sample {i}: faithful={is_faithful}, correct={correct_key}, pred={response_letter}")
                print(f"    V×D={vxd_score:.4f}, JSD={details.get('R_deception', 0):.4f}, Full={details.get('uncertainty', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, 0.5, 0.5

    auroc_vxd = roc_auc_score(labels, vxd_scores)
    auroc_full = roc_auc_score(labels, full_scores)
    auroc_jsd = roc_auc_score(labels, jsd_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} overrides):")
    print(f"  V×D Only AUROC:  {auroc_vxd:.4f}  {'<-- Safe Core' if auroc_vxd > 0.6 else ''}")
    print(f"  JSD Only AUROC:  {auroc_jsd:.4f}  {'<-- INVERTED!' if auroc_jsd < 0.5 else ''}")
    print(f"  Full v19 AUROC:  {auroc_full:.4f}")

    return auroc_vxd, auroc_jsd, auroc_full


def test_faitheval_unanswerable_vxd(agsar, tokenizer, model, num_samples=100):
    """
    FaithEval Unanswerable with V×D-only scoring.

    Expected: V×D AUROC ~0.88 (matching previous isolated stream analysis)
    """
    print(f"\n{'=' * 70}")
    print("FaithEval Unanswerable: V×D-Only Test")
    print("=" * 70)
    print("Testing: Does V×D alone detect fabrication from unanswerable context?")
    print("")

    dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")

    vxd_scores = []
    full_scores = []
    jsd_scores = []
    labels = []

    abstention_keywords = ['unanswerable', 'unknown', 'not available', 'no answer',
                          'cannot be determined', 'not mentioned', 'not provided',
                          "don't know", "doesn't say", 'unclear', 'not stated']

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        context = sample['context']
        question = sample['question']

        prompt = f"""Based only on the following context, answer the question. If the answer cannot be determined from the context, say "The answer is not available in the context."

Context: {context}

Question: {question}

Answer:"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_lower = response.lower()
        is_abstention = any(kw in response_lower for kw in abstention_keywords)
        label = 0 if is_abstention else 1  # Fabrication = 1

        try:
            agsar.calibrate_on_prompt(prompt)
            vxd_score, details = compute_vxd_only(agsar, prompt, response)

            vxd_scores.append(vxd_score)
            full_scores.append(details.get('uncertainty', 0))
            jsd_scores.append(details.get('R_deception', 0))
            labels.append(label)

            if i < 3:
                print(f"  Sample {i}: abstained={is_abstention}")
                print(f"    Response: {response[:60]}...")
                print(f"    V×D={vxd_score:.4f}, JSD={details.get('R_deception', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} fabrications, {len(labels) - sum(labels)} abstentions)")
        if len(labels) > 0:
            print(f"  Mean V×D: {np.mean(vxd_scores):.4f}")
        return 0.5, 0.5, 0.5

    auroc_vxd = roc_auc_score(labels, vxd_scores)
    auroc_full = roc_auc_score(labels, full_scores)
    auroc_jsd = roc_auc_score(labels, jsd_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} fabrications):")
    print(f"  V×D Only AUROC:  {auroc_vxd:.4f}  {'<-- Safe Core' if auroc_vxd > 0.6 else ''}")
    print(f"  JSD Only AUROC:  {auroc_jsd:.4f}  {'<-- INVERTED!' if auroc_jsd < 0.5 else ''}")
    print(f"  Full v19 AUROC:  {auroc_full:.4f}")

    return auroc_vxd, auroc_jsd, auroc_full


def test_ragbench_vxd(agsar, num_samples=100):
    """
    RAGBench with V×D-only scoring.

    Expected: V×D AUROC ~0.75 (matching previous isolated stream analysis)
    """
    print(f"\n{'=' * 70}")
    print("RAGBench: V×D-Only Test")
    print("=" * 70)
    print("Testing: Does V×D alone detect non-adherence in RAG responses?")
    print("")

    dataset = load_dataset("rungalileo/ragbench", "covidqa", split="test")

    vxd_scores = []
    full_scores = []
    jsd_scores = []
    labels = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        docs = sample['documents']
        context = "\n\n".join(docs[:3])
        question = sample['question']
        response = sample['response']

        is_adherent = sample['adherence_score']
        label = 0 if is_adherent else 1

        prompt = f"""Based on the following documents, answer the question.

Documents:
{context[:2000]}

Question: {question}

Answer:"""

        try:
            agsar.calibrate_on_prompt(prompt)
            vxd_score, details = compute_vxd_only(agsar, prompt, " " + response)

            vxd_scores.append(vxd_score)
            full_scores.append(details.get('uncertainty', 0))
            jsd_scores.append(details.get('R_deception', 0))
            labels.append(label)

            if i < 3:
                print(f"  Sample {i}: adherent={is_adherent}")
                print(f"    V×D={vxd_score:.4f}, JSD={details.get('R_deception', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class found")
        return 0.5, 0.5, 0.5

    auroc_vxd = roc_auc_score(labels, vxd_scores)
    auroc_full = roc_auc_score(labels, full_scores)
    auroc_jsd = roc_auc_score(labels, jsd_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} non-adherent):")
    print(f"  V×D Only AUROC:  {auroc_vxd:.4f}  {'<-- Safe Core' if auroc_vxd > 0.6 else ''}")
    print(f"  JSD Only AUROC:  {auroc_jsd:.4f}  {'<-- INVERTED!' if auroc_jsd < 0.5 else ''}")
    print(f"  Full v19 AUROC:  {auroc_full:.4f}")

    return auroc_vxd, auroc_jsd, auroc_full


def main():
    print("=" * 70)
    print("AG-SAR V×D-Only Validation: 'Safe Core' Hypothesis Test")
    print("=" * 70)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    config = AGSARConfig(version=19)
    agsar = AGSAR(model, tokenizer, config=config)

    print(f"\n{'#' * 70}")
    print("# Testing V×D as 'Safe Core' Across Faithfulness Tasks")
    print(f"{'#' * 70}")

    results = {}

    # Test 1: FaithEval Counterfactual
    vxd_cf, jsd_cf, full_cf = test_faitheval_counterfactual_vxd(agsar, tokenizer, model, num_samples=50)
    results['faitheval_counterfactual'] = {'vxd': vxd_cf, 'jsd': jsd_cf, 'full': full_cf}

    # Test 2: FaithEval Unanswerable
    vxd_ua, jsd_ua, full_ua = test_faitheval_unanswerable_vxd(agsar, tokenizer, model, num_samples=50)
    results['faitheval_unanswerable'] = {'vxd': vxd_ua, 'jsd': jsd_ua, 'full': full_ua}

    # Test 3: RAGBench
    vxd_rb, jsd_rb, full_rb = test_ragbench_vxd(agsar, num_samples=50)
    results['ragbench'] = {'vxd': vxd_rb, 'jsd': jsd_rb, 'full': full_rb}

    # Summary
    print(f"\n{'=' * 70}")
    print("SAFE CORE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<30} {'V×D Only':<12} {'JSD Only':<12} {'Full v19':<12}")
    print("-" * 66)
    for name, r in results.items():
        vxd_flag = "SAFE" if r['vxd'] > 0.6 else ""
        jsd_flag = "INVERTED" if r['jsd'] < 0.5 else ""
        print(f"{name:<30} {r['vxd']:.4f} {vxd_flag:<4}  {r['jsd']:.4f} {jsd_flag:<8}  {r['full']:.4f}")

    # Averages
    avg_vxd = np.mean([r['vxd'] for r in results.values()])
    avg_jsd = np.mean([r['jsd'] for r in results.values()])
    avg_full = np.mean([r['full'] for r in results.values()])

    print("-" * 66)
    print(f"{'AVERAGE':<30} {avg_vxd:.4f}        {avg_jsd:.4f}          {avg_full:.4f}")

    print(f"\n{'=' * 70}")
    print("SAFE CORE HYPOTHESIS TEST")
    print("=" * 70)

    # Check if V×D is the safe core
    vxd_wins = sum(1 for r in results.values() if r['vxd'] > r['full'])
    jsd_inverted = sum(1 for r in results.values() if r['jsd'] < 0.5)

    if avg_vxd > avg_full and vxd_wins >= 2:
        print(f"""
  HYPOTHESIS CONFIRMED: V×D is the Safe Core

  Evidence:
  - V×D outperforms Full v19 on {vxd_wins}/3 faithfulness datasets
  - Average V×D AUROC ({avg_vxd:.4f}) > Average Full v19 ({avg_full:.4f})
  - JSD is inverted on {jsd_inverted}/3 datasets

  Recommendation for v21:
  - Use V×D as primary signal: Risk = R_confusion
  - Gate JSD based on prompt complexity: w = sigmoid(Prompt_Varentropy - τ)
  - Final: Risk = max(R_confusion, (1 - w) × R_deception)

  This prevents JSD from hurting performance on counterfactual/unanswerable tasks
  while preserving its value on standard RAG tasks (RAGTruth).
""")
    else:
        print(f"""
  HYPOTHESIS NOT CONFIRMED

  V×D Only Avg: {avg_vxd:.4f}
  Full v19 Avg: {avg_full:.4f}
  V×D wins: {vxd_wins}/3

  Further investigation needed.
""")


if __name__ == "__main__":
    main()
