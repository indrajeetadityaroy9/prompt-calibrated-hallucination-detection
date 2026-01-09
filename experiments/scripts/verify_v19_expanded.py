#!/usr/bin/env python3
"""
AG-SAR v19 Expanded Evaluation - Domain Generalization Tests

Tests v19 on additional datasets to validate cross-domain SOTA claims:
1. PubMedQA - Medical domain (high intrinsic entropy, tests semantic shielding)
2. TruthfulQA - Factuality (tests detection of common misconceptions)

Hypothesis:
- v19's V×D conjunction should protect high-entropy medical terms (low FP rate)
- v19's JSD should detect parametric hallucinations on TruthfulQA
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


def test_pubmedqa(agsar, num_samples=100):
    """
    Test on PubMedQA - Medical domain generalization.

    Challenge: Medical text has high intrinsic entropy (complex terminology).
    Risk: Simple varentropy methods might flag *correct* medical terms as hallucinations.
    Test: Does V×D conjunction correctly protect high-entropy valid tokens?

    Dataset structure:
    - question: The medical question
    - context: Research abstract (context)
    - long_answer: Full answer from paper
    - final_decision: yes/no/maybe (ground truth)
    """
    print(f"\n{'=' * 60}")
    print("Dataset: PubMedQA (Medical Domain Generalization)")
    print("=" * 60)
    print("Testing: Does V×D protect high-entropy medical terminology?")

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    uncertainties = []
    labels = []
    details_list = []

    # For PubMedQA, we test if the model can correctly answer based on context
    # "yes" answers that match context = correct (label 0)
    # We create synthetic hallucinations by using wrong answers

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Build prompt with context
        context = " ".join(sample['context']['contexts']) if isinstance(sample['context'], dict) else str(sample['context'])
        question = sample['question']
        answer = sample['final_decision']  # yes/no/maybe

        prompt = f"Based on the following medical research:\n{context[:1000]}\n\nQuestion: {question}\nAnswer:"

        # Create response - use the actual decision as "correct" response
        # For hallucination testing, we'd flip the answer
        is_hall = (i % 2 == 0)  # Alternate: every other sample is "hallucinated" (wrong answer)

        if is_hall:
            # Flip the answer to create synthetic hallucination
            wrong_answer = {"yes": "no", "no": "yes", "maybe": "no"}.get(answer, "no")
            response = f" {wrong_answer}. This is definitively the case based on the evidence."
        else:
            response = f" {answer}. This conclusion is supported by the research findings."

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            uncertainty = details.get('uncertainty', details.get('score', 0))
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
            details_list.append(details)

            if i < 3:
                r_dec = details.get('R_deception', 0)
                r_conf = details.get('R_confusion', 0)
                print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
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

    # Analysis: Are we over-flagging medical terminology?
    fact_details = [d for d, l in zip(details_list, labels) if l == 0]
    if fact_details and 'R_confusion' in fact_details[0]:
        avg_fact_confusion = np.mean([d['R_confusion'] for d in fact_details])
        print(f"  Avg R_confusion on FACTS: {avg_fact_confusion:.4f} (should be LOW if shielding works)")

    return auroc, {}


def test_truthfulqa(agsar, num_samples=100):
    """
    Test on TruthfulQA - Factuality and common misconceptions.

    Challenge: Models often confidently state common misconceptions.
    Test: Does JSD detect when the model overrides context to state false beliefs?

    Dataset structure:
    - question: Question that often elicits false answers
    - best_answer: The truthful answer
    - correct_answers: List of correct answers
    - incorrect_answers: List of common incorrect answers (hallucinations)
    """
    print(f"\n{'=' * 60}")
    print("Dataset: TruthfulQA (Factuality - Common Misconceptions)")
    print("=" * 60)
    print("Testing: Does JSD detect parametric hallucinations?")

    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    uncertainties = []
    labels = []
    details_list = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        question = sample['question']
        best_answer = sample['best_answer']
        incorrect_answers = sample['incorrect_answers']

        prompt = f"Question: {question}\nAnswer:"

        # Alternate between truthful and hallucinated responses
        is_hall = (i % 2 == 0)

        if is_hall and incorrect_answers:
            # Use a common incorrect answer (hallucination)
            response = f" {incorrect_answers[0]}"
        else:
            # Use the best (truthful) answer
            response = f" {best_answer}"

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            uncertainty = details.get('uncertainty', details.get('score', 0))
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)
            details_list.append(details)

            if i < 3:
                r_dec = details.get('R_deception', 0)
                r_conf = details.get('R_confusion', 0)
                print(f"  Sample {i}: hall={is_hall}, unc={uncertainty:.4f}, R_dec={r_dec:.4f}, R_conf={r_conf:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
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

    # Analysis: Which detector is firing?
    hall_details = [d for d, l in zip(details_list, labels) if l == 1]
    fact_details = [d for d, l in zip(details_list, labels) if l == 0]
    if hall_details and 'R_deception' in hall_details[0]:
        avg_hall_jsd = np.mean([d['R_deception'] for d in hall_details])
        avg_fact_jsd = np.mean([d['R_deception'] for d in fact_details])
        print(f"  Avg R_deception: Halls={avg_hall_jsd:.4f}, Facts={avg_fact_jsd:.4f}")
        print(f"  JSD Gap: {avg_hall_jsd - avg_fact_jsd:.4f} (positive = JSD detecting halls)")

    return auroc, {}


def main():
    print("=" * 70)
    print("AG-SAR v19 Expanded Evaluation - Domain Generalization")
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

    # Test v19 configuration
    config = AGSARConfig(version=19)
    agsar = AGSAR(model, tokenizer, config=config)

    print(f"\n{'#' * 70}")
    print(f"# VERSION: v19 (Hinge-Risk Architecture)")
    print(f"{'#' * 70}")

    results = {}

    # Test on PubMedQA
    auroc_pubmed, _ = test_pubmedqa(agsar, num_samples=100)
    results['pubmedqa'] = auroc_pubmed

    # Test on TruthfulQA
    auroc_truthful, _ = test_truthfulqa(agsar, num_samples=100)
    results['truthfulqa'] = auroc_truthful

    # Summary
    print(f"\n{'=' * 70}")
    print("EXPANDED EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  PubMedQA (Medical Domain):  AUROC = {results['pubmedqa']:.4f}")
    print(f"  TruthfulQA (Factuality):    AUROC = {results['truthfulqa']:.4f}")
    print(f"  Average:                    AUROC = {np.mean(list(results.values())):.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
  PubMedQA Tests: Does V×D protect high-entropy medical terminology?
  - High AUROC = Good discrimination, low false positive rate on valid medical terms
  - Low AUROC = May be over-flagging complex but valid medical language

  TruthfulQA Tests: Does JSD detect parametric hallucinations?
  - High AUROC = JSD catches when model states confident misconceptions
  - Low AUROC = Model's parametric hallucinations not detected

  Combined with core benchmarks:
  | Dataset       | v19 AUROC | Signal Used        |
  |---------------|-----------|-------------------|
  | HaluEval QA   | 0.84      | V×D (Confusion)   |
  | HaluEval Summ | 0.61      | V×D (Confusion)   |
  | RAGTruth QA   | 0.72      | JSD (Deception)   |
  | RAGTruth Summ | 0.56      | JSD (Deception)   |
  | PubMedQA      | {:.2f}      | Semantic Shield   |
  | TruthfulQA    | {:.2f}      | JSD (Parametric)  |
""".format(results['pubmedqa'], results['truthfulqa']))


if __name__ == "__main__":
    main()
