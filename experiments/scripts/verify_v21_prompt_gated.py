#!/usr/bin/env python3
"""
AG-SAR v21 Prompt-Gated Fusion Evaluation

Tests v21's prompt complexity gating mechanism:
1. Faithfulness Tasks (FaithEval, RAGBench): v21 should match V×D performance (~0.78)
2. Core Benchmarks (HaluEval, RAGTruth): v21 should match v19 performance

Key Innovation:
- JSD is gated based on prompt varentropy (complexity)
- Complex prompts (counterfactual): JSD suppressed → V×D only
- Simple prompts (standard RAG): JSD active → full detection
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


def run_comparison(model, tokenizer, prompt, response, verbose=False):
    """Run both v19 and v21 on the same prompt/response, returning results."""
    # v19
    config_v19 = AGSARConfig(version=19)
    agsar_v19 = AGSAR(model, tokenizer, config=config_v19)
    agsar_v19.calibrate_on_prompt(prompt)
    d19 = agsar_v19.compute_uncertainty(prompt, response, return_details=True)
    agsar_v19.cleanup()

    # v21
    config_v21 = AGSARConfig(version=21)
    agsar_v21 = AGSAR(model, tokenizer, config=config_v21)
    agsar_v21.calibrate_on_prompt(prompt)
    d21 = agsar_v21.compute_uncertainty(prompt, response, return_details=True)
    agsar_v21.cleanup()

    return d19, d21


def test_faitheval_counterfactual(model, tokenizer, num_samples=50):
    """FaithEval Counterfactual: Context contradicts world knowledge."""
    print(f"\n{'=' * 70}")
    print("FaithEval Counterfactual: Prompt Gating Test")
    print("=" * 70)

    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

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
                **inputs, max_new_tokens=10, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_letter = response.strip().upper()[:1]
        is_faithful = response_letter == correct_key
        label = 0 if is_faithful else 1

        try:
            d19, d21 = run_comparison(model, tokenizer, prompt, response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i < 3:
                print(f"  Sample {i}: faithful={is_faithful}")
                print(f"    v19: unc={d19.get('uncertainty', 0):.4f}")
                print(f"    v21: unc={d21.get('uncertainty', 0):.4f}, w_prompt={d21.get('w_prompt', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, 0.5

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} overrides):")
    print(f"  v19 AUROC: {auroc_v19:.4f}")
    print(f"  v21 AUROC: {auroc_v21:.4f}  {'<-- IMPROVED!' if auroc_v21 > auroc_v19 + 0.05 else ''}")
    print(f"  Avg w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21


def test_faitheval_unanswerable(model, tokenizer, num_samples=50):
    """FaithEval Unanswerable: Context doesn't contain the answer."""
    print(f"\n{'=' * 70}")
    print("FaithEval Unanswerable: Prompt Gating Test")
    print("=" * 70)

    dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

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
                **inputs, max_new_tokens=50, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_lower = response.lower()
        is_abstention = any(kw in response_lower for kw in abstention_keywords)
        label = 0 if is_abstention else 1

        try:
            d19, d21 = run_comparison(model, tokenizer, prompt, response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i < 3:
                print(f"  Sample {i}: abstained={is_abstention}")
                print(f"    v19: unc={d19.get('uncertainty', 0):.4f}")
                print(f"    v21: unc={d21.get('uncertainty', 0):.4f}, w_prompt={d21.get('w_prompt', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} fabrications)")
        return 0.5, 0.5

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} fabrications):")
    print(f"  v19 AUROC: {auroc_v19:.4f}")
    print(f"  v21 AUROC: {auroc_v21:.4f}  {'<-- IMPROVED!' if auroc_v21 > auroc_v19 + 0.05 else ''}")
    print(f"  Avg w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21


def test_ragbench(model, tokenizer, num_samples=50):
    """RAGBench: Pre-labeled RAG adherence scores."""
    print(f"\n{'=' * 70}")
    print("RAGBench: Prompt Gating Test")
    print("=" * 70)

    dataset = load_dataset("rungalileo/ragbench", "covidqa", split="test")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

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
            d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i < 3:
                print(f"  Sample {i}: adherent={is_adherent}")
                print(f"    v19: unc={d19.get('uncertainty', 0):.4f}")
                print(f"    v21: unc={d21.get('uncertainty', 0):.4f}, w_prompt={d21.get('w_prompt', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class found")
        return 0.5, 0.5

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} non-adherent):")
    print(f"  v19 AUROC: {auroc_v19:.4f}")
    print(f"  v21 AUROC: {auroc_v21:.4f}  {'<-- IMPROVED!' if auroc_v21 > auroc_v19 + 0.05 else ''}")
    print(f"  Avg w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21


def test_halueval_qa(model, tokenizer, num_samples=50):
    """HaluEval QA: Standard hallucination detection (confusion)."""
    print(f"\n{'=' * 70}")
    print("HaluEval QA: Regression Test (Core Benchmark)")
    print("=" * 70)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        knowledge = sample.get('knowledge', '')
        question = sample.get('question', '')
        # Correct keys: 'answer' and 'hallucination'
        right_answer = sample.get('answer', '')
        hallucinated = sample.get('hallucination', '')

        if not right_answer or not hallucinated:
            continue

        prompt = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"

        # Test both correct and hallucinated answers
        for response, is_hall in [(right_answer, False), (hallucinated, True)]:
            try:
                d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

                v19_scores.append(d19.get('uncertainty', 0))
                v21_scores.append(d21.get('uncertainty', 0))
                labels.append(1 if is_hall else 0)
                w_prompts.append(d21.get('w_prompt', 0))

            except Exception as e:
                continue

        if i < 3:
            print(f"  Sample {i}: w_prompt={w_prompts[-1]:.4f}")

    if len(set(labels)) < 2:
        return 0.5, 0.5

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} halls):")
    print(f"  v19 AUROC: {auroc_v19:.4f}")
    print(f"  v21 AUROC: {auroc_v21:.4f}  {'<-- REGRESSION!' if auroc_v21 < auroc_v19 - 0.05 else 'OK'}")
    print(f"  Avg w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21


def test_ragtruth_qa(model, tokenizer, num_samples=50):
    """RAGTruth QA: RAG hallucination detection (deception)."""
    print(f"\n{'=' * 70}")
    print("RAGTruth QA: Regression Test (Core Benchmark)")
    print("=" * 70)

    dataset = load_dataset("flowaicom/RAGTruth_test", split="qa")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        source_info = sample.get('source_info', [])
        if isinstance(source_info, list):
            context = " ".join([s.get('text', '') for s in source_info[:3] if isinstance(s, dict)])
        else:
            context = str(source_info)

        prompt = sample.get('prompt', '')
        if not prompt:
            continue

        prompt = f"Context: {context[:1500]}\n\n{prompt}"
        response = sample.get('response', '')

        # score: 1 = faithful, 0 = hallucination
        score = sample.get('score', 1)
        label = 0 if score == 1 else 1  # label: 0 = faithful, 1 = hallucination

        try:
            d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i < 3:
                print(f"  Sample {i}: hall={label==1}, w_prompt={d21.get('w_prompt', 0):.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} halls out of {len(labels)})")
        return 0.5, 0.5

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"\n  Results ({len(labels)} samples, {sum(labels)} halls):")
    print(f"  v19 AUROC: {auroc_v19:.4f}")
    print(f"  v21 AUROC: {auroc_v21:.4f}  {'<-- REGRESSION!' if auroc_v21 < auroc_v19 - 0.05 else 'OK'}")
    print(f"  Avg w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21


def main():
    print("=" * 70)
    print("AG-SAR v21 Prompt-Gated Fusion Evaluation")
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

    print(f"\n{'#' * 70}")
    print("# Comparing v19 (Hinge-Risk) vs v21 (Prompt-Gated Fusion)")
    print(f"{'#' * 70}")

    results = {}

    # Faithfulness Tasks (v21 should excel)
    print(f"\n{'*' * 70}")
    print("* FAITHFULNESS TASKS (v21 should improve over v19)")
    print(f"{'*' * 70}")

    v19_cf, v21_cf = test_faitheval_counterfactual(model, tokenizer, num_samples=30)
    results['faitheval_counterfactual'] = {'v19': v19_cf, 'v21': v21_cf}

    v19_ua, v21_ua = test_faitheval_unanswerable(model, tokenizer, num_samples=30)
    results['faitheval_unanswerable'] = {'v19': v19_ua, 'v21': v21_ua}

    v19_rb, v21_rb = test_ragbench(model, tokenizer, num_samples=30)
    results['ragbench'] = {'v19': v19_rb, 'v21': v21_rb}

    # Core Benchmarks (v21 should maintain v19 performance)
    print(f"\n{'*' * 70}")
    print("* CORE BENCHMARKS (v21 should maintain v19 performance)")
    print(f"{'*' * 70}")

    v19_hq, v21_hq = test_halueval_qa(model, tokenizer, num_samples=30)
    results['halueval_qa'] = {'v19': v19_hq, 'v21': v21_hq}

    v19_rq, v21_rq = test_ragtruth_qa(model, tokenizer, num_samples=30)
    results['ragtruth_qa'] = {'v19': v19_rq, 'v21': v21_rq}

    # Summary
    print(f"\n{'=' * 70}")
    print("V21 PROMPT-GATED FUSION EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Dataset':<30} {'v19 AUROC':<12} {'v21 AUROC':<12} {'Delta':<10}")
    print("-" * 64)

    total_v19 = 0
    total_v21 = 0
    faith_v19 = 0
    faith_v21 = 0
    core_v19 = 0
    core_v21 = 0

    faith_datasets = ['faitheval_counterfactual', 'faitheval_unanswerable', 'ragbench']

    for name, r in results.items():
        delta = r['v21'] - r['v19']
        if name in faith_datasets:
            faith_v19 += r['v19']
            faith_v21 += r['v21']
        else:
            core_v19 += r['v19']
            core_v21 += r['v21']

        total_v19 += r['v19']
        total_v21 += r['v21']
        print(f"{name:<30} {r['v19']:.4f}       {r['v21']:.4f}       {delta:+.4f}")

    print("-" * 64)
    print(f"{'Faithfulness Avg':<30} {faith_v19/3:.4f}       {faith_v21/3:.4f}       {(faith_v21-faith_v19)/3:+.4f}")
    print(f"{'Core Benchmark Avg':<30} {core_v19/2:.4f}       {core_v21/2:.4f}       {(core_v21-core_v19)/2:+.4f}")
    print(f"{'OVERALL AVG':<30} {total_v19/5:.4f}       {total_v21/5:.4f}       {(total_v21-total_v19)/5:+.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)

    faith_improved = faith_v21/3 > faith_v19/3 + 0.05
    core_maintained = abs(core_v21/2 - core_v19/2) < 0.10

    if faith_improved and core_maintained:
        print("""
  SUCCESS: v21 Prompt-Gated Fusion achieves the goal!

  1. Faithfulness Tasks: IMPROVED
     - JSD is correctly gated on complex/counterfactual prompts
     - V×D "Safe Core" handles these tasks effectively

  2. Core Benchmarks: MAINTAINED
     - JSD remains active on simple prompts (standard RAG)
     - No significant regression on HaluEval or RAGTruth

  v21 is ready for production use. Set version=21 in AGSARConfig.
""")
    elif faith_improved:
        print("""
  PARTIAL SUCCESS: Faithfulness improved but core regressed.

  Need to tune prompt_gate_threshold to allow more JSD on core benchmarks.
  Try increasing threshold from 3.0 to 3.5-4.0.
""")
    else:
        print("""
  NEEDS INVESTIGATION: Results don't match expectations.

  Check:
  1. Are prompts being correctly classified as simple vs complex?
  2. Is w_prompt in expected range?
  3. Are there edge cases in the gating logic?
""")


if __name__ == "__main__":
    main()
