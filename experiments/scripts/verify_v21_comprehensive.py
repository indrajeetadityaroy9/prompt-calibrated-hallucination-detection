#!/usr/bin/env python3
"""
AG-SAR v21 Comprehensive Evaluation

Tests v21 on ALL datasets and task types:
- FaithEval: Counterfactual, Unanswerable
- RAGBench: CovidQA
- HaluEval: QA, Summarization
- RAGTruth: QA, Summarization
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


def run_comparison(model, tokenizer, prompt, response):
    """Run both v19 and v21 on the same prompt/response."""
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


# ============================================================================
# FAITHEVAL DATASETS
# ============================================================================

def test_faitheval_counterfactual(model, tokenizer, num_samples=100):
    """FaithEval Counterfactual: Context contradicts world knowledge."""
    print(f"\n{'=' * 70}")
    print("FaithEval Counterfactual")
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

            if i % 20 == 0:
                print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

        except Exception as e:
            continue

    if len(set(labels)) < 2:
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} overrides)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


def test_faitheval_unanswerable(model, tokenizer, num_samples=100):
    """FaithEval Unanswerable: Context doesn't contain the answer."""
    print(f"\n{'=' * 70}")
    print("FaithEval Unanswerable")
    print("=" * 70)

    dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

    abstention_keywords = ['unanswerable', 'unknown', 'not available', 'no answer',
                          'cannot be determined', 'not mentioned', 'not provided',
                          "don't know", "doesn't say", 'unclear', 'not stated',
                          'not possible', 'cannot answer', 'no information']

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

            if i % 20 == 0:
                print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

        except Exception as e:
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} fabrications)")
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} fabrications)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


# ============================================================================
# RAGBENCH
# ============================================================================

def test_ragbench(model, tokenizer, num_samples=100):
    """RAGBench: Pre-labeled RAG adherence scores."""
    print(f"\n{'=' * 70}")
    print("RAGBench (CovidQA)")
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

            if i % 20 == 0:
                print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

        except Exception as e:
            continue

    if len(set(labels)) < 2:
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} non-adherent)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


# ============================================================================
# HALUEVAL DATASETS
# ============================================================================

def test_halueval_qa(model, tokenizer, num_samples=100):
    """HaluEval QA: Standard hallucination detection."""
    print(f"\n{'=' * 70}")
    print("HaluEval QA")
    print("=" * 70)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        knowledge = sample.get('knowledge', '')
        question = sample.get('question', '')
        right_answer = sample.get('answer', '')
        hallucinated = sample.get('hallucination', '')

        if not right_answer or not hallucinated:
            continue

        prompt = f"Knowledge: {knowledge}\nQuestion: {question}\nAnswer:"

        for response, is_hall in [(right_answer, False), (hallucinated, True)]:
            try:
                d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

                v19_scores.append(d19.get('uncertainty', 0))
                v21_scores.append(d21.get('uncertainty', 0))
                labels.append(1 if is_hall else 0)
                w_prompts.append(d21.get('w_prompt', 0))

            except Exception as e:
                continue

        if i % 20 == 0:
            print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

    if len(set(labels)) < 2:
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} halls)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


def test_halueval_summarization(model, tokenizer, num_samples=100):
    """HaluEval Summarization: Summarization hallucination detection."""
    print(f"\n{'=' * 70}")
    print("HaluEval Summarization")
    print("=" * 70)

    dataset = load_dataset("pminervini/HaluEval", "summarization_samples", split="data")

    v19_scores, v21_scores, labels, w_prompts = [], [], [], []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        document = sample.get('document', '')
        right_summary = sample.get('right_summary', sample.get('summary', ''))
        hallucinated = sample.get('hallucinated_summary', sample.get('hallucination', ''))

        if not right_summary or not hallucinated:
            continue

        prompt = f"Document: {document[:1500]}\n\nSummarize the above document:"

        for response, is_hall in [(right_summary, False), (hallucinated, True)]:
            try:
                d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

                v19_scores.append(d19.get('uncertainty', 0))
                v21_scores.append(d21.get('uncertainty', 0))
                labels.append(1 if is_hall else 0)
                w_prompts.append(d21.get('w_prompt', 0))

            except Exception as e:
                continue

        if i % 20 == 0:
            print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

    if len(set(labels)) < 2:
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} halls)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


# ============================================================================
# RAGTRUTH DATASETS
# ============================================================================

def test_ragtruth_qa(model, tokenizer, num_samples=100):
    """RAGTruth QA: RAG hallucination detection."""
    print(f"\n{'=' * 70}")
    print("RAGTruth QA")
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

        score = sample.get('score', 1)
        label = 0 if score == 1 else 1

        try:
            d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i % 20 == 0:
                print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

        except Exception as e:
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} halls out of {len(labels)})")
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} halls)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


def test_ragtruth_summarization(model, tokenizer, num_samples=100):
    """RAGTruth Summarization: RAG summarization hallucination detection."""
    print(f"\n{'=' * 70}")
    print("RAGTruth Summarization")
    print("=" * 70)

    dataset = load_dataset("flowaicom/RAGTruth_test", split="summarization")

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

        score = sample.get('score', 1)
        label = 0 if score == 1 else 1

        try:
            d19, d21 = run_comparison(model, tokenizer, prompt, " " + response)

            v19_scores.append(d19.get('uncertainty', 0))
            v21_scores.append(d21.get('uncertainty', 0))
            labels.append(label)
            w_prompts.append(d21.get('w_prompt', 0))

            if i % 20 == 0:
                print(f"  Processed {i+1}/{min(num_samples, len(dataset))} samples...")

        except Exception as e:
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class ({sum(labels)} halls out of {len(labels)})")
        return 0.5, 0.5, 0.0

    auroc_v19 = roc_auc_score(labels, v19_scores)
    auroc_v21 = roc_auc_score(labels, v21_scores)

    print(f"  Samples: {len(labels)} ({sum(labels)} halls)")
    print(f"  v19: {auroc_v19:.4f}, v21: {auroc_v21:.4f}, w_prompt: {np.mean(w_prompts):.4f}")

    return auroc_v19, auroc_v21, np.mean(w_prompts)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("AG-SAR v21 COMPREHENSIVE EVALUATION")
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

    num_samples = 100  # Per dataset

    results = {}

    # FaithEval
    print(f"\n{'#' * 70}")
    print("# FAITHEVAL BENCHMARKS")
    print(f"{'#' * 70}")

    v19, v21, w = test_faitheval_counterfactual(model, tokenizer, num_samples)
    results['FaithEval-Counterfactual'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Faithfulness'}

    v19, v21, w = test_faitheval_unanswerable(model, tokenizer, num_samples)
    results['FaithEval-Unanswerable'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Faithfulness'}

    # RAGBench
    print(f"\n{'#' * 70}")
    print("# RAGBENCH")
    print(f"{'#' * 70}")

    v19, v21, w = test_ragbench(model, tokenizer, num_samples)
    results['RAGBench'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Faithfulness'}

    # HaluEval
    print(f"\n{'#' * 70}")
    print("# HALUEVAL BENCHMARKS")
    print(f"{'#' * 70}")

    v19, v21, w = test_halueval_qa(model, tokenizer, num_samples)
    results['HaluEval-QA'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Core'}

    v19, v21, w = test_halueval_summarization(model, tokenizer, num_samples)
    results['HaluEval-Summ'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Core'}

    # RAGTruth
    print(f"\n{'#' * 70}")
    print("# RAGTRUTH BENCHMARKS")
    print(f"{'#' * 70}")

    v19, v21, w = test_ragtruth_qa(model, tokenizer, num_samples)
    results['RAGTruth-QA'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Core'}

    v19, v21, w = test_ragtruth_summarization(model, tokenizer, num_samples)
    results['RAGTruth-Summ'] = {'v19': v19, 'v21': v21, 'w_prompt': w, 'category': 'Core'}

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Dataset':<28} {'v19 AUROC':<12} {'v21 AUROC':<12} {'Delta':<10} {'w_prompt':<10}")
    print("-" * 72)

    faith_v19, faith_v21 = 0, 0
    core_v19, core_v21 = 0, 0
    faith_count, core_count = 0, 0

    for name, r in results.items():
        delta = r['v21'] - r['v19']
        status = "+" if delta > 0 else ""
        print(f"{name:<28} {r['v19']:.4f}       {r['v21']:.4f}       {status}{delta:.4f}     {r['w_prompt']:.4f}")

        if r['category'] == 'Faithfulness':
            faith_v19 += r['v19']
            faith_v21 += r['v21']
            faith_count += 1
        else:
            core_v19 += r['v19']
            core_v21 += r['v21']
            core_count += 1

    print("-" * 72)

    if faith_count > 0:
        print(f"{'Faithfulness Avg':<28} {faith_v19/faith_count:.4f}       {faith_v21/faith_count:.4f}       {(faith_v21-faith_v19)/faith_count:+.4f}")

    if core_count > 0:
        print(f"{'Core Benchmark Avg':<28} {core_v19/core_count:.4f}       {core_v21/core_count:.4f}       {(core_v21-core_v19)/core_count:+.4f}")

    total_v19 = faith_v19 + core_v19
    total_v21 = faith_v21 + core_v21
    total_count = faith_count + core_count

    print(f"{'OVERALL AVERAGE':<28} {total_v19/total_count:.4f}       {total_v21/total_count:.4f}       {(total_v21-total_v19)/total_count:+.4f}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS")
    print("=" * 80)

    print(f"""
  v21 Prompt-Gated Fusion Key Metrics:

  Faithfulness Tasks (FaithEval + RAGBench):
    - v19 Average: {faith_v19/faith_count:.4f}
    - v21 Average: {faith_v21/faith_count:.4f}
    - Improvement: {(faith_v21-faith_v19)/faith_count:+.4f}

  Core Benchmarks (HaluEval + RAGTruth):
    - v19 Average: {core_v19/core_count:.4f}
    - v21 Average: {core_v21/core_count:.4f}
    - Change: {(core_v21-core_v19)/core_count:+.4f}

  Prompt Gating Analysis:
    - High w_prompt (>0.6) = JSD suppressed = rely on V×D
    - Low w_prompt (<0.4) = JSD active = full detection

  Interpretation:
    - v21 gates JSD based on prompt complexity (varentropy)
    - Complex prompts (counterfactual/unanswerable): w_prompt high → V×D only
    - Standard prompts: w_prompt moderate → balanced detection
""")


if __name__ == "__main__":
    main()
