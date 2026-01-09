#!/usr/bin/env python3
"""
AG-SAR v21 Full Evaluation - All 7 Datasets

Datasets:
1. FaithEval-Counterfactual: Context contradicts world knowledge
2. FaithEval-Unanswerable: Context doesn't contain the answer
3. RAGBench: Pre-labeled adherence scores
4. HaluEval-QA: Knowledge-grounded QA
5. HaluEval-Summarization: Document summarization
6. RAGTruth-QA: RAG question answering
7. RAGTruth-Summarization: RAG summarization
"""

import os
import sys
import warnings
import json
from datetime import datetime
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ag_sar import AGSAR, AGSARConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_metrics(labels, scores):
    """Compute all evaluation metrics."""
    labels = np.array(labels)
    scores = np.array(scores)

    if len(np.unique(labels)) < 2:
        return {"auroc": 0.5, "auprc": 0.5, "f1": 0.0, "tpr_at_5fpr": 0.0}

    auroc = roc_auc_score(labels, scores)

    # AUPRC
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)

    # TPR at 5% FPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(fpr - 0.05))
    tpr_at_5fpr = tpr[idx]

    # F1 at optimal threshold
    best_f1 = 0
    for t in np.linspace(0, 1, 100):
        preds = (scores > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": best_f1,
        "tpr_at_5fpr": tpr_at_5fpr,
    }


def test_faitheval_counterfactual(agsar, tokenizer, model, num_samples=200):
    """FaithEval Counterfactual: Context contradicts world knowledge."""
    print(f"\n{'='*70}")
    print("Dataset: FaithEval-Counterfactual")
    print("="*70)

    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")

    scores, labels = [], []

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
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_letter = response.strip().upper()[:1]
        is_faithful = response_letter == correct_key
        label = 0 if is_faithful else 1

        try:
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            score = details.get('uncertainty', details.get('score', 0))
            scores.append(score)
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i}/{min(num_samples, len(dataset))} samples...")
        except Exception as e:
            continue

    metrics = compute_metrics(labels, scores)
    print(f"  Samples: {len(scores)} (hall={sum(labels)}, faithful={len(labels)-sum(labels)})")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    return metrics


def test_faitheval_unanswerable(agsar, tokenizer, model, num_samples=200):
    """FaithEval Unanswerable: Context doesn't contain the answer."""
    print(f"\n{'='*70}")
    print("Dataset: FaithEval-Unanswerable")
    print("="*70)

    dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")

    scores, labels = [], []
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
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        response_lower = response.lower()
        is_abstention = any(kw in response_lower for kw in abstention_keywords)
        label = 0 if is_abstention else 1

        try:
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            score = details.get('uncertainty', details.get('score', 0))
            scores.append(score)
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i}/{min(num_samples, len(dataset))} samples...")
        except Exception as e:
            continue

    metrics = compute_metrics(labels, scores)
    print(f"  Samples: {len(scores)} (fabricated={sum(labels)}, abstained={len(labels)-sum(labels)})")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    return metrics


def test_ragbench(agsar, num_samples=200):
    """RAGBench: Pre-labeled adherence scores."""
    print(f"\n{'='*70}")
    print("Dataset: RAGBench")
    print("="*70)

    dataset = load_dataset("rungalileo/ragbench", "covidqa", split="test")

    scores, labels = [], []

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
            details = agsar.compute_uncertainty(prompt, " " + response, return_details=True)
            score = details.get('uncertainty', details.get('score', 0))
            scores.append(score)
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i}/{min(num_samples, len(dataset))} samples...")
        except Exception as e:
            continue

    metrics = compute_metrics(labels, scores)
    print(f"  Samples: {len(scores)} (non-adherent={sum(labels)}, adherent={len(labels)-sum(labels)})")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    return metrics


def test_halueval(agsar, tokenizer, model, variant="qa", num_samples=200):
    """HaluEval QA or Summarization."""
    print(f"\n{'='*70}")
    print(f"Dataset: HaluEval-{variant.upper()}")
    print("="*70)

    from experiments.data.halueval import HaluEvalDataset
    dataset = HaluEvalDataset(variant=variant, num_samples=num_samples, seed=42)

    scores, labels = [], []

    for i, sample in enumerate(dataset):
        prompt = sample.prompt
        response = sample.response
        label = sample.label

        try:
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            score = details.get('uncertainty', details.get('score', 0))
            scores.append(score)
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i}/{len(dataset)} samples...")
        except Exception as e:
            continue

    metrics = compute_metrics(labels, scores)
    print(f"  Samples: {len(scores)} (hall={sum(labels)}, faithful={len(labels)-sum(labels)})")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    return metrics


def test_ragtruth(agsar, task_type="QA", num_samples=200):
    """RAGTruth QA or Summarization."""
    print(f"\n{'='*70}")
    print(f"Dataset: RAGTruth-{task_type}")
    print("="*70)

    from experiments.data.ragtruth import RAGTruthDataset
    dataset = RAGTruthDataset(task_type=task_type, num_samples=num_samples, seed=42)

    scores, labels = [], []

    for i, sample in enumerate(dataset):
        prompt = sample.prompt
        response = sample.response
        label = sample.label

        try:
            details = agsar.compute_uncertainty(prompt, response, return_details=True)
            score = details.get('uncertainty', details.get('score', 0))
            scores.append(score)
            labels.append(label)

            if i % 50 == 0:
                print(f"  Processed {i}/{len(dataset)} samples...")
        except Exception as e:
            continue

    metrics = compute_metrics(labels, scores)
    print(f"  Samples: {len(scores)} (hall={sum(labels)}, faithful={len(labels)-sum(labels)})")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    return metrics


def main():
    print("="*70)
    print("AG-SAR v21 FULL EVALUATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # Configuration
    NUM_SAMPLES = 200
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    VERSION = 21

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  AG-SAR Version: {VERSION}")
    print(f"  Samples per dataset: {NUM_SAMPLES}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    print(f"  Model loaded on {next(model.parameters()).device}")

    # Initialize AG-SAR
    config = AGSARConfig(version=VERSION)
    agsar = AGSAR(model, tokenizer, config=config)
    print(f"  AG-SAR v{VERSION} initialized")

    # Run evaluations
    results = {}

    # 1. FaithEval-Counterfactual
    results['FaithEval-Counterfactual'] = test_faitheval_counterfactual(
        agsar, tokenizer, model, NUM_SAMPLES
    )

    # 2. FaithEval-Unanswerable
    results['FaithEval-Unanswerable'] = test_faitheval_unanswerable(
        agsar, tokenizer, model, NUM_SAMPLES
    )

    # 3. RAGBench
    results['RAGBench'] = test_ragbench(agsar, NUM_SAMPLES)

    # 4. HaluEval-QA
    results['HaluEval-QA'] = test_halueval(agsar, tokenizer, model, "qa", NUM_SAMPLES)

    # 5. HaluEval-Summarization
    results['HaluEval-Summarization'] = test_halueval(
        agsar, tokenizer, model, "summarization", NUM_SAMPLES
    )

    # 6. RAGTruth-QA
    results['RAGTruth-QA'] = test_ragtruth(agsar, "QA", NUM_SAMPLES)

    # 7. RAGTruth-Summarization
    results['RAGTruth-Summarization'] = test_ragtruth(agsar, "summarization", NUM_SAMPLES)

    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY - AG-SAR v21")
    print("="*70)
    print(f"\n{'Dataset':<30} {'AUROC':>10} {'AUPRC':>10} {'F1':>10} {'TPR@5%':>10}")
    print("-"*70)

    aurocs = []
    for name, metrics in results.items():
        aurocs.append(metrics['auroc'])
        print(f"{name:<30} {metrics['auroc']:>10.4f} {metrics['auprc']:>10.4f} "
              f"{metrics['f1']:>10.4f} {metrics['tpr_at_5fpr']:>10.4f}")

    print("-"*70)
    print(f"{'AVERAGE':<30} {np.mean(aurocs):>10.4f}")
    print("="*70)

    # Save results
    output_file = f"results/v21_full_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            'version': VERSION,
            'model': MODEL_NAME,
            'num_samples': NUM_SAMPLES,
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'average_auroc': np.mean(aurocs),
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
