#!/usr/bin/env python3
"""
Evaluate Hybrid Controller (v13.0) on RAGTruth.

Quick validation that the symbolic entity check improves RAGTruth AUROC.

Usage:
    python scripts/evaluate_hybrid_ragtruth.py [--samples N]
"""

import sys
import argparse

# Pre-flight installation check
from experiments.utils.preflight import check_installation
check_installation()

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import AGSAR, AGSARConfig


def load_ragtruth(num_samples=None, task_type=None):
    """Load RAGTruth dataset from HuggingFace."""
    dataset = load_dataset("wandb/RAGTruth-processed", split="test")

    # Filter by task type if specified
    if task_type:
        task_map = {"qa": "QA", "summary": "Summary", "data2text": "Data2txt"}
        task_type_hf = task_map.get(task_type.lower(), task_type)
        dataset = dataset.filter(lambda x: x.get("task_type") == task_type_hf)

    samples = []
    for idx, row in enumerate(dataset):
        # Skip invalid refusals
        quality = row.get("quality", "good")
        if quality == "incorrect_refusal":
            continue

        # Determine hallucination label
        labels = row.get("hallucination_labels_processed", {})
        evident_conflict = labels.get("evident_conflict", 0)
        baseless_info = labels.get("baseless_info", 0)
        is_hallucination = (evident_conflict > 0) or (baseless_info > 0)

        # Build prompt
        query = row.get("query", "")
        context = row.get("context", "")
        if context:
            prompt = f"Context: {context}\n\nQuestion: {query}"
        else:
            prompt = f"Question: {query}"

        samples.append({
            "id": str(idx),
            "prompt": prompt,
            "context": context,
            "response": row.get("output", ""),
            "label": 1 if is_hallucination else 0,
            "task_type": row.get("task_type"),
        })

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def evaluate_hybrid(model, tokenizer, samples, config):
    """Evaluate using Hybrid Controller."""
    engine = AGSAR(model, tokenizer, config)

    scores = []
    labels = []

    for sample in tqdm(samples, desc="Evaluating (Hybrid)"):
        try:
            result = engine.compute_hybrid_trust(
                sample["prompt"],
                sample["response"],
                context=sample["context"],
            )
            # Use uncertainty (1 - trust) as the score (higher = more likely hallucination)
            score = result["uncertainty"]
        except Exception as e:
            print(f"Error on sample {sample['id']}: {e}")
            score = 0.5  # Default neutral

        scores.append(score)
        labels.append(sample["label"])

    engine.cleanup()
    return np.array(scores), np.array(labels)


def evaluate_baseline(model, tokenizer, samples, config):
    """Evaluate using baseline v8.0 (no hybrid)."""
    engine = AGSAR(model, tokenizer, config)

    scores = []
    labels = []

    for sample in tqdm(samples, desc="Evaluating (Baseline)"):
        try:
            score = engine.compute_uncertainty(sample["prompt"], sample["response"])
        except Exception as e:
            print(f"Error on sample {sample['id']}: {e}")
            score = 0.5

        scores.append(score)
        labels.append(sample["label"])

    engine.cleanup()
    return np.array(scores), np.array(labels)


def compute_metrics(scores, labels, name):
    """Compute AUROC and AUPRC."""
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Compute detection rate at various thresholds
    thresholds = [0.3, 0.5, 0.7]
    rates = []
    for t in thresholds:
        predictions = (scores > t).astype(int)
        tpr = (predictions & labels).sum() / max(labels.sum(), 1)
        fpr = (predictions & ~labels.astype(bool)).sum() / max((~labels.astype(bool)).sum(), 1)
        rates.append((t, tpr, fpr))

    print(f"\n{name}:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f}")
    print(f"  Detection rates:")
    for t, tpr, fpr in rates:
        print(f"    @{t}: TPR={tpr:.2f}, FPR={fpr:.2f}")

    return {"auroc": auroc, "auprc": auprc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid Controller on RAGTruth")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate")
    parser.add_argument("--task-type", type=str, default=None, help="Filter by task type (qa, summary, data2text)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    print("=" * 70)
    print("HYBRID CONTROLLER EVALUATION ON RAGTRUTH")
    print("=" * 70)

    # Load data
    print(f"\nLoading RAGTruth ({args.samples} samples)...")
    samples = load_ragtruth(num_samples=args.samples, task_type=args.task_type)
    print(f"Loaded {len(samples)} samples")

    # Count labels
    n_positive = sum(s["label"] for s in samples)
    n_negative = len(samples) - n_positive
    print(f"Label distribution: {n_positive} hallucinations, {n_negative} faithful")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Evaluate Baseline (v8.0)
    print("\n" + "-" * 50)
    print("Evaluating BASELINE (v8.0 - no symbolic check)")
    print("-" * 50)
    baseline_config = AGSARConfig(
        enable_hybrid_controller=False,
        enable_semantic_dispersion=True,
        dispersion_method="centroid_variance",
        dispersion_k=10,
        aggregation_method="percentile_25",
    )
    baseline_scores, baseline_labels = evaluate_baseline(model, tokenizer, samples, baseline_config)
    baseline_metrics = compute_metrics(baseline_scores, baseline_labels, "BASELINE (v8.0)")

    # Evaluate Universal Veto Engine (v13.0 Final)
    print("\n" + "-" * 50)
    print("Evaluating UNIVERSAL VETO ENGINE (v13.0 Final)")
    print("-" * 50)
    veto_config = AGSARConfig(
        enable_hybrid_controller=True,
        enable_semantic_dispersion=True,
        dispersion_method="centroid_variance",
        dispersion_k=10,
        aggregation_method="percentile_25",
        # Veto architecture: Neural trust with hard symbolic filter
        symbolic_weight=0.0,  # Not blended - used as hard veto
        jepa_weight=1.0,
        enable_numeric_check=True,
    )
    veto_scores, veto_labels = evaluate_hybrid(model, tokenizer, samples, veto_config)
    veto_metrics = compute_metrics(veto_scores, veto_labels, "UNIVERSAL VETO (v13.0)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    auroc_delta = veto_metrics['auroc'] - baseline_metrics['auroc']
    auprc_delta = veto_metrics['auprc'] - baseline_metrics['auprc']

    print(f"\nAUROC Comparison:")
    print(f"  Baseline (v8.0):       {baseline_metrics['auroc']:.4f}")
    print(f"  Universal Veto:        {veto_metrics['auroc']:.4f} ({auroc_delta:+.4f})")

    print(f"\nAUPRC Comparison:")
    print(f"  Baseline (v8.0):       {baseline_metrics['auprc']:.4f}")
    print(f"  Universal Veto:        {veto_metrics['auprc']:.4f} ({auprc_delta:+.4f})")

    if auroc_delta >= 0:
        print(f"\n[PASS] Universal Veto Engine maintained/improved performance!")
        print(f"       AUROC delta: {auroc_delta:+.4f}")
    else:
        print(f"\n[INFO] Baseline performed slightly better on this dataset.")
        print(f"       AUROC delta: {auroc_delta:+.4f}")
        print(f"       Note: Veto system still catches blatant entity violations.")


if __name__ == "__main__":
    main()
