#!/usr/bin/env python3
"""
Comprehensive Evaluation: Universal Veto Engine (v13.0)

Tests the final production system on all benchmark datasets:
1. RAGTruth - Real RAG hallucinations (extrinsic)
2. TruthfulQA - Myths and misconceptions (intrinsic)
3. WikiBio - Entity hallucinations (stress test)

Usage:
    python scripts/evaluate_universal_veto.py [--samples N]
"""

import sys
import os
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import AGSAR, AGSARConfig


def load_ragtruth(num_samples=None):
    """Load RAGTruth dataset."""
    dataset = load_dataset("wandb/RAGTruth-processed", split="test")

    samples = []
    for idx, row in enumerate(dataset):
        quality = row.get("quality", "good")
        if quality == "incorrect_refusal":
            continue

        labels = row.get("hallucination_labels_processed", {})
        evident_conflict = labels.get("evident_conflict", 0)
        baseless_info = labels.get("baseless_info", 0)
        is_hallucination = (evident_conflict > 0) or (baseless_info > 0)

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
        })

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def load_truthfulqa(num_samples=None):
    """Load TruthfulQA dataset."""
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    samples = []
    for idx, row in enumerate(dataset):
        question = row.get("question", "")
        best_answer = row.get("best_answer", "")
        correct_answers = row.get("correct_answers", [])
        incorrect_answers = row.get("incorrect_answers", [])

        # Create pairs: correct answer (label=0) and incorrect answer (label=1)
        if correct_answers:
            samples.append({
                "id": f"{idx}_correct",
                "prompt": f"Question: {question}\nAnswer:",
                "context": question,
                "response": f" {correct_answers[0]}",
                "label": 0,  # Not hallucination
            })

        if incorrect_answers:
            samples.append({
                "id": f"{idx}_incorrect",
                "prompt": f"Question: {question}\nAnswer:",
                "context": question,
                "response": f" {incorrect_answers[0]}",
                "label": 1,  # Hallucination (false belief)
            })

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def load_wikibio(num_samples=None):
    """Load WikiBio GPT-3 hallucination dataset."""
    try:
        dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split="evaluation")
    except Exception as e:
        print(f"Warning: Could not load WikiBio dataset: {e}")
        return []

    samples = []
    for idx, row in enumerate(dataset):
        wiki_bio_text = row.get("wiki_bio_text", "")
        gpt3_text = row.get("gpt3_text", "")
        annotation = row.get("annotation", [])

        # Annotation can be a list - check if any entry indicates inaccuracy
        if isinstance(annotation, list):
            # Check if any annotation indicates hallucination
            is_hallucination = any("inaccurate" in str(a).lower() or "minor" in str(a).lower() or "major" in str(a).lower()
                                  for a in annotation if a)
        else:
            is_hallucination = "accurate" not in str(annotation).lower()

        samples.append({
            "id": str(idx),
            "prompt": f"Biography: {wiki_bio_text}\n\nGenerated text:",
            "context": wiki_bio_text,
            "response": f" {gpt3_text}",
            "label": 1 if is_hallucination else 0,
        })

        if num_samples and len(samples) >= num_samples:
            break

    return samples


def evaluate_dataset(engine, samples, dataset_name, use_hybrid=True):
    """Evaluate on a dataset using either hybrid or baseline method."""
    scores = []
    labels = []
    veto_count = 0

    method_name = "Universal Veto" if use_hybrid else "Baseline"

    for sample in tqdm(samples, desc=f"{dataset_name} ({method_name})"):
        try:
            if use_hybrid:
                result = engine.compute_hybrid_trust(
                    sample["prompt"],
                    sample["response"],
                    context=sample.get("context"),
                )
                score = result["uncertainty"]
                if result.get("veto_triggered", False):
                    veto_count += 1
            else:
                score = engine.compute_uncertainty(sample["prompt"], sample["response"])
        except Exception as e:
            score = 0.5

        scores.append(score)
        labels.append(sample["label"])

    scores = np.array(scores)
    labels = np.array(labels)

    # Compute metrics
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float('nan')

    try:
        auprc = average_precision_score(labels, scores)
    except ValueError:
        auprc = float('nan')

    # Veto rate
    veto_rate = veto_count / len(samples) if samples else 0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "veto_count": veto_count,
        "veto_rate": veto_rate,
        "n_samples": len(samples),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Universal Veto Engine Evaluation")
    parser.add_argument("--samples", type=int, default=200, help="Samples per dataset")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", type=str, default="results/universal_veto_eval.json")
    args = parser.parse_args()

    print("=" * 70)
    print("UNIVERSAL VETO ENGINE - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples per dataset: {args.samples}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Configs
    baseline_config = AGSARConfig(
        enable_hybrid_controller=False,
        enable_semantic_dispersion=True,
        dispersion_method="centroid_variance",
        dispersion_k=10,
        aggregation_method="percentile_25",
    )

    veto_config = AGSARConfig(
        enable_hybrid_controller=True,
        enable_semantic_dispersion=True,
        dispersion_method="centroid_variance",
        dispersion_k=10,
        aggregation_method="percentile_25",
        symbolic_weight=0.0,
        jepa_weight=1.0,
        enable_numeric_check=True,
    )

    results = {}

    # ==================== RAGTruth ====================
    print("\n" + "=" * 70)
    print("DATASET 1: RAGTruth (Extrinsic Hallucinations)")
    print("=" * 70)

    ragtruth_samples = load_ragtruth(args.samples)
    print(f"Loaded {len(ragtruth_samples)} samples")
    print(f"Labels: {sum(s['label'] for s in ragtruth_samples)} hallucinations, "
          f"{len(ragtruth_samples) - sum(s['label'] for s in ragtruth_samples)} faithful")

    # Baseline
    engine = AGSAR(model, tokenizer, baseline_config)
    ragtruth_baseline = evaluate_dataset(engine, ragtruth_samples, "RAGTruth", use_hybrid=False)
    engine.cleanup()

    # Universal Veto
    engine = AGSAR(model, tokenizer, veto_config)
    ragtruth_veto = evaluate_dataset(engine, ragtruth_samples, "RAGTruth", use_hybrid=True)
    engine.cleanup()

    results["ragtruth"] = {
        "baseline": ragtruth_baseline,
        "veto": ragtruth_veto,
    }

    print(f"\nRAGTruth Results:")
    print(f"  Baseline AUROC: {ragtruth_baseline['auroc']:.4f}")
    print(f"  Veto AUROC:     {ragtruth_veto['auroc']:.4f} ({ragtruth_veto['auroc'] - ragtruth_baseline['auroc']:+.4f})")
    print(f"  Veto Rate:      {ragtruth_veto['veto_rate']*100:.1f}%")

    # ==================== TruthfulQA ====================
    print("\n" + "=" * 70)
    print("DATASET 2: TruthfulQA (Intrinsic Hallucinations)")
    print("=" * 70)

    truthfulqa_samples = load_truthfulqa(args.samples)
    print(f"Loaded {len(truthfulqa_samples)} samples")
    print(f"Labels: {sum(s['label'] for s in truthfulqa_samples)} false, "
          f"{len(truthfulqa_samples) - sum(s['label'] for s in truthfulqa_samples)} true")

    # Baseline
    engine = AGSAR(model, tokenizer, baseline_config)
    truthfulqa_baseline = evaluate_dataset(engine, truthfulqa_samples, "TruthfulQA", use_hybrid=False)
    engine.cleanup()

    # Universal Veto
    engine = AGSAR(model, tokenizer, veto_config)
    truthfulqa_veto = evaluate_dataset(engine, truthfulqa_samples, "TruthfulQA", use_hybrid=True)
    engine.cleanup()

    results["truthfulqa"] = {
        "baseline": truthfulqa_baseline,
        "veto": truthfulqa_veto,
    }

    print(f"\nTruthfulQA Results:")
    print(f"  Baseline AUROC: {truthfulqa_baseline['auroc']:.4f}")
    print(f"  Veto AUROC:     {truthfulqa_veto['auroc']:.4f} ({truthfulqa_veto['auroc'] - truthfulqa_baseline['auroc']:+.4f})")
    print(f"  Veto Rate:      {truthfulqa_veto['veto_rate']*100:.1f}%")

    # ==================== WikiBio ====================
    print("\n" + "=" * 70)
    print("DATASET 3: WikiBio (Entity Hallucinations)")
    print("=" * 70)

    wikibio_samples = load_wikibio(args.samples)

    if wikibio_samples:
        print(f"Loaded {len(wikibio_samples)} samples")
        print(f"Labels: {sum(s['label'] for s in wikibio_samples)} inaccurate, "
              f"{len(wikibio_samples) - sum(s['label'] for s in wikibio_samples)} accurate")

        # Baseline
        engine = AGSAR(model, tokenizer, baseline_config)
        wikibio_baseline = evaluate_dataset(engine, wikibio_samples, "WikiBio", use_hybrid=False)
        engine.cleanup()

        # Universal Veto
        engine = AGSAR(model, tokenizer, veto_config)
        wikibio_veto = evaluate_dataset(engine, wikibio_samples, "WikiBio", use_hybrid=True)
        engine.cleanup()

        results["wikibio"] = {
            "baseline": wikibio_baseline,
            "veto": wikibio_veto,
        }

        print(f"\nWikiBio Results:")
        print(f"  Baseline AUROC: {wikibio_baseline['auroc']:.4f}")
        print(f"  Veto AUROC:     {wikibio_veto['auroc']:.4f} ({wikibio_veto['auroc'] - wikibio_baseline['auroc']:+.4f})")
        print(f"  Veto Rate:      {wikibio_veto['veto_rate']*100:.1f}%")
    else:
        print("WikiBio dataset not available, skipping...")
        results["wikibio"] = None

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: UNIVERSAL VETO ENGINE (v13.0)")
    print("=" * 70)

    print("\n┌─────────────────┬──────────────┬──────────────┬──────────┬───────────┐")
    print("│ Dataset         │ Baseline     │ Veto         │ Delta    │ Veto Rate │")
    print("├─────────────────┼──────────────┼──────────────┼──────────┼───────────┤")

    for name, data in results.items():
        if data is None:
            continue
        baseline_auroc = data["baseline"]["auroc"]
        veto_auroc = data["veto"]["auroc"]
        delta = veto_auroc - baseline_auroc
        veto_rate = data["veto"]["veto_rate"] * 100
        print(f"│ {name:15} │ {baseline_auroc:12.4f} │ {veto_auroc:12.4f} │ {delta:+8.4f} │ {veto_rate:8.1f}% │")

    print("└─────────────────┴──────────────┴──────────────┴──────────┴───────────┘")

    # Save results
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "samples_per_dataset": args.samples,
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
