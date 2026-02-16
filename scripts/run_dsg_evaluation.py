#!/usr/bin/env python3
"""
DSG (Decoupled Spectral Grounding) Evaluation on Llama 3.1.

Runs the full DSG hallucination detection pipeline (5 signals, entropy-gated fusion)
on QA benchmarks:
  - TriviaQA (trivia with Wikipedia context)
  - SQuAD v2 (extractive QA)

For each sample:
  1. DSGDetector.detect() generates an answer with full signal computation
     (CUS, POS, DPS, DoLa, CGD)
  2. F1 matching against ground truth determines hallucination label
  3. DSG response_risk is the detector's score

Metrics: AUROC, AUPRC, TPR@5%FPR, AURC, E-AURC, Risk@90, ECE, Brier

Usage:
    python scripts/run_dsg_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 200
    python scripts/run_dsg_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --all --n-samples 100
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Answer Matching (from SQuAD evaluation script)
# ============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_short_answer(text: str) -> str:
    """Extract the core answer from a verbose model response.

    Chat models often generate 'Answer\\n\\nExplanation: ...' or repeat the question.
    Extract just the first meaningful line/sentence.
    """
    text = text.strip()
    # Take first line (before double newline or "Explanation:" or "Question:")
    for sep in ["\n\n", "\nExplanation:", "\nQuestion:", "\nNote:", "\nContext:"]:
        if sep in text:
            text = text[:text.index(sep)].strip()
    # Take first sentence if still long
    if len(text.split()) > 20:
        for end in [". ", ".\n"]:
            if end in text:
                text = text[:text.index(end) + 1].strip()
                break
    return text


def max_f1_score(prediction: str, ground_truths: List[str]) -> float:
    if not prediction.strip() or not ground_truths:
        return 0.0
    # Try both raw and extracted short answer, take the max
    short = extract_short_answer(prediction)
    raw_f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    short_f1 = max(compute_f1(short, gt) for gt in ground_truths)
    return max(raw_f1, short_f1)


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_triviaqa(n_samples: int = 100) -> List[Dict]:
    from datasets import load_dataset
    print("Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        question = item["question"]
        answers = list(set(item["answer"]["aliases"] + [item["answer"]["value"]]))

        context = ""
        if item.get("search_results") and item["search_results"].get("search_context"):
            ctx_list = item["search_results"]["search_context"]
            if ctx_list:
                context = ctx_list[0][:2000]
        if not context and item.get("entity_pages") and item["entity_pages"].get("wiki_context"):
            ctx_list = item["entity_pages"]["wiki_context"]
            if ctx_list:
                context = ctx_list[0][:2000]

        if not context:
            continue

        samples.append({"question": question, "answers": answers, "context": context, "dataset": "triviaqa"})

    print(f"Loaded {len(samples)} TriviaQA samples")
    return samples


def load_squad(n_samples: int = 100) -> List[Dict]:
    from datasets import load_dataset
    print("Loading SQuAD v2...")
    dataset = load_dataset("squad_v2", split="validation", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        if not item["answers"]["text"]:
            continue
        samples.append({
            "question": item["question"],
            "answers": list(set(item["answers"]["text"])),
            "context": item["context"],
            "dataset": "squad",
        })

    print(f"Loaded {len(samples)} SQuAD samples")
    return samples


DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
}


# ============================================================================
# DSG Evaluation
# ============================================================================

@dataclass
class DSGSampleResult:
    """Result for a single evaluated sample."""
    question: str
    generated_answer: str
    ground_truths: List[str]
    f1: float
    is_hallucination: bool  # F1 < threshold
    # DSG scores
    response_risk: float
    mean_cus: float
    mean_pos: float
    mean_dps: float
    mean_dola: float
    mean_cgd: float
    n_tokens: int
    is_flagged: bool


def run_dsg_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    max_new_tokens: int = 64,
    f1_threshold: float = 0.3,
) -> Tuple[List[DSGSampleResult], Dict]:
    """Run DSG evaluation on a list of QA samples."""
    from ag_sar.config import DSGConfig
    from ag_sar.icml.dsg_detector import DSGDetector
    from ag_sar.evaluation.metrics import (
        compute_metrics, compute_aurc, compute_e_aurc,
        compute_risk_at_coverage, bootstrap_auroc_ci,
    )

    config = DSGConfig(layer_subset="all")
    detector = DSGDetector(model, tokenizer, config)

    results: List[DSGSampleResult] = []
    errors = 0

    from tqdm import tqdm
    for i, sample in enumerate(tqdm(samples, desc=f"DSG eval ({samples[0]['dataset']})")):
        try:
            result = detector.detect(
                question=sample["question"],
                context=sample["context"],
                max_new_tokens=max_new_tokens,
            )

            generated = result.generated_text.strip()
            f1 = max_f1_score(generated, sample["answers"])
            is_hall = f1 < f1_threshold

            # Per-signal means
            cus_vals = [s.cus for s in result.token_signals]
            pos_vals = [s.pos for s in result.token_signals]
            dps_vals = [s.dps for s in result.token_signals]
            dola_vals = [s.dola for s in result.token_signals]
            cgd_vals = [s.cgd for s in result.token_signals]

            results.append(DSGSampleResult(
                question=sample["question"],
                generated_answer=generated[:200],
                ground_truths=sample["answers"][:3],
                f1=f1,
                is_hallucination=is_hall,
                response_risk=result.response_risk,
                mean_cus=float(np.mean(cus_vals)) if cus_vals else 0.0,
                mean_pos=float(np.mean(pos_vals)) if pos_vals else 0.0,
                mean_dps=float(np.mean(dps_vals)) if dps_vals else 0.0,
                mean_dola=float(np.mean(dola_vals)) if dola_vals else 0.0,
                mean_cgd=float(np.mean(cgd_vals)) if cgd_vals else 0.0,
                n_tokens=result.num_tokens,
                is_flagged=result.is_flagged,
            ))

            # Progress every 25 samples
            if (i + 1) % 25 == 0:
                n_hall = sum(1 for r in results if r.is_hallucination)
                print(f"  [{i+1}/{len(samples)}] hall_rate={n_hall/len(results):.2%}, "
                      f"avg_risk={np.mean([r.response_risk for r in results]):.3f}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on sample {i}: {e}")
            continue

    if errors:
        print(f"  Total errors: {errors}/{len(samples)}")

    # Compute metrics
    if len(results) < 5:
        return results, {"error": "Too few valid results"}

    labels = [int(r.is_hallucination) for r in results]
    scores = [r.response_risk for r in results]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: Degenerate labels (n_pos={n_pos}, n_neg={n_neg})")
        return results, {
            "n_samples": len(results),
            "n_hallucinations": n_pos,
            "hallucination_rate": n_pos / len(results),
            "error": "Single-class labels, cannot compute AUROC",
        }

    metrics = compute_metrics(scores, labels, include_selective=True)

    # Per-signal AUROCs
    from sklearn.metrics import roc_auc_score
    signal_aurocs = {}
    for sig_name in ["mean_cus", "mean_pos", "mean_dps", "mean_dola", "mean_cgd", "response_risk"]:
        vals = [getattr(r, sig_name) for r in results]
        try:
            signal_aurocs[sig_name] = float(roc_auc_score(labels, vals))
        except ValueError:
            signal_aurocs[sig_name] = 0.5

    # Bootstrap CI for response_risk AUROC
    ci_low, ci_high = bootstrap_auroc_ci(scores, labels, n_bootstrap=1000)

    summary = {
        "dataset": samples[0]["dataset"],
        "n_samples": len(results),
        "n_hallucinations": n_pos,
        "n_correct": n_neg,
        "hallucination_rate": n_pos / len(results),
        "errors": errors,
        # Primary metrics
        "auroc": metrics.auroc,
        "auroc_ci_95": [ci_low, ci_high],
        "auprc": metrics.auprc,
        "tpr_at_5_fpr": metrics.tpr_at_5_fpr,
        "f1": metrics.f1,
        # Selective prediction
        "aurc": metrics.aurc,
        "e_aurc": metrics.e_aurc,
        "risk_at_90_coverage": metrics.risk_at_90_coverage,
        # Calibration
        "ece": metrics.expected_calibration_error,
        "brier": metrics.brier_score,
        # Per-signal AUROCs
        "signal_aurocs": signal_aurocs,
        # Confusion matrix
        "tp": metrics.true_positives,
        "fp": metrics.false_positives,
        "tn": metrics.true_negatives,
        "fn": metrics.false_negatives,
    }

    return results, summary


def print_results(summary: Dict):
    """Print formatted evaluation results."""
    print(f"\n{'='*65}")
    print(f"  DSG Evaluation Results: {summary.get('dataset', 'unknown').upper()}")
    print(f"{'='*65}")
    print(f"  Samples:           {summary['n_samples']}")
    print(f"  Hallucinations:    {summary['n_hallucinations']} ({summary['hallucination_rate']:.1%})")
    print(f"  Errors:            {summary.get('errors', 0)}")
    print()
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return

    print(f"  ---- Detection Performance ----")
    ci = summary.get('auroc_ci_95', [0, 0])
    print(f"  AUROC:             {summary['auroc']:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
    print(f"  AUPRC:             {summary['auprc']:.4f}")
    print(f"  TPR@5%FPR:         {summary['tpr_at_5_fpr']:.4f}")
    print(f"  F1 (at t=0.5):     {summary['f1']:.4f}")
    print()
    print(f"  ---- Selective Prediction ----")
    print(f"  AURC:              {summary['aurc']:.4f}")
    print(f"  E-AURC:            {summary['e_aurc']:.4f}")
    print(f"  Risk@90%:          {summary['risk_at_90_coverage']:.4f}")
    print()
    print(f"  ---- Calibration ----")
    print(f"  ECE:               {summary['ece']:.4f}")
    print(f"  Brier:             {summary['brier']:.4f}")
    print()
    print(f"  ---- Per-Signal AUROC ----")
    for sig, auroc in summary.get('signal_aurocs', {}).items():
        print(f"  {sig:20s} {auroc:.4f}")
    print()
    print(f"  ---- Confusion Matrix (t=0.5) ----")
    print(f"  TP={summary['tp']:4d}  FP={summary['fp']:4d}")
    print(f"  FN={summary['fn']:4d}  TN={summary['tn']:4d}")
    print(f"{'='*65}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DSG Evaluation on Llama 3.1")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--f1-threshold", type=float, default=0.3,
                        help="F1 below this = hallucination")
    parser.add_argument("--output", default="results/dsg_eval_{dataset}_{model_short}.json")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Must specify --dataset or --all")

    datasets_to_run = list(DATASET_LOADERS.keys()) if args.all else [args.dataset]
    model_short = args.model.split("/")[-1]

    # Load model
    token = os.environ.get("HF_TOKEN")
    print(f"Loading model: {args.model}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=token,
        attn_implementation="eager",  # Required for attention weight extraction
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s on {next(model.parameters()).device}")

    all_summaries = {}

    for dataset_name in datasets_to_run:
        print(f"\n{'#'*65}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*65}")

        samples = DATASET_LOADERS[dataset_name](n_samples=args.n_samples)
        if not samples:
            print(f"No samples loaded for {dataset_name}")
            continue

        t0 = time.time()
        results, summary = run_dsg_evaluation(
            model, tokenizer, samples,
            max_new_tokens=args.max_new_tokens,
            f1_threshold=args.f1_threshold,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = elapsed
        summary["model"] = args.model
        summary["samples_per_second"] = len(results) / elapsed if elapsed > 0 else 0

        print_results(summary)

        # Save per-dataset results
        out_path = args.output.format(dataset=dataset_name, model_short=model_short)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        # Save summary + per-sample results
        output_data = {
            "summary": summary,
            "samples": [asdict(r) for r in results],
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to {out_path}")

        all_summaries[dataset_name] = summary

    # Final cross-dataset summary
    if len(all_summaries) > 1:
        print(f"\n{'#'*65}")
        print("# CROSS-DATASET SUMMARY")
        print(f"{'#'*65}")
        print(f"{'Dataset':<15} {'AUROC':>8} {'AUPRC':>8} {'TPR@5':>8} {'AURC':>8} {'E-AURC':>8} {'Hall%':>8}")
        print("-" * 65)
        for name, s in all_summaries.items():
            if "error" in s:
                print(f"{name:<15} ERROR: {s['error']}")
            else:
                print(f"{name:<15} {s['auroc']:>8.4f} {s['auprc']:>8.4f} {s['tpr_at_5_fpr']:>8.4f} "
                      f"{s['aurc']:>8.4f} {s['e_aurc']:>8.4f} {s['hallucination_rate']:>7.1%}")


if __name__ == "__main__":
    main()
