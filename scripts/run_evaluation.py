#!/usr/bin/env python3
"""
AG-SAR Evaluation on QA Benchmarks.

Runs the full hallucination detection pipeline (5 signals, entropy-gated fusion)
on QA benchmarks:
  - TriviaQA (trivia with Wikipedia context)
  - SQuAD v2 (extractive QA)

For each sample:
  1. Detector.detect() generates an answer with full signal computation
     (CUS, POS, DPS, DoLa, CGD)
  2. F1 matching against ground truth determines hallucination label
  3. Response risk is the detector's score

Metrics: AUROC, AUPRC, TPR@5%FPR, AURC, E-AURC, Risk@90, ECE, Brier

Usage:
    python scripts/run_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 200
    python scripts/run_evaluation.py --model meta-llama/Llama-3.1-8B-Instruct --all --n-samples 100
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.answer_matching import max_f1_score, compute_adaptive_f1_threshold
from evaluation.loaders.triviaqa import load_triviaqa
from evaluation.loaders.squad import load_squad


DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
}


@dataclass
class SampleResult:
    """Result for a single evaluated sample."""
    question: str
    generated_answer: str
    ground_truths: List[str]
    f1: float
    is_hallucination: bool
    response_risk: float
    mean_cus: float
    mean_pos: float
    mean_dps: float
    mean_dola: float
    mean_cgd: float
    n_tokens: int
    is_flagged: bool


def run_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    max_new_tokens: int = 64,
    f1_threshold: float = 0.3,
) -> Tuple[List[SampleResult], Dict]:
    """Run evaluation on a list of QA samples."""
    from ag_sar.detector import Detector
    from evaluation.metrics import (
        compute_metrics, bootstrap_auroc_ci,
    )

    detector = Detector(model, tokenizer)

    results: List[SampleResult] = []

    from tqdm import tqdm
    for i, sample in enumerate(tqdm(samples, desc=f"Eval ({samples[0]['dataset']})")):
        result = detector.detect(
            question=sample["question"],
            context=sample["context"],
            max_new_tokens=max_new_tokens,
        )

        generated = result.generated_text.strip()
        f1 = max_f1_score(generated, sample["answers"])
        is_hall = f1 < f1_threshold

        cus_vals = [s.cus for s in result.token_signals]
        pos_vals = [s.pos for s in result.token_signals]
        dps_vals = [s.dps for s in result.token_signals]
        dola_vals = [s.dola for s in result.token_signals]
        cgd_vals = [s.cgd for s in result.token_signals]

        results.append(SampleResult(
            question=sample["question"],
            generated_answer=generated[:200],
            ground_truths=sample["answers"][:3],
            f1=f1,
            is_hallucination=is_hall,
            response_risk=result.response_risk,
            mean_cus=float(np.mean(cus_vals)),
            mean_pos=float(np.mean(pos_vals)),
            mean_dps=float(np.mean(dps_vals)),
            mean_dola=float(np.mean(dola_vals)),
            mean_cgd=float(np.mean(cgd_vals)),
            n_tokens=result.num_tokens,
            is_flagged=result.is_flagged,
        ))

        if (i + 1) % 25 == 0:
            n_hall = sum(1 for r in results if r.is_hallucination)
            print(f"  [{i+1}/{len(samples)}] hall_rate={n_hall/len(results):.2%}, "
                  f"avg_risk={np.mean([r.response_risk for r in results]):.3f}")

    if len(results) < 5:
        return results, {"error": "Too few valid results"}

    # Adaptive F1 threshold: Otsu on collected F1 scores, then re-label
    f1_scores = [r.f1 for r in results]
    adaptive_threshold = compute_adaptive_f1_threshold(f1_scores)
    if adaptive_threshold != f1_threshold:
        print(f"  Adaptive F1 threshold: {adaptive_threshold:.3f} (was {f1_threshold:.3f})")
        for r in results:
            r.is_hallucination = r.f1 < adaptive_threshold
        f1_threshold = adaptive_threshold

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

    metrics = compute_metrics(scores, labels)

    from sklearn.metrics import roc_auc_score
    signal_aurocs = {}
    for sig_name in ["mean_cus", "mean_pos", "mean_dps", "mean_dola", "mean_cgd", "response_risk"]:
        vals = [getattr(r, sig_name) for r in results]
        signal_aurocs[sig_name] = float(roc_auc_score(labels, vals))

    ci_low, ci_high = bootstrap_auroc_ci(scores, labels, n_bootstrap=1000)

    summary = {
        "dataset": samples[0]["dataset"],
        "f1_threshold": f1_threshold,
        "n_samples": len(results),
        "n_hallucinations": n_pos,
        "n_correct": n_neg,
        "hallucination_rate": n_pos / len(results),
        "auroc": metrics.auroc,
        "auroc_ci_95": [ci_low, ci_high],
        "auprc": metrics.auprc,
        "tpr_at_5_fpr": metrics.tpr_at_5_fpr,
        "f1": metrics.f1,
        "aurc": metrics.aurc,
        "e_aurc": metrics.e_aurc,
        "risk_at_90_coverage": metrics.risk_at_90_coverage,
        "ece": metrics.expected_calibration_error,
        "brier": metrics.brier_score,
        "signal_aurocs": signal_aurocs,
        "tp": metrics.true_positives,
        "fp": metrics.false_positives,
        "tn": metrics.true_negatives,
        "fn": metrics.false_negatives,
    }

    return results, summary


def print_results(summary: Dict):
    """Print formatted evaluation results."""
    print(f"\n{'='*65}")
    print(f"  AG-SAR Evaluation Results: {summary.get('dataset', 'unknown').upper()}")
    print(f"{'='*65}")
    print(f"  Samples:           {summary['n_samples']}")
    print(f"  Hallucinations:    {summary['n_hallucinations']} ({summary['hallucination_rate']:.1%})")
    print()
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


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--all", action="store_true", help="Run all datasets")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--f1-threshold", type=float, default=0.3,
                        help="F1 below this = hallucination")
    parser.add_argument("--output", default="results/eval_{dataset}_{model_short}.json")
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Must specify --dataset or --all")

    datasets_to_run = list(DATASET_LOADERS.keys()) if args.all else [args.dataset]
    model_short = args.model.split("/")[-1]

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
        attn_implementation="eager",
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
        results, summary = run_evaluation(
            model, tokenizer, samples,
            max_new_tokens=args.max_new_tokens,
            f1_threshold=args.f1_threshold,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = elapsed
        summary["model"] = args.model
        summary["samples_per_second"] = len(results) / elapsed if elapsed > 0 else 0

        print_results(summary)

        out_path = args.output.format(dataset=dataset_name, model_short=model_short)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "summary": summary,
            "samples": [asdict(r) for r in results],
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Results saved to {out_path}")

        all_summaries[dataset_name] = summary

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
