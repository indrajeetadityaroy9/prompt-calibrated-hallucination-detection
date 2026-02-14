#!/usr/bin/env python3
"""Recompute hallucination labels and metrics from saved DSG evaluation results.

Applies extract_short_answer() to fix verbose model output inflating F1 denominators.
"""

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np


def normalize_answer(s: str) -> str:
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
    text = text.strip()
    for sep in ["\n\n", "\nExplanation:", "\nQuestion:", "\nNote:", "\nContext:"]:
        if sep in text:
            text = text[:text.index(sep)].strip()
    if len(text.split()) > 20:
        for end in [". ", ".\n"]:
            if end in text:
                text = text[:text.index(end) + 1].strip()
                break
    return text


def max_f1_score(prediction: str, ground_truths: List[str]) -> float:
    if not prediction.strip() or not ground_truths:
        return 0.0
    short = extract_short_answer(prediction)
    raw_f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    short_f1 = max(compute_f1(short, gt) for gt in ground_truths)
    return max(raw_f1, short_f1)


def compute_auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2:
        return float('nan')
    return roc_auc_score(labels, scores)


def compute_auprc(labels, scores):
    from sklearn.metrics import average_precision_score
    if len(set(labels)) < 2:
        return float('nan')
    return average_precision_score(labels, scores)


def main():
    results_dir = Path("results")

    for result_file in sorted(results_dir.glob("dsg_eval_*.json")):
        print(f"\n{'='*65}")
        print(f"  Recomputing: {result_file.name}")
        print(f"{'='*65}")

        with open(result_file) as f:
            data = json.load(f)

        samples = data["samples"]
        f1_threshold = 0.3

        # Recompute F1 with extraction
        old_halls = sum(1 for s in samples if s["is_hallucination"])

        for s in samples:
            old_f1 = s["f1"]
            new_f1 = max_f1_score(s["generated_answer"], s["ground_truths"])
            s["f1_old"] = old_f1
            s["f1"] = new_f1
            s["is_hallucination"] = new_f1 < f1_threshold

        new_halls = sum(1 for s in samples if s["is_hallucination"])

        print(f"  Old hallucinations: {old_halls}/{len(samples)} ({100*old_halls/len(samples):.1f}%)")
        print(f"  New hallucinations: {new_halls}/{len(samples)} ({100*new_halls/len(samples):.1f}%)")
        print(f"  Reclassified:       {old_halls - new_halls} samples")

        # Show some reclassified examples
        reclassified = [s for s in samples if s["f1_old"] < f1_threshold <= s["f1"]]
        if reclassified:
            print(f"\n  Reclassified examples (was hall -> now correct):")
            for s in reclassified[:5]:
                print(f"    F1: {s['f1_old']:.3f} -> {s['f1']:.3f}")
                ans_short = extract_short_answer(s['generated_answer'])
                print(f"    ANS: {ans_short[:80]}")
                print(f"    GT:  {s['ground_truths'][:3]}")
                print()

        # Recompute metrics
        labels = [1 if s["is_hallucination"] else 0 for s in samples]
        risks = [s["response_risk"] for s in samples]

        print(f"  ---- Updated Detection Performance ----")
        try:
            auroc = compute_auroc(labels, risks)
            auprc = compute_auprc(labels, risks)
            print(f"  AUROC:             {auroc:.4f}")
            print(f"  AUPRC:             {auprc:.4f}")
        except Exception as e:
            print(f"  AUROC/AUPRC error: {e}")

        # Per-signal AUROC
        print(f"\n  ---- Per-Signal AUROC ----")
        for sig in ["mean_cus", "mean_pos", "mean_dps", "response_risk"]:
            sig_vals = []
            for s in samples:
                if sig == "response_risk":
                    sig_vals.append(s["response_risk"])
                elif sig == "mean_cus":
                    sig_vals.append(np.mean([ts["cus"] for ts in s["token_signals"]]))
                elif sig == "mean_pos":
                    sig_vals.append(np.mean([ts["pos"] for ts in s["token_signals"]]))
                elif sig == "mean_dps":
                    sig_vals.append(np.mean([ts["dps"] for ts in s["token_signals"]]))
            try:
                auroc = compute_auroc(labels, sig_vals)
                print(f"  {sig:20s} {auroc:.4f}")
            except:
                print(f"  {sig:20s} N/A")

        # Risk distribution by class
        hall_risks = [s["response_risk"] for s in samples if s["is_hallucination"]]
        correct_risks = [s["response_risk"] for s in samples if not s["is_hallucination"]]
        print(f"\n  ---- Risk Distribution ----")
        if hall_risks:
            print(f"  Hallucination risks: mean={np.mean(hall_risks):.3f}, "
                  f"std={np.std(hall_risks):.3f}, "
                  f"[{np.min(hall_risks):.3f}, {np.max(hall_risks):.3f}]")
        if correct_risks:
            print(f"  Correct risks:       mean={np.mean(correct_risks):.3f}, "
                  f"std={np.std(correct_risks):.3f}, "
                  f"[{np.min(correct_risks):.3f}, {np.max(correct_risks):.3f}]")

        # Save updated results
        out_file = result_file.with_name(result_file.stem + "_fixed.json")
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\n  Saved to: {out_file}")


if __name__ == "__main__":
    main()
