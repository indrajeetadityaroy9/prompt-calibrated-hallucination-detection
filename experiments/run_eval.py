"""Evaluation orchestration — runs AG-SAR detection on QA benchmarks."""

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ag_sar.detector import Detector
from .answer_matching import max_f1_score, compute_adaptive_f1_threshold
from .metrics import compute_metrics, bootstrap_auroc_ci

from .schema import ExperimentConfig
from .common import load_dataset, save_results


@dataclass
class SampleResult:
    question: str
    generated_answer: str
    ground_truths: list[str]
    f1: float
    is_hallucination: bool
    response_risk: float
    mean_cus: float
    mean_pos: float
    mean_dps: float
    mean_spt: float
    mean_spectral_gap: float
    n_tokens: int
    is_flagged: bool


def _run_dataset(
    detector: Detector,
    samples: list[dict],
    config: ExperimentConfig,
) -> tuple[list[SampleResult], dict]:
    """Evaluate a single dataset."""
    results: list[SampleResult] = []
    print_interval = max(1, config.evaluation.n_samples // 4)

    for i, sample in enumerate(tqdm(samples, desc=f"Eval ({samples[0]['dataset']})")):
        result = detector.detect(
            question=sample["question"],
            context=sample["context"],
            max_new_tokens=config.evaluation.max_new_tokens,
        )

        generated = result.generated_text.strip()
        f1 = max_f1_score(generated, sample["answers"])

        mean_cus = float(np.mean([s.cus for s in result.token_signals]))
        mean_pos = float(np.mean([s.pos for s in result.token_signals]))
        mean_dps = float(np.mean([s.dps for s in result.token_signals]))
        mean_spt = float(np.mean([s.spt for s in result.token_signals]))
        mean_gap = float(np.mean([s.spectral_gap for s in result.token_signals]))

        results.append(SampleResult(
            question=sample["question"],
            generated_answer=generated,
            ground_truths=sample["answers"],
            f1=f1,
            is_hallucination=False,  # set by adaptive Otsu below
            response_risk=result.response_risk,
            mean_cus=mean_cus,
            mean_pos=mean_pos,
            mean_dps=mean_dps,
            mean_spt=mean_spt,
            mean_spectral_gap=mean_gap,
            n_tokens=result.num_tokens,
            is_flagged=result.is_flagged,
        ))

        if (i + 1) % print_interval == 0:
            print(f"  [{i+1}/{len(samples)}] "
                  f"avg_risk={np.mean([r.response_risk for r in results]):.3f}")

    # Adaptive F1 threshold via Otsu — parameter-free hallucination labeling
    f1_scores = [r.f1 for r in results]
    adaptive_threshold = compute_adaptive_f1_threshold(f1_scores)
    print(f"  Adaptive F1 threshold: {adaptive_threshold:.3f}")
    for r in results:
        r.is_hallucination = r.f1 < adaptive_threshold

    labels = [int(r.is_hallucination) for r in results]
    scores = [r.response_risk for r in results]

    metrics = compute_metrics(scores, labels)

    signal_aurocs = {}
    for sig_name in ["mean_cus", "mean_pos", "mean_dps", "mean_spt", "mean_spectral_gap", "response_risk"]:
        vals = [getattr(r, sig_name) for r in results]
        signal_aurocs[sig_name] = float(roc_auc_score(labels, vals))

    ci_low, ci_high = bootstrap_auroc_ci(scores, labels, seed=config.evaluation.seed)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    summary = {
        "dataset": samples[0]["dataset"],
        "f1_threshold": adaptive_threshold,
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


def _print_results(summary: dict):
    """Print formatted evaluation results."""
    print(f"\n{'='*65}")
    print(f"  AG-SAR Evaluation Results: {summary['dataset'].upper()}")
    print(f"{'='*65}")
    print(f"  Samples:           {summary['n_samples']}")
    print(f"  Hallucinations:    {summary['n_hallucinations']} ({summary['hallucination_rate']:.1%})")
    print()
    print(f"  ---- Detection Performance ----")
    ci = summary['auroc_ci_95']
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
    for sig, auroc in summary['signal_aurocs'].items():
        print(f"  {sig:20s} {auroc:.4f}")
    print()
    print(f"  ---- Confusion Matrix (t=0.5) ----")
    print(f"  TP={summary['tp']:4d}  FP={summary['fp']:4d}")
    print(f"  FN={summary['fn']:4d}  TN={summary['tn']:4d}")
    print(f"{'='*65}\n")


def run_evaluation(model, tokenizer, config: ExperimentConfig) -> dict:
    """Run evaluation across all configured datasets."""
    import time

    detector = Detector(model, tokenizer)
    model_short = config.model.name.split("/")[-1]
    all_summaries = {}

    for dataset_name in config.evaluation.datasets:
        print(f"\n{'#'*65}")
        print(f"# DATASET: {dataset_name.upper()}")
        print(f"{'#'*65}")

        samples = load_dataset(
            dataset_name,
            config.evaluation.n_samples,
            config.evaluation.max_context_chars,
        )

        t0 = time.time()
        results, summary = _run_dataset(detector, samples, config)
        elapsed = time.time() - t0

        summary["elapsed_seconds"] = elapsed
        summary["model"] = config.model.name
        summary["samples_per_second"] = len(results) / elapsed

        _print_results(summary)

        out_path = f"{config.output.dir}/eval_{dataset_name}_{model_short}.json"
        save_results({"summary": summary, "samples": [asdict(r) for r in results]}, out_path)

        all_summaries[dataset_name] = summary

    if len(all_summaries) > 1:
        print(f"\n{'#'*65}")
        print("# CROSS-DATASET SUMMARY")
        print(f"{'#'*65}")
        print(f"{'Dataset':<15} {'AUROC':>8} {'AUPRC':>8} {'TPR@5':>8} {'AURC':>8} {'E-AURC':>8} {'Hall%':>8}")
        print("-" * 65)
        for name, s in all_summaries.items():
            print(f"{name:<15} {s['auroc']:>8.4f} {s['auprc']:>8.4f} {s['tpr_at_5_fpr']:>8.4f} "
                  f"{s['aurc']:>8.4f} {s['e_aurc']:>8.4f} {s['hallucination_rate']:>7.1%}")

    return all_summaries
