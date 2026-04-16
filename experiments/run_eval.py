import time
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.config import SIGNAL_NAMES
from src.detector import Detector
from src.numerics import otsu

from experiments.answer_matching import max_f1_score
from experiments.common import PROMPT_TEMPLATE, load_samples, save_results
from experiments.metrics import bootstrap_auroc_ci, compute_metrics
from experiments.schema import ExperimentConfig


@dataclass
class SampleResult:
    question: str
    generated_answer: str
    ground_truths: list[str]
    f1: float
    is_hallucination: bool
    response_risk: float
    mean_rho: float
    mean_phi: float
    mean_spf: float
    mean_mlp: float
    mean_ent: float
    n_tokens: int
    is_flagged: bool


def _run_dataset(detector: Detector, dataset_name: str, samples: list[dict], config: ExperimentConfig) -> tuple[list[SampleResult], dict]:
    results: list[SampleResult] = []
    print_interval = config.evaluation.n_samples // 4

    for i, sample in enumerate(tqdm(samples, desc=f"Eval ({dataset_name})")):
        prompt = PROMPT_TEMPLATE.format(context=sample["context"], question=sample["question"])
        result = detector.detect(prompt=prompt, max_new_tokens=config.evaluation.max_new_tokens)
        generated = result.generated_text.strip()
        f1 = max_f1_score(generated, sample["answers"])

        means = {f"mean_{s}": float(np.mean([getattr(t, s) for t in result.token_signals])) for s in SIGNAL_NAMES}

        results.append(SampleResult(question=sample["question"], generated_answer=generated, ground_truths=sample["answers"], f1=f1, is_hallucination=False, response_risk=result.response_risk, **means, n_tokens=result.num_tokens, is_flagged=result.is_flagged))

        if (i + 1) % print_interval == 0:
            print(f"  [{i+1}/{len(samples)}] avg_risk={np.mean([r.response_risk for r in results]):.3f}")

    adaptive_threshold = otsu([r.f1 for r in results])[0]
    print(f"  Adaptive F1 threshold: {adaptive_threshold:.3f}")
    for r in results:
        r.is_hallucination = r.f1 < adaptive_threshold

    labels = [int(r.is_hallucination) for r in results]
    scores = [r.response_risk for r in results]
    metrics = compute_metrics(scores, labels)

    sig_keys = [f"mean_{s}" for s in SIGNAL_NAMES] + ["response_risk"]
    signal_aurocs = {k: float(roc_auc_score(labels, [getattr(r, k) for r in results])) for k in sig_keys}

    ci_low, ci_high = bootstrap_auroc_ci(scores, labels, seed=config.evaluation.seed)
    n_pos = sum(labels)

    summary = {
        "dataset": dataset_name, "f1_threshold": adaptive_threshold,
        "n_samples": len(results), "n_hallucinations": n_pos, "n_correct": len(labels) - n_pos,
        "hallucination_rate": n_pos / len(results),
        "auroc": metrics.auroc, "auroc_ci_95": [ci_low, ci_high], "auprc": metrics.auprc,
        "fpr_at_95_tpr": metrics.fpr_at_95_tpr, "f1_optimal": metrics.f1_optimal,
        "optimal_threshold": metrics.optimal_threshold,
        "aurc": metrics.aurc, "e_aurc": metrics.e_aurc,
        "ece": metrics.expected_calibration_error, "brier": metrics.brier_score,
        "signal_aurocs": signal_aurocs,
    }
    return results, summary


def _print_results(s: dict):
    ci = s['auroc_ci_95']
    print(f"\n{'='*65}\n  Evaluation Results: {s['dataset'].upper()}\n{'='*65}")
    print(f"  Samples:           {s['n_samples']}")
    print(f"  Hallucinations:    {s['n_hallucinations']} ({s['hallucination_rate']:.1%})")
    print(f"\n  ---- Detection ----")
    print(f"  AUROC:             {s['auroc']:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
    print(f"  AUPRC:             {s['auprc']:.4f}")
    print(f"  FPR@95%TPR:        {s['fpr_at_95_tpr']:.4f}")
    print(f"  F1 (Youden):       {s['f1_optimal']:.4f}  (t={s['optimal_threshold']:.3f})")
    print(f"\n  ---- Selective Prediction ----")
    print(f"  AURC:              {s['aurc']:.4f}")
    print(f"  E-AURC:            {s['e_aurc']:.4f}")
    print(f"\n  ---- Calibration ----")
    print(f"  ECE:               {s['ece']:.4f}")
    print(f"  Brier:             {s['brier']:.4f}")
    print(f"\n  ---- Per-Signal AUROC ----")
    for sig, auroc in s['signal_aurocs'].items():
        print(f"  {sig:20s} {auroc:.4f}")
    print(f"{'='*65}\n")


def run_evaluation(model, tokenizer, config: ExperimentConfig) -> dict:
    detector = Detector(model, tokenizer)
    model_short = config.model.name.split("/")[-1]
    all_summaries = {}

    for dataset_name in config.evaluation.datasets:
        print(f"\n{'#'*65}\n# DATASET: {dataset_name.upper()}\n{'#'*65}")
        samples = load_samples(dataset_name, config.evaluation.n_samples, config.evaluation.max_context_chars)

        t0 = time.time()
        results, summary = _run_dataset(detector, dataset_name, samples, config)
        elapsed = time.time() - t0

        summary.update({"elapsed_seconds": elapsed, "model": config.model.name, "samples_per_second": len(results) / elapsed})
        _print_results(summary)

        save_results({"summary": summary, "samples": [asdict(r) for r in results]}, f"{config.output_dir}/eval_{dataset_name}_{model_short}.json")
        all_summaries[dataset_name] = summary

    if len(all_summaries) > 1:
        print(f"\n{'#'*65}\n# CROSS-DATASET SUMMARY\n{'#'*65}")
        print(f"{'Dataset':<15} {'AUROC':>8} {'AUPRC':>8} {'FPR@95':>8} {'AURC':>8} {'E-AURC':>8} {'Hall%':>8}")
        print("-" * 65)
        for name, s in all_summaries.items():
            print(f"{name:<15} {s['auroc']:>8.4f} {s['auprc']:>8.4f} {s['fpr_at_95_tpr']:>8.4f} {s['aurc']:>8.4f} {s['e_aurc']:>8.4f} {s['hallucination_rate']:>7.1%}")

    return all_summaries
