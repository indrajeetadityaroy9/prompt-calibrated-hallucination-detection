"""Ablation orchestration — leave-one-out signal ablation study."""

from __future__ import annotations

import copy

import numpy as np
from tqdm import tqdm

from ag_sar.detector import Detector
from ag_sar.aggregation.fusion import PromptAnchoredAggregator
from .answer_matching import max_f1_score, compute_adaptive_f1_threshold
from .metrics import compute_metrics

from .schema import ExperimentConfig
from .common import load_dataset, save_results


def _generate_baseline(
    detector: Detector,
    samples: list[dict],
    config: ExperimentConfig,
) -> list[dict]:
    """Generate baseline once per sample. Cache signals and prompt stats for re-aggregation."""
    signals = config.ablation.signals
    cached = []

    for sample in tqdm(samples, desc="Generating baseline"):
        result = detector.detect(
            question=sample["question"],
            context=sample["context"],
            max_new_tokens=config.evaluation.max_new_tokens,
        )

        response_signals = {
            sig: np.array([getattr(s, sig) for s in result.token_signals])
            for sig in signals
        }

        generated = result.generated_text.strip()
        f1 = max_f1_score(generated, sample["answers"])

        cached.append({
            "response_signals": response_signals,
            "prompt_stats": copy.deepcopy(detector.prompt_stats),
            "response_risk": result.response_risk,
            "f1": f1,
        })

    return cached


def _evaluate_condition(
    cached: list[dict],
    labels: list[int],
    disabled_signals: set[str],
) -> dict:
    """Re-aggregate from cached signals with optional signal disabling."""
    aggregator = PromptAnchoredAggregator()
    scores = []

    for c in cached:
        if not disabled_signals:
            scores.append(c["response_risk"])
        else:
            agg = aggregator.compute_risk(
                c["prompt_stats"],
                c["response_signals"],
                disabled_signals=disabled_signals,
            )
            scores.append(agg.risk)

    metrics = compute_metrics(scores, labels)
    return {"auroc": metrics.auroc, "auprc": metrics.auprc}


def run_ablation(model, tokenizer, config: ExperimentConfig) -> dict:
    """Run leave-one-out signal ablation study."""
    detector = Detector(model, tokenizer)
    model_short = config.model.name.split("/")[-1]
    signals_to_ablate = config.ablation.signals

    dataset_name = config.evaluation.datasets[0]
    samples = load_dataset(
        dataset_name,
        config.evaluation.n_samples,
        config.evaluation.max_context_chars,
    )

    # Phase 1: Generate baseline
    print(f"\n{'='*65}")
    print("Generating baseline (all signals, single pass)...")
    cached = _generate_baseline(detector, samples, config)

    # Compute labels via adaptive Otsu threshold
    f1_values = [c["f1"] for c in cached]
    threshold = compute_adaptive_f1_threshold(f1_values)
    labels = [int(f < threshold) for f in f1_values]

    # Phase 2: Evaluate each condition
    baseline = _evaluate_condition(cached, labels, disabled_signals=set())
    baseline_auroc = baseline["auroc"]
    print(f"Baseline AUROC: {baseline_auroc:.4f}")

    ablation_results = {"baseline": baseline}
    for signal in signals_to_ablate:
        print(f"\nAblating: {signal}...")
        result = _evaluate_condition(cached, labels, disabled_signals={signal})
        ablation_results[f"without_{signal}"] = result

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  AG-SAR Leave-One-Out Signal Ablation: {dataset_name.upper()}")
    print(f"{'='*65}")
    print(f"  {'Condition':<25} {'AUROC':>8} {'Delta':>8}")
    print(f"  {'-'*41}")
    print(f"  {'All signals (baseline)':<25} {baseline_auroc:>8.4f} {'---':>8}")

    for signal in signals_to_ablate:
        key = f"without_{signal}"
        auroc = ablation_results[key]["auroc"]
        delta = auroc - baseline_auroc
        print(f"  {'w/o ' + signal:<25} {auroc:>8.4f} {delta:>+8.4f}")

    print(f"{'='*65}\n")

    # Save results
    out_path = f"{config.output.dir}/ablation_{dataset_name}_{model_short}.json"
    save_results(ablation_results, out_path)

    return ablation_results
