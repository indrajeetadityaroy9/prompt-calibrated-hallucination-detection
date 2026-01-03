#!/usr/bin/env python3
"""
ICML-Grade Baseline Comparison Runner (2x H100).

Compares AG-SAR against:
1. Predictive Entropy (simple baseline)
2. EigenScore (spectral competitor - like SAPLMA)
3. SelfCheckGPT-Ngram (accuracy gold standard)

GPU Allocation:
- GPU 0: AG-SAR, Predictive Entropy, EigenScore (fast methods, shared model)
- GPU 1: SelfCheckGPT-Ngram (slow, generation-heavy)

Includes:
- Bootstrap confidence intervals (95%)
- Statistical significance testing
- Token-level analysis
- Comprehensive result logging
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp

warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class MethodResult:
    """Results for a single method."""
    name: str
    scores: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    extra: Dict = field(default_factory=dict)


def bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_scores)

    bootstrap_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(y_true[idx], y_scores[idx])
            bootstrap_scores.append(score)
        except ValueError:
            continue

    if not bootstrap_scores:
        return point_estimate, point_estimate, point_estimate

    bootstrap_scores = np.array(bootstrap_scores)
    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

    return point_estimate, ci_lower, ci_upper


def compute_metrics_with_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
) -> Dict:
    """Compute AUROC and AUPRC with confidence intervals."""
    auroc, auroc_lo, auroc_hi = bootstrap_ci(
        labels, scores, roc_auc_score, n_bootstrap
    )
    auprc, auprc_lo, auprc_hi = bootstrap_ci(
        labels, scores, average_precision_score, n_bootstrap
    )

    return {
        "auroc": auroc,
        "auroc_ci": (auroc_lo, auroc_hi),
        "auprc": auprc,
        "auprc_ci": (auprc_lo, auprc_hi),
    }


def mcnemar_test(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """
    McNemar's test for comparing two classifiers.

    Returns p-value (< 0.05 means significantly different).
    """
    # Binarize predictions at optimal threshold
    def optimal_threshold(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    thresh_a = optimal_threshold(labels, pred_a)
    thresh_b = optimal_threshold(labels, pred_b)

    binary_a = (pred_a >= thresh_a).astype(int)
    binary_b = (pred_b >= thresh_b).astype(int)

    # Contingency table
    # b01: A wrong, B correct
    # b10: A correct, B wrong
    b01 = np.sum((binary_a != labels) & (binary_b == labels))
    b10 = np.sum((binary_a == labels) & (binary_b != labels))

    if b01 + b10 == 0:
        return 1.0  # No difference

    # McNemar's chi-squared statistic with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_fast_methods(
    samples: List[Dict],
    config: dict,
    device: str = "cuda:0",
) -> Dict[str, MethodResult]:
    """
    Run AG-SAR, Predictive Entropy, and EigenScore on GPU 0.

    These share a model for efficiency.
    """
    from ag_sar import AGSAR, AGSARConfig
    from ag_sar import enable_h100_optimizations, get_optimal_dtype
    from baselines import PredictiveEntropyBaseline, EigenScoreBaseline

    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print(f"\n[GPU {device}] Loading model for fast methods...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize methods
    ag_config = AGSARConfig(
        semantic_layers=config["config"].get("semantic_layers", 4),
        power_iteration_steps=config["config"].get("power_iteration_steps", 3),
        residual_weight=config["config"].get("residual_weight", 0.5),
        enable_register_filter=config["config"].get("enable_register_filter", True),
        enable_spectral_roughness=config["config"].get("enable_spectral_roughness", True),
    )
    ag_sar = AGSAR(model, tokenizer, config=ag_config)
    entropy_baseline = PredictiveEntropyBaseline(model, tokenizer)
    eigenscore_baseline = EigenScoreBaseline(model, tokenizer)

    results = {
        "ag_sar": MethodResult(name="AG-SAR"),
        "entropy": MethodResult(name="Predictive Entropy"),
        "eigenscore": MethodResult(name="EigenScore"),
    }
    labels = []
    token_counts = []

    print(f"[GPU {device}] Running AG-SAR + Entropy + EigenScore on {len(samples)} samples...")

    for sample in tqdm(samples, desc=f"Fast methods ({device})", ncols=100):
        prompt, response, label = sample["prompt"], sample["response"], sample["label"]
        labels.append(label)

        # Count tokens
        full_ids = tokenizer.encode(prompt + response)
        token_counts.append(len(full_ids))

        # AG-SAR
        t0 = time.time()
        try:
            ag_result = ag_sar.compute_uncertainty(prompt, response, return_details=True)
            results["ag_sar"].scores.append(ag_result["score"])
            results["ag_sar"].times.append(time.time() - t0)
            results["ag_sar"].confidences.append(ag_result.get("confidence", 0.5))
        except Exception as e:
            results["ag_sar"].scores.append(0.5)
            results["ag_sar"].times.append(0)
            results["ag_sar"].confidences.append(0.5)

        # Predictive Entropy
        t0 = time.time()
        try:
            ent_result = entropy_baseline.compute_score(prompt, response)
            results["entropy"].scores.append(ent_result["score"])
            results["entropy"].times.append(time.time() - t0)
        except Exception:
            results["entropy"].scores.append(0.5)
            results["entropy"].times.append(0)

        # EigenScore
        t0 = time.time()
        try:
            eig_result = eigenscore_baseline.compute_score(prompt, response)
            results["eigenscore"].scores.append(eig_result["score"])
            results["eigenscore"].times.append(time.time() - t0)
        except Exception:
            results["eigenscore"].scores.append(0.5)
            results["eigenscore"].times.append(0)

    ag_sar.cleanup()
    del model
    torch.cuda.empty_cache()

    return {
        "results": results,
        "labels": labels,
        "token_counts": token_counts,
    }


def run_selfcheck(
    samples: List[Dict],
    config: dict,
    device: str = "cuda:1",
) -> Dict:
    """Run SelfCheckGPT-Ngram on GPU 1."""
    from ag_sar import enable_h100_optimizations, get_optimal_dtype
    from baselines import SelfCheckNgramBaseline

    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print(f"\n[GPU {device}] Loading model for SelfCheck...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selfcheck = SelfCheckNgramBaseline(
        model,
        tokenizer,
        num_samples=config.get("selfcheck_samples", 5),
        max_new_tokens=config.get("selfcheck_max_tokens", 100),
    )

    result = MethodResult(name="SelfCheck-Ngram")
    labels = []

    # Limit samples for SelfCheck (expensive)
    limit = min(config.get("selfcheck_limit", len(samples)), len(samples))
    samples = samples[:limit]

    print(f"[GPU {device}] Running SelfCheck on {len(samples)} samples...")

    for sample in tqdm(samples, desc=f"SelfCheck ({device})", ncols=100):
        prompt, response, label = sample["prompt"], sample["response"], sample["label"]
        labels.append(label)

        t0 = time.time()
        try:
            sc_result = selfcheck.compute_score(prompt, response)
            result.scores.append(sc_result["score"])
            result.times.append(time.time() - t0)
        except Exception:
            result.scores.append(0.5)
            result.times.append(0)

    del model
    torch.cuda.empty_cache()

    return {"result": result, "labels": labels}


def format_ci(value: float, ci: Tuple[float, float]) -> str:
    """Format value with CI: 0.75 [0.72, 0.78]"""
    return f"{value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"


def main():
    parser = argparse.ArgumentParser(description="ICML Baseline Comparison (2x H100)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap samples")
    args = parser.parse_args()

    config = load_config(args.config)
    n_bootstrap = args.bootstrap

    print("=" * 80)
    print("ICML-GRADE BASELINE COMPARISON (2x H100)")
    print("=" * 80)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Samples: {config['num_samples']}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"GPU 0: AG-SAR, Entropy, EigenScore")
    print(f"GPU 1: SelfCheck-Ngram")
    print("=" * 80)

    # Load dataset
    from loaders import load_halueval, load_ragtruth

    dataset_name = config["dataset"]
    num_samples = config["num_samples"]

    if dataset_name == "ragtruth":
        eval_samples = load_ragtruth(split="test", num_samples=num_samples, task="QA")
    elif "halueval" in dataset_name:
        variant = dataset_name.replace("halueval_", "")
        eval_samples = load_halueval(dataset_variant=variant, num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nLoaded {len(eval_samples)} evaluation samples")

    # Run in parallel on 2x H100
    start_time = time.time()
    print("\n" + "=" * 80)
    print("PARALLEL EXECUTION ON 2x H100")
    print("=" * 80)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fast_future = executor.submit(run_fast_methods, eval_samples, config, "cuda:0")
        selfcheck_future = executor.submit(run_selfcheck, eval_samples, config, "cuda:1")

        fast_data = fast_future.result()
        selfcheck_data = selfcheck_future.result()

    total_time = time.time() - start_time

    # Extract results
    labels = np.array(fast_data["labels"])
    token_counts = np.array(fast_data["token_counts"])
    fast_results = fast_data["results"]

    selfcheck_result = selfcheck_data["result"]
    selfcheck_labels = np.array(selfcheck_data["labels"])

    # Compute metrics with CIs
    print("\n" + "=" * 80)
    print("RESULTS WITH 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    all_metrics = {}

    # Fast methods
    for method_name, method_result in fast_results.items():
        scores = np.array(method_result.scores)
        times = np.array(method_result.times)

        min_len = min(len(labels), len(scores))
        method_labels = labels[:min_len]
        method_scores = scores[:min_len]

        metrics = compute_metrics_with_ci(method_labels, method_scores, n_bootstrap)
        metrics["mean_latency_ms"] = float(times.mean() * 1000)
        metrics["std_latency_ms"] = float(times.std() * 1000)
        metrics["n_samples"] = len(method_scores)

        all_metrics[method_name] = metrics

        print(f"\n[{method_result.name}]")
        print(f"  AUROC:         {format_ci(metrics['auroc'], metrics['auroc_ci'])}")
        print(f"  AUPRC:         {format_ci(metrics['auprc'], metrics['auprc_ci'])}")
        print(f"  Latency:       {metrics['mean_latency_ms']:.1f} ± {metrics['std_latency_ms']:.1f} ms")
        print(f"  Samples:       {metrics['n_samples']}")

    # SelfCheck
    if len(selfcheck_result.scores) > 10:
        sc_scores = np.array(selfcheck_result.scores)
        sc_times = np.array(selfcheck_result.times)

        metrics = compute_metrics_with_ci(selfcheck_labels, sc_scores, n_bootstrap)
        metrics["mean_latency_ms"] = float(sc_times.mean() * 1000)
        metrics["std_latency_ms"] = float(sc_times.std() * 1000)
        metrics["n_samples"] = len(sc_scores)

        all_metrics["selfcheck"] = metrics

        print(f"\n[{selfcheck_result.name}]")
        print(f"  AUROC:         {format_ci(metrics['auroc'], metrics['auroc_ci'])}")
        print(f"  AUPRC:         {format_ci(metrics['auprc'], metrics['auprc_ci'])}")
        print(f"  Latency:       {metrics['mean_latency_ms']:.1f} ± {metrics['std_latency_ms']:.1f} ms")
        print(f"  Samples:       {metrics['n_samples']}")

    # Statistical significance testing (AG-SAR vs others)
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE (McNemar's Test vs AG-SAR)")
    print("=" * 80)

    ag_scores = np.array(fast_results["ag_sar"].scores)

    for method_name, method_result in fast_results.items():
        if method_name == "ag_sar":
            continue
        other_scores = np.array(method_result.scores)
        min_len = min(len(ag_scores), len(other_scores), len(labels))
        p_value = mcnemar_test(labels[:min_len], ag_scores[:min_len], other_scores[:min_len])
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  AG-SAR vs {method_result.name}: p = {p_value:.4f} {sig}")

    if "selfcheck" in all_metrics:
        min_len = min(len(ag_scores), len(selfcheck_result.scores), len(selfcheck_labels))
        p_value = mcnemar_test(
            selfcheck_labels[:min_len],
            ag_scores[:min_len],
            np.array(selfcheck_result.scores)[:min_len]
        )
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  AG-SAR vs SelfCheck: p = {p_value:.4f} {sig}")

    # Token analysis
    print("\n" + "=" * 80)
    print("TOKEN LENGTH ANALYSIS")
    print("=" * 80)
    print(f"  Mean tokens:   {token_counts.mean():.1f}")
    print(f"  Median tokens: {np.median(token_counts):.1f}")
    print(f"  Range:         [{token_counts.min()}, {token_counts.max()}]")
    print(f"  Std:           {token_counts.std():.1f}")

    # Summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY TABLE (ICML Format)")
    print("=" * 80)

    selfcheck_latency = all_metrics.get("selfcheck", {}).get("mean_latency_ms", 1000)

    print(f"\n{'Method':<20} {'AUROC':<25} {'AUPRC':<25} {'Latency (ms)':<15} {'Speedup':<10}")
    print("-" * 95)

    for method_name in ["ag_sar", "entropy", "eigenscore", "selfcheck"]:
        if method_name in all_metrics:
            m = all_metrics[method_name]
            speedup = selfcheck_latency / m["mean_latency_ms"] if m["mean_latency_ms"] > 0 else 0
            auroc_str = format_ci(m["auroc"], m["auroc_ci"])
            auprc_str = format_ci(m["auprc"], m["auprc_ci"])
            display_name = {
                "ag_sar": "AG-SAR (Ours)",
                "entropy": "Pred. Entropy",
                "eigenscore": "EigenScore",
                "selfcheck": "SelfCheck-Ngram",
            }.get(method_name, method_name)
            print(f"{display_name:<20} {auroc_str:<25} {auprc_str:<25} {m['mean_latency_ms']:<15.1f} {speedup:.0f}x")

    print(f"\nTotal execution time: {total_time:.1f}s (parallel on 2x H100)")

    # Save results
    output_dir = Path(config.get("output_dir", "results/icml"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert tuples to lists for JSON
    json_metrics = {}
    for k, v in all_metrics.items():
        json_metrics[k] = {
            key: list(val) if isinstance(val, tuple) else val
            for key, val in v.items()
        }

    results_data = {
        "config": config,
        "metrics": json_metrics,
        "token_stats": {
            "mean": float(token_counts.mean()),
            "median": float(np.median(token_counts)),
            "std": float(token_counts.std()),
            "min": int(token_counts.min()),
            "max": int(token_counts.max()),
        },
        "execution_time_seconds": total_time,
        "n_bootstrap": n_bootstrap,
    }

    output_file = output_dir / "icml_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
