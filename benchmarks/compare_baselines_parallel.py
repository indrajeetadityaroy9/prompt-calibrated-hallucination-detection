"""
Parallel Baseline Comparison Runner (2x H100).

Utilizes both GPUs:
- GPU 0: AG-SAR, LogProb, Entropy (fast methods)
- GPU 1: SelfCheckGPT (slow, generation-heavy)

Uses multiprocessing to run both in parallel.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings("ignore")

import torch
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_fast_methods(samples: List[Dict], config: dict, device: str = "cuda:0") -> Dict:
    """Run AG-SAR, LogProb, and Entropy on GPU 0."""
    from ag_sar import AGSAR, AGSARConfig
    from ag_sar import enable_h100_optimizations, get_optimal_dtype
    from baselines import LogProbBaseline, PredictiveEntropyBaseline

    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print(f"[GPU {device}] Loading model for fast methods...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize methods
    ag_config = AGSARConfig(
        semantic_layers=config["config"].get("semantic_layers", 4),
        power_iteration_steps=config["config"].get("power_iteration_steps", 3),
        residual_weight=config["config"].get("residual_weight", 0.5),
        enable_register_filter=config["config"].get("enable_register_filter", False),
        enable_spectral_roughness=config["config"].get("enable_spectral_roughness", False),
    )
    ag_sar = AGSAR(model, tokenizer, config=ag_config)
    logprob_baseline = LogProbBaseline(model, tokenizer)
    entropy_baseline = PredictiveEntropyBaseline(model, tokenizer)

    results = {
        "ag_sar": {"scores": [], "times": [], "confidences": []},
        "logprob": {"scores": [], "times": []},
        "entropy": {"scores": [], "times": []},
    }
    labels = []

    print(f"[GPU {device}] Running AG-SAR + LogProb + Entropy on {len(samples)} samples...")
    for sample in tqdm(samples, desc=f"Fast methods ({device})", ncols=80):
        prompt, response, label = sample["prompt"], sample["response"], sample["label"]
        labels.append(label)

        # AG-SAR
        t0 = time.time()
        try:
            ag_result = ag_sar.compute_uncertainty(prompt, response, return_details=True)
            results["ag_sar"]["scores"].append(ag_result["score"])
            results["ag_sar"]["times"].append(time.time() - t0)
            results["ag_sar"]["confidences"].append(ag_result["confidence"])
        except:
            results["ag_sar"]["scores"].append(0.5)
            results["ag_sar"]["times"].append(0)
            results["ag_sar"]["confidences"].append(0.5)

        # LogProb
        t0 = time.time()
        try:
            lp_result = logprob_baseline.compute_score(prompt, response)
            results["logprob"]["scores"].append(lp_result["score"])
            results["logprob"]["times"].append(time.time() - t0)
        except:
            results["logprob"]["scores"].append(0.5)
            results["logprob"]["times"].append(0)

        # Entropy
        t0 = time.time()
        try:
            ent_result = entropy_baseline.compute_score(prompt, response)
            results["entropy"]["scores"].append(ent_result["score"])
            results["entropy"]["times"].append(time.time() - t0)
        except:
            results["entropy"]["scores"].append(0.5)
            results["entropy"]["times"].append(0)

    ag_sar.cleanup()
    del model
    torch.cuda.empty_cache()

    results["labels"] = labels
    return results


def run_selfcheck(samples: List[Dict], config: dict, device: str = "cuda:1") -> Dict:
    """Run SelfCheckGPT on GPU 1."""
    from ag_sar import enable_h100_optimizations, get_optimal_dtype
    from baselines import SelfCheckNgramBaseline

    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print(f"[GPU {device}] Loading model for SelfCheck...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    selfcheck = SelfCheckNgramBaseline(
        model, tokenizer,
        num_samples=config.get("selfcheck_samples", 5),
        max_new_tokens=config.get("selfcheck_max_tokens", 100),
    )

    results = {"scores": [], "times": []}
    labels = []

    limit = min(config.get("selfcheck_limit", len(samples)), len(samples))
    samples = samples[:limit]

    print(f"[GPU {device}] Running SelfCheck on {len(samples)} samples...")
    for sample in tqdm(samples, desc=f"SelfCheck ({device})", ncols=80):
        prompt, response, label = sample["prompt"], sample["response"], sample["label"]
        labels.append(label)

        t0 = time.time()
        try:
            sc_result = selfcheck.compute_score(prompt, response)
            results["scores"].append(sc_result["score"])
            results["times"].append(time.time() - t0)
        except Exception as e:
            results["scores"].append(0.5)
            results["times"].append(0)

    del model
    torch.cuda.empty_cache()

    results["labels"] = labels
    return results


def main():
    parser = argparse.ArgumentParser(description="Parallel Baseline Comparison (2x GPU)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 70)
    print("PARALLEL BASELINE COMPARISON (2x H100)")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Samples: {config['num_samples']}")
    print(f"GPU 0: AG-SAR, LogProb, Entropy")
    print(f"GPU 1: SelfCheck-Ngram")

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

    print(f"\nLoaded {len(eval_samples)} samples")

    # Run both in parallel using threads (separate CUDA contexts)
    start_time = time.time()

    print("\n[Starting parallel execution on 2x H100]")

    with ThreadPoolExecutor(max_workers=2) as executor:
        fast_future = executor.submit(run_fast_methods, eval_samples, config, "cuda:0")
        selfcheck_future = executor.submit(run_selfcheck, eval_samples, config, "cuda:1")

        fast_results = fast_future.result()
        selfcheck_results = selfcheck_future.result()

    total_time = time.time() - start_time

    # Compute final metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    labels = np.array(fast_results["labels"])
    confidences = np.array(fast_results["ag_sar"]["confidences"])
    confidence_threshold = config.get("confidence_threshold", 0.7)

    print(f"\nTotal samples evaluated: {len(labels)}")

    metrics = {}

    # Fast methods
    for method in ["ag_sar", "logprob", "entropy"]:
        scores = np.array(fast_results[method]["scores"])
        times = np.array(fast_results[method]["times"])

        # Ensure same length
        min_len = min(len(labels), len(scores))
        method_labels = labels[:min_len]
        method_scores = scores[:min_len]
        method_confidences = confidences[:min_len]

        auroc = roc_auc_score(method_labels, method_scores)
        auprc = average_precision_score(method_labels, method_scores)

        confident_mask = method_confidences > confidence_threshold
        auroc_conf = roc_auc_score(method_labels[confident_mask], method_scores[confident_mask]) if confident_mask.sum() > 10 else None

        metrics[method] = {
            "auroc_full": float(auroc),
            "auprc_full": float(auprc),
            "auroc_confident": float(auroc_conf) if auroc_conf else None,
            "mean_latency_ms": float(times.mean() * 1000),
            "n_samples": len(method_scores),
        }

        print(f"\n[{method.upper()}]")
        print(f"  AUROC (full):     {auroc:.4f}")
        print(f"  AUPRC (full):     {auprc:.4f}")
        if auroc_conf:
            print(f"  AUROC (conf):     {auroc_conf:.4f}")
        print(f"  Mean Latency:     {times.mean()*1000:.1f} ms")

    # SelfCheck
    sc_scores = np.array(selfcheck_results["scores"])
    sc_times = np.array(selfcheck_results["times"])
    sc_labels = np.array(selfcheck_results["labels"])

    if len(sc_scores) > 10:
        sc_auroc = roc_auc_score(sc_labels, sc_scores)
        sc_auprc = average_precision_score(sc_labels, sc_scores)

        metrics["selfcheck"] = {
            "auroc_full": sc_auroc,
            "auprc_full": sc_auprc,
            "mean_latency_ms": sc_times.mean() * 1000,
            "n_samples": len(sc_scores),
        }

        print(f"\n[SELFCHECK]")
        print(f"  AUROC (full):     {sc_auroc:.4f}")
        print(f"  AUPRC (full):     {sc_auprc:.4f}")
        print(f"  Mean Latency:     {sc_times.mean()*1000:.1f} ms")
        print(f"  Samples:          {len(sc_scores)}")

    # Summary table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    selfcheck_time = metrics.get("selfcheck", {}).get("mean_latency_ms", 1000)

    print(f"\n{'Method':<15} {'AUROC':<10} {'Latency (ms)':<15} {'Speedup':<10}")
    print("-" * 50)

    for method in ["ag_sar", "logprob", "entropy", "selfcheck"]:
        if method in metrics:
            m = metrics[method]
            speedup = selfcheck_time / m["mean_latency_ms"] if m["mean_latency_ms"] > 0 else 0
            print(f"{method:<15} {m['auroc_full']:.4f}     {m['mean_latency_ms']:<15.1f} {speedup:.1f}x")

    print(f"\nTotal time: {total_time:.1f}s (parallel execution)")

    # Save results
    output_dir = Path(config.get("output_dir", "results/baseline_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
