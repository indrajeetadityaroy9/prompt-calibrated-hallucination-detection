"""
Baseline Comparison Runner.

Compares AG-SAR against:
1. LogProb (1 - mean token probability)
2. Predictive Entropy (Shannon entropy of softmax)
3. SelfCheckGPT-Ngram (SOTA approximation)

Reports AUROC, AUPRC, and latency for each method.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import warnings

warnings.filterwarnings("ignore")

import torch
import yaml
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import AGSAR, AGSARConfig
from ag_sar import enable_h100_optimizations, get_optimal_dtype

from loaders import load_halueval, load_ragtruth
from baselines import LogProbBaseline, PredictiveEntropyBaseline, SelfCheckNgramBaseline


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_comparison(config: dict) -> Dict:
    """Run all baselines and AG-SAR on the same dataset."""

    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print("=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Samples: {config['num_samples']}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=dtype,
        device_map=config["model"].get("device_map", "cuda:0"),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device
    print(f"Device: {device}, Dtype: {dtype}")

    # Load dataset
    dataset_name = config["dataset"]
    num_samples = config["num_samples"]

    if dataset_name == "ragtruth":
        eval_samples = load_ragtruth(split="test", num_samples=num_samples, task="QA")
    elif "halueval" in dataset_name:
        variant = dataset_name.replace("halueval_", "")
        eval_samples = load_halueval(dataset_variant=variant, num_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"\nEvaluation samples: {len(eval_samples)}")
    print(f"Hallucinated: {sum(1 for s in eval_samples if s['label'] == 1)}")
    print(f"Factual: {sum(1 for s in eval_samples if s['label'] == 0)}")

    # Initialize methods
    print("\nInitializing methods...")

    # 1. AG-SAR (our method)
    ag_config = AGSARConfig(
        semantic_layers=config["config"].get("semantic_layers", 4),
        power_iteration_steps=config["config"].get("power_iteration_steps", 3),
        residual_weight=config["config"].get("residual_weight", 0.5),
        enable_register_filter=config["config"].get("enable_register_filter", False),
        enable_spectral_roughness=config["config"].get("enable_spectral_roughness", False),
    )
    ag_sar = AGSAR(model, tokenizer, config=ag_config)

    # 2. LogProb baseline
    logprob_baseline = LogProbBaseline(model, tokenizer)

    # 3. Predictive Entropy baseline
    entropy_baseline = PredictiveEntropyBaseline(model, tokenizer)

    # 4. SelfCheckGPT-Ngram (SOTA approximation) - only on subset due to cost
    selfcheck_baseline = SelfCheckNgramBaseline(
        model, tokenizer,
        num_samples=config.get("selfcheck_samples", 4),
        max_new_tokens=config.get("selfcheck_max_tokens", 50),
    )

    # Results storage
    results = {
        "ag_sar": {"scores": [], "times": []},
        "logprob": {"scores": [], "times": []},
        "entropy": {"scores": [], "times": []},
        "selfcheck": {"scores": [], "times": []},
    }
    labels = []
    confidences = []

    # Determine SelfCheck sample size (expensive, so limit)
    selfcheck_limit = min(config.get("selfcheck_limit", 200), len(eval_samples))
    run_selfcheck = config.get("run_selfcheck", True)

    print(f"\nRunning evaluation...")
    print(f"  Total samples: {len(eval_samples)}")
    print(f"  SelfCheck: {'Yes (' + str(selfcheck_limit) + ' samples)' if run_selfcheck else 'No'}")
    print(f"  Methods: AG-SAR, LogProb, Entropy" + (", SelfCheck-Ngram" if run_selfcheck else ""))

    # Track running metrics for real-time display
    running_auroc = {"ag_sar": [], "logprob": [], "entropy": [], "selfcheck": []}

    for idx, sample in enumerate(tqdm(eval_samples, desc="Evaluating", ncols=100)):
        prompt = sample["prompt"]
        response = sample["response"]
        label = sample["label"]
        labels.append(label)

        # 1. AG-SAR
        t0 = time.time()
        try:
            ag_result = ag_sar.compute_uncertainty(prompt, response, return_details=True)
            results["ag_sar"]["scores"].append(ag_result["score"])
            results["ag_sar"]["times"].append(time.time() - t0)
            confidences.append(ag_result["confidence"])
        except Exception as e:
            results["ag_sar"]["scores"].append(0.5)
            results["ag_sar"]["times"].append(0)
            confidences.append(0.5)

        # 2. LogProb
        t0 = time.time()
        try:
            lp_result = logprob_baseline.compute_score(prompt, response)
            results["logprob"]["scores"].append(lp_result["score"])
            results["logprob"]["times"].append(time.time() - t0)
        except Exception:
            results["logprob"]["scores"].append(0.5)
            results["logprob"]["times"].append(0)

        # 3. Entropy
        t0 = time.time()
        try:
            ent_result = entropy_baseline.compute_score(prompt, response)
            results["entropy"]["scores"].append(ent_result["score"])
            results["entropy"]["times"].append(time.time() - t0)
        except Exception:
            results["entropy"]["scores"].append(0.5)
            results["entropy"]["times"].append(0)

        # 4. SelfCheck (only on subset)
        if run_selfcheck and idx < selfcheck_limit:
            t0 = time.time()
            try:
                sc_result = selfcheck_baseline.compute_score(prompt, response)
                results["selfcheck"]["scores"].append(sc_result["score"])
                results["selfcheck"]["times"].append(time.time() - t0)
            except Exception:
                results["selfcheck"]["scores"].append(0.5)
                results["selfcheck"]["times"].append(0)

        # Periodic progress report (every 50 samples)
        if (idx + 1) % 50 == 0:
            labels_so_far = np.array(labels)
            print(f"\n  [Progress {idx+1}/{len(eval_samples)}]")
            for method in ["ag_sar", "logprob", "entropy"]:
                scores = np.array(results[method]["scores"])
                if len(scores) > 10:
                    try:
                        auroc = roc_auc_score(labels_so_far, scores)
                        print(f"    {method}: AUROC={auroc:.4f}")
                    except:
                        pass
            if run_selfcheck and len(results["selfcheck"]["scores"]) > 10:
                try:
                    sc_labels = labels_so_far[:len(results["selfcheck"]["scores"])]
                    sc_auroc = roc_auc_score(sc_labels, results["selfcheck"]["scores"])
                    print(f"    selfcheck: AUROC={sc_auroc:.4f}")
                except:
                    pass

    # Compute metrics
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    labels_arr = np.array(labels)
    confidences_arr = np.array(confidences)
    confidence_threshold = config.get("confidence_threshold", 0.7)

    metrics = {}

    for method_name, method_data in results.items():
        scores = np.array(method_data["scores"])
        times = np.array(method_data["times"])

        if len(scores) == 0:
            continue

        # Use appropriate labels (selfcheck has fewer samples)
        if method_name == "selfcheck":
            method_labels = labels_arr[:len(scores)]
            method_confidences = confidences_arr[:len(scores)]
        else:
            method_labels = labels_arr
            method_confidences = confidences_arr

        # Full dataset metrics
        try:
            auroc = roc_auc_score(method_labels, scores)
            auprc = average_precision_score(method_labels, scores)
        except Exception:
            auroc, auprc = 0.5, 0.5

        # Confident subset metrics
        confident_mask = method_confidences > confidence_threshold
        n_confident = confident_mask.sum()

        if n_confident > 10:
            try:
                auroc_conf = roc_auc_score(
                    method_labels[confident_mask],
                    scores[confident_mask]
                )
            except Exception:
                auroc_conf = 0.5
        else:
            auroc_conf = None

        # Latency
        mean_time = times.mean() * 1000  # Convert to ms
        total_time = times.sum()

        metrics[method_name] = {
            "auroc_full": auroc,
            "auprc_full": auprc,
            "auroc_confident": auroc_conf,
            "n_confident": int(n_confident) if method_name != "selfcheck" else None,
            "mean_latency_ms": mean_time,
            "total_time_s": total_time,
            "n_samples": len(scores),
        }

        print(f"\n[{method_name.upper()}]")
        print(f"  Samples:        {len(scores)}")
        print(f"  AUROC (full):   {auroc:.4f}")
        print(f"  AUPRC (full):   {auprc:.4f}")
        if auroc_conf is not None:
            print(f"  AUROC (conf):   {auroc_conf:.4f}")
        print(f"  Mean Latency:   {mean_time:.1f} ms")
        print(f"  Total Time:     {total_time:.1f} s")

    # Comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    # Sort by AUROC
    sorted_methods = sorted(
        metrics.items(),
        key=lambda x: x[1]["auroc_full"],
        reverse=True
    )

    print(f"\n{'Method':<15} {'AUROC':<10} {'Latency (ms)':<15} {'Speedup vs SelfCheck':<20}")
    print("-" * 60)

    selfcheck_time = metrics.get("selfcheck", {}).get("mean_latency_ms", 1)
    for method_name, m in sorted_methods:
        speedup = selfcheck_time / m["mean_latency_ms"] if m["mean_latency_ms"] > 0 else 0
        print(f"{method_name:<15} {m['auroc_full']:.4f}     {m['mean_latency_ms']:<15.1f} {speedup:.1f}x")

    # Save results
    output_dir = Path(config.get("output_dir", "results/baseline_comparison"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "comparison_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Cleanup
    ag_sar.cleanup()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_comparison(config)


if __name__ == "__main__":
    main()
