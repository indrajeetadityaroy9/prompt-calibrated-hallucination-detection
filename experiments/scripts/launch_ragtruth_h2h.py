#!/usr/bin/env python3
"""
Parallel Launcher for Head-to-Head RAGTruth Benchmark.

Runs two AG-SAR variants simultaneously on 2x H100 GPUs:
- GPU 0: Baseline (top1_projection)
- GPU 1: JEPA (centroid_variance)

This is the "Final Boss" benchmark - real RAG failures from production LLMs.

Usage:
    python scripts/prepare_ragtruth.py  # Run first to prepare data
    python scripts/launch_ragtruth_h2h.py
    python scripts/launch_ragtruth_h2h.py --config experiments/configs/benchmark_ragtruth.yaml
"""

import argparse
import json
import os
import sys
import time
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# Pre-flight installation check
from experiments.utils.preflight import check_installation, get_project_root
check_installation()

import yaml
import torch

PROJECT_ROOT = get_project_root()

# Set HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN


@dataclass
class RAGTruthSample:
    """A single RAGTruth evaluation sample."""
    id: str
    prompt: str
    response: str
    label: int
    task_type: str = ""
    model: str = ""


def load_ragtruth_jsonl(data_file: Path, num_samples: int = None) -> List[RAGTruthSample]:
    """Load RAGTruth samples from JSONL file."""
    samples = []
    with open(data_file) as f:
        for line in f:
            row = json.loads(line.strip())
            samples.append(RAGTruthSample(
                id=row["id"],
                prompt=row["prompt"],
                response=row["response"],
                label=row["label"],
                task_type=row.get("task_type", ""),
                model=row.get("model", ""),
            ))
            if num_samples and len(samples) >= num_samples:
                break
    return samples


def run_worker(method_config: Dict, base_config: Dict, gpu_id: int, output_dir: Path):
    """
    Worker function to run one AG-SAR method on one GPU.

    Args:
        method_config: Configuration for this specific method
        base_config: Base experiment configuration
        gpu_id: GPU to run on
        output_dir: Directory for results
    """
    method_name = method_config["name"]
    print(f"[GPU {gpu_id}] Launching worker for {method_name}...")

    # Import inside worker to avoid CUDA context issues
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ag_sar import AGSAR, AGSARConfig

    # Set CUDA device
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Create output directory for this method
    method_output_dir = output_dir / method_name
    method_output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_config = base_config["model"]
    print(f"[GPU {gpu_id}] Loading model: {model_config['name']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        token=HF_TOKEN,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(model_config.get("dtype", "bfloat16"), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=model_config.get("attn_implementation", "eager"),
        trust_remote_code=model_config.get("trust_remote_code", True),
    )
    model.eval()
    print(f"[GPU {gpu_id}] Model loaded on {device}")

    # Load dataset from JSONL
    ds_config = base_config["dataset"]
    data_file = PROJECT_ROOT / ds_config["data_file"]
    if not data_file.exists():
        print(f"[GPU {gpu_id}] ERROR: Data file not found: {data_file}")
        print(f"[GPU {gpu_id}] Run: python scripts/prepare_ragtruth.py")
        return None

    samples = load_ragtruth_jsonl(data_file, ds_config.get("num_samples"))
    print(f"[GPU {gpu_id}] Dataset loaded: {len(samples)} samples")

    # Create AG-SAR engine with method-specific config
    agsar_config_dict = method_config.get("agsar", {})
    agsar_config = AGSARConfig(
        semantic_layers=agsar_config_dict.get("semantic_layers", 4),
        power_iteration_steps=agsar_config_dict.get("power_iteration_steps", 3),
        residual_weight=agsar_config_dict.get("residual_weight", 0.5),
        enable_unified_gating=agsar_config_dict.get("enable_unified_gating", True),
        stability_sensitivity=agsar_config_dict.get("stability_sensitivity", 1.0),
        parametric_weight=agsar_config_dict.get("parametric_weight", 0.5),
        enable_semantic_dispersion=agsar_config_dict.get("enable_semantic_dispersion", True),
        dispersion_k=agsar_config_dict.get("dispersion_k", 5),
        dispersion_sensitivity=agsar_config_dict.get("dispersion_sensitivity", 1.0),
        dispersion_method=agsar_config_dict.get("dispersion_method", "top1_projection"),
        nucleus_top_p=agsar_config_dict.get("nucleus_top_p", 0.95),
        aggregation_method=agsar_config_dict.get("aggregation_method", "mean"),
        # Layer Drift (v11.0)
        enable_layer_drift=agsar_config_dict.get("enable_layer_drift", False),
        drift_layer_ratio=agsar_config_dict.get("drift_layer_ratio", 0.5),
        drift_sensitivity=agsar_config_dict.get("drift_sensitivity", 1.0),
    )

    engine = AGSAR(model, tokenizer, agsar_config)
    print(f"[GPU {gpu_id}] AG-SAR engine created:")
    print(f"[GPU {gpu_id}]   dispersion_method={agsar_config.dispersion_method}")
    print(f"[GPU {gpu_id}]   aggregation_method={agsar_config.aggregation_method}")
    print(f"[GPU {gpu_id}]   enable_layer_drift={agsar_config.enable_layer_drift}")
    if agsar_config.enable_layer_drift:
        print(f"[GPU {gpu_id}]   drift_layer_ratio={agsar_config.drift_layer_ratio}")
        print(f"[GPU {gpu_id}]   drift_sensitivity={agsar_config.drift_sensitivity}")

    # Run evaluation
    results = []
    scores = []
    labels = []
    latencies = []

    from tqdm import tqdm

    print(f"[GPU {gpu_id}] Starting evaluation...")
    for sample in tqdm(samples, desc=f"[GPU {gpu_id}] {method_name}", position=gpu_id):
        try:
            result = engine.compute_uncertainty(
                sample.prompt,
                sample.response,
                return_details=True,
            )

            uncertainty = result["score"]
            latency = result.get("latency_ms", 0)

            results.append({
                "id": sample.id,
                "label": sample.label,
                "uncertainty": uncertainty,
                "authority": result.get("authority", 0),
                "latency_ms": latency,
                "task_type": sample.task_type,
                "source_model": sample.model,
            })

            scores.append(uncertainty)
            labels.append(sample.label)
            latencies.append(latency)

        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing sample {sample.id}: {e}")
            continue

    # Cleanup
    engine.cleanup()

    # Compute metrics
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, f1_score

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # Check for valid labels
    n_pos = (labels_arr == 1).sum()
    n_neg = (labels_arr == 0).sum()
    print(f"[GPU {gpu_id}] Label distribution: {n_pos} hallucinated, {n_neg} faithful")

    # AUROC
    auroc = roc_auc_score(labels_arr, scores_arr) if len(np.unique(labels_arr)) > 1 else 0.0

    # AUPRC (critical for imbalanced data)
    auprc = average_precision_score(labels_arr, scores_arr) if len(np.unique(labels_arr)) > 1 else 0.0

    # TPR @ 5% FPR
    def compute_tpr_at_fpr(y_true, y_scores, fpr_target=0.05):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(fpr - fpr_target))
        return tpr[idx]

    tpr_at_5fpr = compute_tpr_at_fpr(labels_arr, scores_arr) if len(np.unique(labels_arr)) > 1 else 0.0

    # AURC (Area Under Risk-Coverage)
    def compute_aurc(y_true, y_scores):
        sorted_idx = np.argsort(y_scores)  # Sort by uncertainty ascending (most confident first)
        sorted_labels = y_true[sorted_idx]
        n = len(sorted_labels)
        risks = []
        for i in range(1, n + 1):
            risk = sorted_labels[:i].sum() / i  # Fraction of hallucinations in top-i
            risks.append(risk)
        coverages = np.arange(1, n + 1) / n
        return np.trapz(risks, coverages)

    aurc = compute_aurc(labels_arr, scores_arr)

    # Max F1
    thresholds = np.percentile(scores_arr, np.arange(0, 101, 5))
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (scores_arr >= thresh).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0)
        best_f1 = max(best_f1, f1)

    # Average latency
    avg_latency = np.mean(latencies) if latencies else 0.0

    # Avg uncertainty for positive vs negative
    pos_mask = labels_arr == 1  # Hallucinated
    neg_mask = labels_arr == 0  # Faithful
    avg_unc_hallucinated = scores_arr[pos_mask].mean() if pos_mask.sum() > 0 else 0.0
    avg_unc_faithful = scores_arr[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

    metrics = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tpr_at_5fpr": float(tpr_at_5fpr),
        "aurc": float(aurc),
        "max_f1": float(best_f1),
        "avg_latency_ms": float(avg_latency),
        "avg_unc_hallucinated": float(avg_unc_hallucinated),
        "avg_unc_faithful": float(avg_unc_faithful),
        "n_samples": len(results),
        "n_hallucinated": int(n_pos),
        "n_faithful": int(n_neg),
    }

    # Save results
    output_file = method_output_dir / "results.json"
    output_data = {
        "method": method_name,
        "config": {
            "model": model_config["name"],
            "dataset": "ragtruth",
            "agsar": agsar_config_dict,
        },
        "metrics": metrics,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[GPU {gpu_id}] {method_name} Complete!")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  AUPRC: {auprc:.4f} (KEY METRIC)")
    print(f"  TPR@5%FPR: {tpr_at_5fpr:.4f}")
    print(f"  AURC: {aurc:.4f}")
    print(f"  Avg Latency: {avg_latency:.2f}ms")
    print(f"  Results saved to: {output_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Launch Head-to-Head RAGTruth Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/benchmark_ragtruth.yaml",
        help="Path to benchmark config",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Override data file path",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override data file if specified
    if args.data_file:
        config["dataset"]["data_file"] = args.data_file

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / config["output"]["output_dir"] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RAGTruth Head-to-Head Benchmark (Final Boss)")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Data File: {config['dataset']['data_file']}")
    print(f"Samples: {config['dataset'].get('num_samples', 'all')}")
    print(f"Output: {output_dir}")
    print()

    # Check data file exists
    data_file = PROJECT_ROOT / config["dataset"]["data_file"]
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        print("Run: python scripts/prepare_ragtruth.py")
        sys.exit(1)

    # Check GPU availability
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    print()

    # Save config copy
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Launch parallel workers
    methods = config.get("parallel_methods", [])
    if not methods:
        print("Error: No parallel_methods defined in config")
        sys.exit(1)

    print(f"Launching {len(methods)} parallel workers...")
    for method in methods:
        print(f"  - {method['name']} on GPU {method['gpu_id']}")
    print()

    processes = []
    for method in methods:
        gpu_id = method.get("gpu_id", 0)
        p = multiprocessing.Process(
            target=run_worker,
            args=(method, config, gpu_id, output_dir),
        )
        p.start()
        processes.append(p)
        # Stagger slightly to avoid race conditions
        time.sleep(3)

    # Wait for all workers
    for p in processes:
        p.join()

    print()
    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print()
    print("Run analysis script:")
    print(f"  python scripts/print_h2h_table.py --results-dir {output_dir}")


if __name__ == "__main__":
    # Use spawn to avoid CUDA fork issues
    multiprocessing.set_start_method("spawn", force=True)
    main()
