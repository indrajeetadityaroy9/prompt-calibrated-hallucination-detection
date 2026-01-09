#!/usr/bin/env python3
"""
AG-SAR Comprehensive Evaluation Sweep.

Runs AG-SAR on ALL datasets and task types to establish SOTA performance.

Datasets:
- HaluEval: qa, summarization, dialogue
- RAGTruth: qa, summarization, data2txt
- TruthfulQA
- FAVA

Usage:
    export HF_TOKEN=your_token
    python -m experiments.scripts.run_agsar_sweep
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Ensure ag_sar is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.preflight import check_installation
check_installation()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from ag_sar import AGSAR, AGSARConfig
from experiments.data import (
    HaluEvalDataset,
    RAGTruthDataset,
    TruthfulQADataset,
    FAVADataset,
)
from experiments.evaluation.metrics import MetricsCalculator


# Configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B"
NUM_SAMPLES = 500
SEED = 42
OUTPUT_DIR = Path("results/agsar_comprehensive_sweep")


DATASETS = [
    # HaluEval variants
    {"name": "halueval_qa", "class": HaluEvalDataset, "kwargs": {"variant": "qa"}},
    {"name": "halueval_summarization", "class": HaluEvalDataset, "kwargs": {"variant": "summarization"}},
    {"name": "halueval_dialogue", "class": HaluEvalDataset, "kwargs": {"variant": "dialogue"}},
    # RAGTruth variants
    {"name": "ragtruth_qa", "class": RAGTruthDataset, "kwargs": {"task_type": "QA"}},
    {"name": "ragtruth_summarization", "class": RAGTruthDataset, "kwargs": {"task_type": "summarization"}},
    {"name": "ragtruth_data2txt", "class": RAGTruthDataset, "kwargs": {"task_type": "data2txt"}},
    # Other datasets
    {"name": "truthfulqa", "class": TruthfulQADataset, "kwargs": {}},
    {"name": "fava", "class": FAVADataset, "kwargs": {}},
]


def load_model():
    """Load Llama-3.1-8B model."""
    print(f"Loading model: {MODEL_NAME}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Device: {next(model.parameters()).device}")
    print(f"  GQA: {model.config.num_attention_heads}:{model.config.num_key_value_heads}")

    return model, tokenizer


def evaluate_dataset(agsar, dataset_config, metrics_calc):
    """Evaluate AG-SAR on a single dataset."""
    name = dataset_config["name"]
    ds_class = dataset_config["class"]
    kwargs = dataset_config["kwargs"]

    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # Load dataset
    dataset = ds_class(num_samples=NUM_SAMPLES, seed=SEED, **kwargs)
    dataset.load()

    stats = dataset.get_statistics()
    print(f"Samples: {stats['total_samples']} (Hall: {stats['hallucinated']}, "
          f"Fact: {stats['factual']}, Rate: {stats['hallucination_rate']:.1%})")

    # Evaluate
    scores = []
    labels = []
    latencies = []

    for sample in tqdm(dataset, desc=f"  AG-SAR on {name}"):
        try:
            start = time.perf_counter()

            # Calibrate on prompt
            agsar.calibrate_on_prompt(sample.prompt)

            # Compute uncertainty
            uncertainty = agsar.compute_uncertainty(sample.prompt, sample.response)

            elapsed_ms = (time.perf_counter() - start) * 1000

            scores.append(uncertainty)
            labels.append(sample.label)
            latencies.append(elapsed_ms)

        except Exception as e:
            scores.append(float("nan"))
            labels.append(sample.label)
            latencies.append(0.0)

    # Compute metrics
    metrics, ci_bounds = metrics_calc.compute_all(
        labels=labels,
        scores=scores,
        metric_names=["auroc", "auprc", "f1", "accuracy", "ece", "brier", "spearman"],
    )

    mean_latency = sum(latencies) / len(latencies) if latencies else 0

    # Print results
    print(f"\nResults for {name}:")
    for metric, value in metrics.items():
        ci = ci_bounds.get(metric, (0, 0))
        print(f"  {metric.upper()}: {value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"  Mean Latency: {mean_latency:.1f} ms")

    return {
        "dataset": name,
        "metrics": metrics,
        "ci_bounds": {k: list(v) for k, v in ci_bounds.items()},
        "mean_latency_ms": mean_latency,
        "num_samples": len(scores),
        "statistics": stats,
    }


def main():
    """Run comprehensive AG-SAR evaluation."""
    print("="*60)
    print("AG-SAR Comprehensive Evaluation Sweep")
    print("="*60)
    print(f"Model: {MODEL_NAME}")
    print(f"Samples per dataset: {NUM_SAMPLES}")
    print(f"Datasets: {len(DATASETS)}")
    print("="*60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model()

    # Initialize AG-SAR
    config = AGSARConfig(
        version=3,
        semantic_layers=4,
        varentropy_lambda=1.0,
        sigma_multiplier=-1.0,
    )
    agsar = AGSAR(model, tokenizer, config)

    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(bootstrap_samples=1000, confidence_level=0.95)

    # Run evaluation on all datasets
    all_results = []

    for ds_config in DATASETS:
        try:
            result = evaluate_dataset(agsar, ds_config, metrics_calc)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR on {ds_config['name']}: {e}")
            all_results.append({
                "dataset": ds_config["name"],
                "error": str(e),
            })

    # Cleanup
    agsar.cleanup()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"agsar_sweep_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "timestamp": timestamp,
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_file}")

    # Print summary table
    print(f"\n{'Dataset':<30} {'AUROC':<10} {'F1':<10} {'Latency':<10}")
    print("-" * 60)
    for r in all_results:
        if "error" in r:
            print(f"{r['dataset']:<30} ERROR: {r['error'][:30]}")
        else:
            auroc = r["metrics"].get("auroc", 0)
            f1 = r["metrics"].get("f1", 0)
            latency = r["mean_latency_ms"]
            print(f"{r['dataset']:<30} {auroc:<10.4f} {f1:<10.4f} {latency:<10.1f}")


if __name__ == "__main__":
    main()
