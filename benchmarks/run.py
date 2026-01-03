"""
AG-SAR Hallucination Detection Benchmark Runner

Runs experiments for paper validation:
- Experiment 1: Confident Hallucination Detection (HaluEval)
- Computes AUROC, AUPRC, F1 on full and confident subsets
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import torch

# torch.compile is disabled in ag_sar.ops.torch_functional via AG_SAR_ENABLE_COMPILE env var
import yaml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import AGSAR, AGSARConfig
from ag_sar import enable_h100_optimizations, is_h100, get_optimal_dtype

# Import dataset loaders
from loaders import load_halueval, load_ragtruth


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_halueval_dataset(dataset_name: str, num_samples: int) -> List[Dict]:
    """
    Load HaluEval dataset.

    Supported datasets:
    - halueval_summarization
    - halueval_qa
    - halueval_dialogue
    """
    print(f"Loading dataset: {dataset_name}")

    if dataset_name == "halueval_summarization":
        ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    elif dataset_name == "halueval_qa":
        ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    elif dataset_name == "halueval_dialogue":
        ds = load_dataset("pminervini/HaluEval", "dialogue", split="data")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert to list and sample
    samples = list(ds)
    if num_samples < len(samples):
        # Deterministic sampling for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(samples), num_samples, replace=False)
        samples = [samples[i] for i in indices]

    print(f"Loaded {len(samples)} samples")
    return samples


def prepare_prompt_response(sample: Dict, dataset_name: str) -> Tuple[str, str, int]:
    """
    Extract prompt, response, and label from dataset sample.

    Returns:
        prompt: The input prompt
        response: The model response to evaluate
        label: 1 if hallucinated, 0 if not
    """
    if "summarization" in dataset_name:
        # Summarization format
        prompt = f"Document: {sample['document']}\n\nSummarize the above document:"

        # Use hallucinated response for label=1, right response for label=0
        if sample.get("hallucinated_response"):
            response = sample["hallucinated_response"]
            label = 1
        else:
            response = sample["right_response"]
            label = 0

    elif "qa" in dataset_name:
        prompt = f"Question: {sample['question']}\n\nContext: {sample.get('knowledge', '')}\n\nAnswer:"

        if sample.get("hallucinated_answer"):
            response = sample["hallucinated_answer"]
            label = 1
        else:
            response = sample["right_answer"]
            label = 0

    elif "dialogue" in dataset_name:
        prompt = f"Dialogue history:\n{sample['dialogue_history']}\n\nResponse:"

        if sample.get("hallucinated_response"):
            response = sample["hallucinated_response"]
            label = 1
        else:
            response = sample["right_response"]
            label = 0
    else:
        raise ValueError(f"Unknown dataset format: {dataset_name}")

    return prompt, response, label


def compute_model_confidence(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: torch.device,
) -> float:
    """
    Compute model's confidence (mean log probability) for the response.

    Returns:
        confidence: Mean probability of response tokens (0-1 scale)
    """
    # Tokenize
    full_text = prompt + response
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)

    response_start = prompt_ids.size(1)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

        # Get log probs for response tokens
        log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
        response_tokens = full_ids[:, response_start:]

        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

        # Convert to probability and average
        mean_prob = token_log_probs.exp().mean().item()

    return mean_prob


def run_experiment(config: dict) -> Dict:
    """Run the hallucination detection experiment."""

    # Setup
    enable_h100_optimizations()
    dtype = get_optimal_dtype()

    print("=" * 60)
    print(f"Experiment: {config['experiment_name']}")
    print(f"Model: {config['model']['name']}")
    print(f"Dataset: {config['dataset']}")
    print(f"Samples: {config['num_samples']}")

    # Load model first, then extract device from it
    print("\nLoading model...")
    model_name = config["model"]["name"]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=config["model"].get("device_map", "auto"),
        attn_implementation=config["model"].get("attn_implementation", "sdpa"),
        trust_remote_code=True,
    )

    # Get device from model (critical for multi-GPU setups)
    device = next(model.parameters()).device
    print(f"Device: {device}, Dtype: {dtype}")
    print("=" * 60)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize AG-SAR
    print("Initializing AG-SAR...")
    ag_config = AGSARConfig(
        semantic_layers=config["config"].get("semantic_layers", 4),
        power_iteration_steps=config["config"].get("power_iteration_steps", 3),
        residual_weight=config["config"].get("residual_weight", 0.5),
        lambda_roughness=config["config"].get("lambda_roughness", config["config"].get("roughness_lambda", 20.0)),
        enable_register_filter=config["config"].get("enable_register_filter", True),
        enable_spectral_roughness=config["config"].get("enable_spectral_roughness", True),
        kurtosis_threshold=config["config"].get("kurtosis_threshold", 2.0),
        ema_decay=config["config"].get("ema_decay", 0.995),
        sink_token_count=config["config"].get("sink_token_count", 4),
    )
    print(f"  Register Filter: {ag_config.enable_register_filter}")
    print(f"  MLP Divergence: {ag_config.enable_spectral_roughness}")
    print(f"  Lambda Roughness: {ag_config.lambda_roughness}")

    ag_sar = AGSAR(model, tokenizer, config=ag_config)

    # Load dataset based on config
    dataset_name = config["dataset"]
    num_samples = config["num_samples"]

    if dataset_name == "ragtruth":
        # RAGTruth: Natural hallucinations from real RAG tasks
        eval_samples = load_ragtruth(split="test", num_samples=num_samples, task="QA")
    elif "halueval" in dataset_name:
        # HaluEval: Synthetic hallucinations
        variant = dataset_name.replace("halueval_", "")
        eval_samples = load_halueval(dataset_variant=variant, num_samples=num_samples)
    else:
        # Legacy: direct HaluEval loading (backward compatibility)
        samples = load_halueval_dataset(dataset_name, num_samples)

        eval_samples = []
        for sample in samples:
            if "summarization" in dataset_name:
                doc = sample['document'][:2000] + "..." if len(sample['document']) > 2000 else sample['document']

                if sample.get("hallucinated_summary"):
                    eval_samples.append({
                        "prompt": f"Document: {doc}\n\nSummarize the above document:",
                        "response": sample["hallucinated_summary"],
                        "label": 1,
                    })
                if sample.get("right_summary"):
                    eval_samples.append({
                        "prompt": f"Document: {doc}\n\nSummarize the above document:",
                        "response": sample["right_summary"],
                        "label": 0,
                    })
            elif "qa" in dataset_name:
                if sample.get("hallucinated_answer"):
                    eval_samples.append({
                        "prompt": f"Q: {sample['question']}\nContext: {sample.get('knowledge', '')}\nA:",
                        "response": sample["hallucinated_answer"],
                        "label": 1,
                    })
                if sample.get("right_answer"):
                    eval_samples.append({
                        "prompt": f"Q: {sample['question']}\nContext: {sample.get('knowledge', '')}\nA:",
                        "response": sample["right_answer"],
                        "label": 0,
                    })

    print(f"\nTotal evaluation samples: {len(eval_samples)}")
    print(f"Hallucinated: {sum(1 for s in eval_samples if s['label'] == 1)}")
    print(f"Correct: {sum(1 for s in eval_samples if s['label'] == 0)}")

    # Run evaluation
    print("\nRunning AG-SAR evaluation...")
    results = []

    batch_size = config["model"].get("batch_size", 1)
    confidence_threshold = config["evaluation"].get("confidence_threshold", 0.8)

    start_time = time.time()

    # Process in batches for H100 throughput
    num_batches = (len(eval_samples) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Evaluating batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(eval_samples))
        batch_samples = eval_samples[batch_start:batch_end]

        for sample in batch_samples:
            try:
                # Compute AG-SAR uncertainty AND model confidence in ONE forward pass
                result = ag_sar.compute_uncertainty(
                    sample["prompt"],
                    sample["response"],
                    return_details=True,
                )

                results.append({
                    "uncertainty": result["score"],
                    "confidence": result["model_confidence"],
                    "label": sample["label"],
                    "prompt": sample["prompt"][:100],
                    "response": sample["response"][:100],
                })

            except Exception as e:
                print(f"\nError: {str(e)[:80]}")
                continue

        # Periodic progress and cache management
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = len(results) / elapsed
            print(f"\n  Processed {len(results)}/{len(eval_samples)} ({rate:.1f} samples/sec)")
            # Clear CUDA cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({len(results)/total_time:.1f} samples/sec)")

    # Compute metrics
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    labels = np.array([r["label"] for r in results])
    uncertainties = np.array([r["uncertainty"] for r in results])
    confidences = np.array([r["confidence"] for r in results])

    # Full dataset metrics
    print("\n[Full Dataset]")
    auroc_full = roc_auc_score(labels, uncertainties)
    auprc_full = average_precision_score(labels, uncertainties)

    # Binary predictions at threshold
    threshold = np.median(uncertainties)
    preds = (uncertainties > threshold).astype(int)
    f1_full = f1_score(labels, preds)
    prec_full = precision_score(labels, preds)
    rec_full = recall_score(labels, preds)

    print(f"  AUROC:     {auroc_full:.4f}")
    print(f"  AUPRC:     {auprc_full:.4f}")
    print(f"  F1:        {f1_full:.4f}")
    print(f"  Precision: {prec_full:.4f}")
    print(f"  Recall:    {rec_full:.4f}")

    # Confident subset metrics (KEY METRIC)
    print(f"\n[Confident Subset (Model Confidence > {confidence_threshold})]")
    confident_mask = confidences > confidence_threshold
    n_confident = confident_mask.sum()

    if n_confident > 10:
        labels_conf = labels[confident_mask]
        uncert_conf = uncertainties[confident_mask]

        auroc_conf = roc_auc_score(labels_conf, uncert_conf)
        auprc_conf = average_precision_score(labels_conf, uncert_conf)

        threshold_conf = np.median(uncert_conf)
        preds_conf = (uncert_conf > threshold_conf).astype(int)
        f1_conf = f1_score(labels_conf, preds_conf)

        print(f"  Samples:   {n_confident} ({100*n_confident/len(labels):.1f}%)")
        print(f"  AUROC:     {auroc_conf:.4f}  {'<-- SOTA if > 0.80' if auroc_conf > 0.75 else ''}")
        print(f"  AUPRC:     {auprc_conf:.4f}")
        print(f"  F1:        {f1_conf:.4f}")

        if auroc_conf > 0.80:
            print("\n  *** SOTA RESULT: AUROC > 0.80 on Confident Subset! ***")
    else:
        print(f"  Only {n_confident} samples in confident subset (need >10)")
        auroc_conf = None

    # Save results
    output_dir = Path(config["output"].get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "experiment": config["experiment_name"],
        "model": config["model"]["name"],
        "dataset": config["dataset"],
        "num_samples": len(results),
        "total_time_sec": total_time,
        "full_dataset": {
            "auroc": float(auroc_full),
            "auprc": float(auprc_full),
            "f1": float(f1_full),
            "precision": float(prec_full),
            "recall": float(rec_full),
        },
        "confident_subset": {
            "confidence_threshold": confidence_threshold,
            "num_samples": int(n_confident),
            "auroc": float(auroc_conf) if auroc_conf else None,
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if config["output"].get("save_predictions", True):
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}/")

    # Cleanup
    ag_sar.cleanup()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Benchmark Runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load config and print settings",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.dry_run:
        print("Dry run - Config loaded:")
        print(yaml.dump(config, default_flow_style=False))
        return

    results = run_experiment(config)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
