#!/usr/bin/env python3
"""
Calibrate Truth-Head weights using TruthfulQA validation split.

This script identifies "Truth Heads" vs "Induction Heads" based on
correlation between per-head centrality contributions and factual correctness.

Based on ITI (Inference-Time Intervention) by Li et al. (2024).

Usage:
    python scripts/calibrate_truth_heads.py \
        --model meta-llama/Llama-3.2-3B \
        --output configs/llama3.2_truth_heads.json

Output:
    JSON file with:
    - head_weights: (L*H,) list of weights in [0, 1]
    - Positive score = Truth Head (weight → 1.0)
    - Negative score = Induction Head (weight → 0.0)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory for ag_sar import
sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar import AGSAR
from ag_sar.config import AGSARConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Truth-Head weights")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/llama3.2_truth_heads.json",
        help="Output path for calibrated weights JSON",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of TruthfulQA samples for calibration",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=5.0,
        help="Sigmoid temperature for weight normalization",
    )
    parser.add_argument(
        "--semantic-layers",
        type=int,
        default=4,
        help="Number of semantic layers to analyze",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    return parser.parse_args()


def load_truthfulqa_calibration(num_samples: int = 50) -> List[Dict]:
    """
    Load TruthfulQA validation set for calibration.

    Uses first N samples for calibration (rest held out for evaluation).

    Returns:
        List of dicts with 'question', 'correct_answers', 'incorrect_answers'
    """
    ds = load_dataset("truthful_qa", "generation")["validation"]

    samples = []
    for i in range(min(num_samples, len(ds))):
        item = ds[i]
        samples.append({
            "question": item["question"],
            "correct_answers": item["correct_answers"],
            "incorrect_answers": item["incorrect_answers"],
        })

    return samples


def compute_per_head_contribution(
    ag_sar: AGSAR,
    prompt: str,
    response: str,
    debug: bool = False,
) -> torch.Tensor:
    """
    Compute per-head contribution using K embedding norms (ITI-style).

    Instead of attention-weighted sums (which collapse to uniform with uniform v),
    we use the K embedding norms directly. This captures per-head activation strength
    which correlates with head importance for that input.

    Returns:
        per_head_mean: (L*H,) mean K-norm per head for response tokens
    """
    # Tokenize
    input_ids, attention_mask, response_start = ag_sar._tokenize(prompt, response)

    # Extract Q/K stacks
    Q_stack, K_stack, _, _ = ag_sar.extractor.extract_semantic_qk(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_flash_attn=ag_sar.config.use_flash_attn
    )

    # K_stack shape: (B, L*H, S, D)
    B, total_heads, S, D = K_stack.shape

    # Focus on response tokens only
    if response_start < S:
        K_response = K_stack[:, :, response_start:, :]  # (B, L*H, S_resp, D)
    else:
        K_response = K_stack

    # Compute L2 norm per head per token, then average over sequence
    # This gives us the "activation strength" of each head for this input
    K_norms = K_response.float().norm(dim=-1)  # (B, L*H, S_resp)

    if debug:
        print(f"  K_norms shape: {K_norms.shape}")
        print(f"  K_norms range: [{K_norms.min():.6f}, {K_norms.max():.6f}]")
        head_means = K_norms.mean(dim=(0, 2))
        print(f"  per-head K-norm means (first 5): {head_means[:5].tolist()}")
        print(f"  per-head K-norm std: {head_means.std():.6f}")
        print(f"  per-head unique values: {len(head_means.unique())}")

    # Average over batch and sequence to get per-head activation strength
    per_head_mean = K_norms.mean(dim=(0, 2))  # (L*H,)

    return per_head_mean.cpu()


def calibrate_truth_heads(
    model_name: str,
    samples: List[Dict],
    semantic_layers: int = 4,
    temperature: float = 5.0,
    device: str = "cuda",
) -> Tuple[torch.Tensor, Dict]:
    """
    Calibrate Truth-Head weights using TruthfulQA samples.

    For each head h (flattened L*H), compute Difference in Means:
        Score_h = Mean(C_h | Correct) - Mean(C_h | Incorrect)

    Then normalize with sigmoid:
        weight_h = Sigmoid(Score_h * temperature)

    Args:
        model_name: HuggingFace model name
        samples: List of TruthfulQA samples
        semantic_layers: Number of layers to analyze
        temperature: Sigmoid temperature for normalization
        device: Device to run on

    Returns:
        weights: (L*H,) tensor of calibrated weights
        metadata: Dict with calibration info
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device} if device.startswith("cuda") else device,
        trust_remote_code=True,
    )
    # Ensure model is on single device
    if device.startswith("cuda"):
        model = model.to(device)

    # Initialize AG-SAR with return_raw=True for per-head contributions
    config = AGSARConfig(
        semantic_layers=semantic_layers,
        use_torch_compile=False,  # Disable for calibration
    )
    ag_sar = AGSAR(model, tokenizer, config)

    # Collect per-head contributions for correct vs incorrect responses
    correct_contribs = []
    incorrect_contribs = []

    print(f"Processing {len(samples)} calibration samples...")
    for idx, sample in enumerate(tqdm(samples)):
        question = sample["question"]
        debug = (idx == 0)  # Debug first sample only

        # Process correct answers
        for answer in sample["correct_answers"][:1]:  # Use first correct answer
            prompt = f"Q: {question}\nA:"
            if debug:
                print(f"\nDebug first sample (correct): {answer[:50]}...")
            contrib = compute_per_head_contribution(ag_sar, prompt, f" {answer}", debug=debug)
            if contrib is not None:
                correct_contribs.append(contrib)

        # Process incorrect answers
        for answer in sample["incorrect_answers"][:1]:  # Use first incorrect answer
            prompt = f"Q: {question}\nA:"
            if debug:
                print(f"\nDebug first sample (incorrect): {answer[:50]}...")
            contrib = compute_per_head_contribution(ag_sar, prompt, f" {answer}", debug=debug)
            if contrib is not None:
                incorrect_contribs.append(contrib)

    if not correct_contribs or not incorrect_contribs:
        raise ValueError("No per-head contributions collected. Check return_raw=True.")

    # Stack and compute means - USE FLOAT32 to avoid precision loss
    correct_stack = torch.stack(correct_contribs).float()    # (N_correct, L*H)
    incorrect_stack = torch.stack(incorrect_contribs).float()  # (N_incorrect, L*H)

    mean_correct = correct_stack.mean(dim=0)     # (L*H,)
    mean_incorrect = incorrect_stack.mean(dim=0)  # (L*H,)

    # Score_h = Mean(correct) - Mean(incorrect)
    # Positive = Truth Head, Negative = Induction Head
    scores = mean_correct - mean_incorrect

    # Standardize scores to have meaningful range before sigmoid
    # This ensures the sigmoid can differentiate between heads
    scores_std = scores.std() + 1e-8
    scores_mean = scores.mean()
    scores_normalized = (scores - scores_mean) / scores_std

    # Normalize to [0, 1] with sigmoid
    weights = torch.sigmoid(scores_normalized * temperature)

    # Get model config for metadata
    num_layers = ag_sar.num_layers
    num_semantic_layers = len(ag_sar._semantic_layer_indices)
    num_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else 32

    metadata = {
        "model": model_name,
        "num_layers": num_layers,
        "num_semantic_layers": num_semantic_layers,
        "num_heads_per_layer": num_heads,
        "total_heads": len(weights),
        "num_calibration_samples": len(samples),
        "num_correct_responses": len(correct_contribs),
        "num_incorrect_responses": len(incorrect_contribs),
        "temperature": temperature,
        "calibration_date": datetime.now().isoformat(),
        "mean_correct": mean_correct.mean().item(),
        "mean_incorrect": mean_incorrect.mean().item(),
        "scores_min": scores.min().item(),
        "scores_max": scores.max().item(),
        "scores_mean": scores.mean().item(),
        "weights_min": weights.min().item(),
        "weights_max": weights.max().item(),
        "weights_mean": weights.mean().item(),
    }

    # Cleanup
    ag_sar.cleanup()
    del model
    torch.cuda.empty_cache()

    return weights, metadata


def main():
    args = parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load calibration samples
    print(f"Loading {args.num_samples} TruthfulQA samples for calibration...")
    samples = load_truthfulqa_calibration(args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Run calibration
    weights, metadata = calibrate_truth_heads(
        model_name=args.model,
        samples=samples,
        semantic_layers=args.semantic_layers,
        temperature=args.temperature,
        device=args.device,
    )

    # Save to JSON
    output_data = {
        **metadata,
        "head_weights": weights.tolist(),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCalibration complete!")
    print(f"  Total heads: {len(weights)}")
    print(f"  Weights range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Weights mean: {weights.mean():.4f}")
    print(f"  Output saved to: {output_path}")

    # Summary statistics
    truth_heads = (weights > 0.6).sum().item()
    induction_heads = (weights < 0.4).sum().item()
    neutral_heads = len(weights) - truth_heads - induction_heads

    print(f"\nHead Classification:")
    print(f"  Truth Heads (w > 0.6): {truth_heads} ({100*truth_heads/len(weights):.1f}%)")
    print(f"  Induction Heads (w < 0.4): {induction_heads} ({100*induction_heads/len(weights):.1f}%)")
    print(f"  Neutral Heads: {neutral_heads} ({100*neutral_heads/len(weights):.1f}%)")


if __name__ == "__main__":
    main()
