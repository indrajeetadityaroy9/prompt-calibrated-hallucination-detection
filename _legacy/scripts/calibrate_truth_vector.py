#!/usr/bin/env python3
"""
Calibrate Truth Vector for a given model.

Extracts a geometric "truthfulness direction" from the model's residual stream
using TruthfulQA and synthetic fact/counterfact pairs.

Usage:
    # Default: Llama-3.1-8B-Instruct with 2000 samples
    python scripts/calibrate_truth_vector.py

    # Custom model
    python scripts/calibrate_truth_vector.py --model meta-llama/Llama-3.1-70B-Instruct

    # Quick test with GPT-2
    python scripts/calibrate_truth_vector.py --model gpt2 --n-samples 100

    # Custom layer ratio
    python scripts/calibrate_truth_vector.py --layer-ratio 0.6

Success Metric:
    Look for: "Bounds: Lie=0.XX | Truth=0.YY"
    If the gap (Truth - Lie) is < 0.01, calibration failed.
    Try: different layer_ratio, more samples, or better data quality.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Set HF token from environment
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
os.environ["HF_TOKEN"] = HF_TOKEN


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate Truth Vector for intrinsic hallucination detection"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=2000,
        help="Total number of pairs (TruthfulQA + Synthetic)",
    )
    parser.add_argument(
        "--layer-ratio",
        type=float,
        default=0.5,
        help="Layer ratio (0.5 = middle layer)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/truth_vectors",
        help="Output directory for truth vectors",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Truth Vector Calibration")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Samples:     {args.n_samples}")
    print(f"Layer Ratio: {args.layer_ratio}")
    print(f"Device:      {args.device}")
    print()

    # Import after argument parsing to show help quickly
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ag_sar.calibration.truth_vector import TruthVectorCalibrator, TruthVectorConfig
    from experiments.data.truthfulqa import TruthfulQADataset
    from experiments.data.synthetic_facts import SyntheticFactGenerator

    # Load model
    print(f"Loading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    target_layer = int(num_layers * args.layer_ratio)
    print(f"Model loaded: {num_layers} layers")
    print(f"Target layer: {target_layer} (ratio={args.layer_ratio})")
    print()

    # Initialize calibrator
    config = TruthVectorConfig(
        layer_ratio=args.layer_ratio,
        normalize=True,
    )
    calibrator = TruthVectorCalibrator(model, tokenizer, config)

    # Split samples: 50% TruthfulQA, 50% Synthetic
    n_tqa = args.n_samples // 2
    n_synthetic = args.n_samples - n_tqa

    # =========================================================================
    # Phase 1: TruthfulQA Pairs
    # =========================================================================
    print("Phase 1: Loading TruthfulQA pairs...")
    try:
        tqa = TruthfulQADataset(num_samples=n_tqa * 2)  # *2 because each yields 2 samples
        tqa.load()

        # Group samples by question ID to get fact/misconception pairs
        pairs_by_id = {}
        for sample in tqa:
            base_id = sample.id.rsplit("_", 1)[0]  # "0_fact" -> "0"
            if base_id not in pairs_by_id:
                pairs_by_id[base_id] = {}
            if sample.label == 0:
                pairs_by_id[base_id]["fact"] = sample
            else:
                pairs_by_id[base_id]["misconception"] = sample

        tqa_count = 0
        for base_id, pair in tqdm(pairs_by_id.items(), desc="TruthfulQA"):
            if "fact" in pair and "misconception" in pair:
                fact_text = pair["fact"].prompt + pair["fact"].response
                lie_text = pair["misconception"].prompt + pair["misconception"].response
                calibrator.add_pair(fact_text, lie_text)
                tqa_count += 1
                if tqa_count >= n_tqa:
                    break

        print(f"Added {tqa_count} TruthfulQA pairs")
    except Exception as e:
        print(f"Warning: Could not load TruthfulQA: {e}")
        print("Proceeding with synthetic data only")
        n_synthetic = args.n_samples

    # =========================================================================
    # Phase 2: Synthetic Pairs
    # =========================================================================
    print(f"\nPhase 2: Generating {n_synthetic} synthetic pairs...")
    generator = SyntheticFactGenerator(seed=42)

    for pair in tqdm(generator.generate_pairs(n_synthetic), desc="Synthetic", total=n_synthetic):
        calibrator.add_pair(pair.fact, pair.counterfact)

    print(f"Total pairs: {calibrator.n_samples}")

    # =========================================================================
    # Phase 3: Compute and Save
    # =========================================================================
    print("\nPhase 3: Computing Truth Vector...")

    # Generate output filename
    model_name = args.model.split("/")[-1].replace("-", "_").lower()
    output_path = Path(args.output_dir) / f"{model_name}.pt"

    vector, meta = calibrator.compute_and_save(str(output_path))

    # Summary
    print()
    print("=" * 60)
    print("Calibration Complete!")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Gap:    {meta['mu_pos'] - meta['mu_neg']:.4f}")

    if meta['mu_pos'] - meta['mu_neg'] >= 0.01:
        print("Status: ✅ SUCCESS - Gap is sufficient")
    else:
        print("Status: ⚠️ WARNING - Gap < 0.01, calibration may have failed")
        print("Try: different layer_ratio, more samples, or better data quality")


if __name__ == "__main__":
    main()
