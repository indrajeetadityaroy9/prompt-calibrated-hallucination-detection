#!/usr/bin/env python3
"""
JEPA Drift Monitor Sanity Check.

Verifies that the trained predictor produces sensible drift values:
- Coherent text → Low drift (~0.2)
- Incoherent/hallucinatory text → High drift (~0.5+)

Usage:
    python scripts/check_drift.py
"""

import sys

# Pre-flight installation check
from experiments.utils.preflight import check_installation
check_installation()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig

# Configuration
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PREDICTOR_PATH = "data/models/jepa_predictor.pt"


def main():
    print("=" * 60)
    print("JEPA Drift Monitor Sanity Check")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Initialize AG-SAR with JEPA Monitor
    config = AGSARConfig(
        enable_jepa_monitor=True,
        jepa_predictor_path=PREDICTOR_PATH,
        jepa_drift_threshold=0.4,
    )
    engine = AGSAR(model, tokenizer, config)

    # Test cases
    test_cases = [
        # (prompt, response, expected_drift_level)
        ("Geography: ", "The capital of France is Paris.", "LOW"),
        ("Geography: ", "The capital of France is watermelon.", "HIGH"),
        ("Math: ", "Two plus two equals four.", "LOW"),
        ("Math: ", "Two plus two equals purple elephant.", "HIGH"),
        ("Science: ", "Water boils at 100 degrees Celsius.", "LOW"),
        ("Science: ", "Water boils at banana kilometers.", "HIGH"),
        ("History: ", "World War II ended in 1945.", "LOW"),
        ("History: ", "World War II ended in refrigerator.", "HIGH"),
    ]

    print("\n" + "-" * 60)
    print("DRIFT MEASUREMENTS")
    print("-" * 60)

    results = []
    for prompt, response, expected in test_cases:
        result = engine.compute_drift(prompt, response)
        drift = result["drift"]
        trust = result["trust_score"]
        is_hall = result["is_hallucination"]

        status = "OK" if (expected == "LOW" and drift < 0.4) or (expected == "HIGH" and drift > 0.3) else "CHECK"

        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print(f"  Drift: {drift:.4f} (Expected: {expected})")
        print(f"  Trust: {trust:.4f}")
        print(f"  Hallucination: {is_hall}")
        print(f"  Status: [{status}]")

        results.append({
            "prompt": prompt,
            "response": response,
            "drift": drift,
            "expected": expected,
            "status": status,
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    low_drifts = [r["drift"] for r in results if r["expected"] == "LOW"]
    high_drifts = [r["drift"] for r in results if r["expected"] == "HIGH"]

    avg_low = sum(low_drifts) / len(low_drifts) if low_drifts else 0
    avg_high = sum(high_drifts) / len(high_drifts) if high_drifts else 0

    print(f"\nCoherent (Expected LOW):   Avg Drift = {avg_low:.4f}")
    print(f"Incoherent (Expected HIGH): Avg Drift = {avg_high:.4f}")
    print(f"Separation Gap: {avg_high - avg_low:.4f}")

    if avg_high > avg_low and (avg_high - avg_low) > 0.05:
        print("\n[PASS] Drift metric shows expected separation between coherent and incoherent text.")
    else:
        print("\n[WARNING] Drift separation may be insufficient. Consider:")
        print("  1. Retraining predictor with more data")
        print("  2. Adjusting the drift threshold")
        print("  3. Using a different monitor layer")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
