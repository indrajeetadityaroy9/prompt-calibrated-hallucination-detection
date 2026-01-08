#!/usr/bin/env python3
"""
AG-SAR Minimal Demo.

Demonstrates basic usage of AG-SAR for hallucination detection.
Run with: python examples/minimal_demo.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ag_sar import AGSAR, AGSARConfig


def main():
    print("=" * 60)
    print("AG-SAR: Single-Pass Hallucination Detection Demo")
    print("=" * 60)

    # Load a small model for demo (GPT-2)
    print("\n[1/4] Loading GPT-2 model...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure AG-SAR
    print("[2/4] Configuring AG-SAR...")
    config = AGSARConfig(
        semantic_layers=2,  # Use last 2 layers for GPT-2
        uncertainty_metric="gse",  # Graph-Shifted Entropy
        hallucination_threshold=0.5,
    )

    agsar = AGSAR(model, tokenizer, config)
    print(f"      Metric: {config.uncertainty_metric}")
    print(f"      Threshold: {config.hallucination_threshold}")

    # Test examples
    print("\n[3/4] Running hallucination detection...")
    print("-" * 60)

    examples = [
        {
            "prompt": "What is the capital of France?",
            "response": "Paris",
            "expected": "grounded (factual)",
        },
        {
            "prompt": "What is 2 + 2?",
            "response": "4",
            "expected": "grounded (factual)",
        },
        {
            "prompt": "Who was the first person to walk on Mars?",
            "response": "Neil Armstrong walked on Mars in 1969.",
            "expected": "hallucination (incorrect)",
        },
        {
            "prompt": "What is the chemical formula for water?",
            "response": "H2O",
            "expected": "grounded (factual)",
        },
    ]

    results = []
    for i, ex in enumerate(examples, 1):
        is_hall, confidence, details = agsar.detect_hallucination(
            ex["prompt"],
            ex["response"],
        )

        status = "HALLUCINATION" if is_hall else "GROUNDED"
        results.append({
            "prompt": ex["prompt"],
            "response": ex["response"],
            "expected": ex["expected"],
            "detected": status,
            "score": details["score"],
            "confidence": confidence,
        })

        print(f"\nExample {i}:")
        print(f"  Prompt:   {ex['prompt']}")
        print(f"  Response: {ex['response']}")
        print(f"  Expected: {ex['expected']}")
        print(f"  Detected: {status} (score: {details['score']:.4f}, conf: {confidence:.4f})")

    # Summary
    print("\n[4/4] Summary")
    print("-" * 60)
    print(f"Total examples: {len(examples)}")
    print(f"Detection threshold: {config.hallucination_threshold}")

    # Show score distribution
    scores = [r["score"] for r in results]
    print(f"\nScore statistics:")
    print(f"  Min:  {min(scores):.4f}")
    print(f"  Max:  {max(scores):.4f}")
    print(f"  Mean: {sum(scores)/len(scores):.4f}")

    # Cleanup
    agsar.cleanup()
    print("\nDemo completed successfully!")


def demo_with_return_details():
    """Demonstrate detailed output from uncertainty computation."""
    print("\n" + "=" * 60)
    print("AG-SAR: Detailed Analysis Demo")
    print("=" * 60)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    config = AGSARConfig(semantic_layers=2)
    agsar = AGSAR(model, tokenizer, config)

    prompt = "Machine learning is"
    response = "a subset of artificial intelligence that enables computers to learn from data."

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    # Get detailed results
    details = agsar.compute_uncertainty(prompt, response, return_details=True)

    print("\nDetailed Results:")
    for key, value in details.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    agsar.cleanup()


if __name__ == "__main__":
    main()

    # Uncomment to run additional demos:
    # demo_with_return_details()
