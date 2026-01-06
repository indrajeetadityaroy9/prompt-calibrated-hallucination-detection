#!/usr/bin/env python3
"""
Test Online JEPA with Test-Time Training (TTT).

This script proves that the predictor adapts to the specific context,
enabling detection of responses that contradict the provided facts
(even if those "facts" are fabricated in the context).

Key Test: We provide a FAKE fact in context. The predictor should:
- Accept responses consistent with the fake context
- Reject responses that contradict the fake context (even if factually true!)

This proves the system detects CONTEXT VIOLATION, not just grammatical errors.

Usage:
    python scripts/test_online_jepa.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig

# Configuration
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def main():
    print("=" * 70)
    print("ONLINE JEPA TEST - Test-Time Training Verification")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Initialize AG-SAR with Online JEPA Monitor (Optimized TTT)
    # NOTE: NO PRETRAINED PRIOR - Fresh predictor each time allows full context adaptation
    config = AGSARConfig(
        enable_jepa_monitor=True,
        enable_online_adaptation=True,  # The key flag
        # Optimization 1: Aggressive TTT settings (from config defaults)
        # online_adaptation_epochs=15, online_adaptation_lr=0.02
        # Optimization 3: DISABLED - pretrained prior hurts normal fact detection
        jepa_predictor_path=None,  # Fresh predictor - no pretrained bias
        jepa_drift_threshold=0.1,
    )
    engine = AGSAR(model, tokenizer, config)

    print("\n" + "=" * 70)
    print("TEST 1: Standard Facts (The Capital of France)")
    print("=" * 70)

    # Standard context with true fact
    context1 = "Geography Facts: The capital of France is Paris. It is known for the Eiffel Tower."
    query1 = "\nQuestion: What is the capital of France?\nAnswer: The capital of France is"

    prompt1 = context1 + query1

    # Consistent response (should have LOW drift)
    result1a = engine.compute_drift(prompt1, " Paris.")
    print(f"\nContext: '{context1[:50]}...'")
    print(f"Response: 'Paris' (Consistent with context)")
    print(f"  Drift: {result1a['drift']:.4f}")
    print(f"  Drift Ratio: {result1a['drift_ratio']:.4f}")
    print(f"  Context Loss: {result1a['context_loss']:.4f}")
    print(f"  Hallucination: {result1a['is_hallucination']}")

    # Inconsistent response (should have HIGH drift)
    result1b = engine.compute_drift(prompt1, " London.")
    print(f"\nResponse: 'London' (Contradicts context)")
    print(f"  Drift: {result1b['drift']:.4f}")
    print(f"  Drift Ratio: {result1b['drift_ratio']:.4f}")
    print(f"  Context Loss: {result1b['context_loss']:.4f}")
    print(f"  Hallucination: {result1b['is_hallucination']}")

    sep1 = result1b['drift'] - result1a['drift']
    print(f"\nSeparation (London - Paris): {sep1:.4f}")

    print("\n" + "=" * 70)
    print("TEST 2: Fabricated Facts (Proving Context Adaptation)")
    print("=" * 70)

    # FAKE context - we claim Paris is NOT the capital
    context2 = "Alternative Geography: In this universe, the capital of France is Watermelon. This is the official capital city."
    query2 = "\nQuestion: What is the capital of France?\nAnswer: The capital of France is"

    prompt2 = context2 + query2

    # Consistent with FAKE context (should have LOW drift)
    result2a = engine.compute_drift(prompt2, " Watermelon.")
    print(f"\nContext: '{context2[:60]}...'")
    print(f"Response: 'Watermelon' (Consistent with FAKE context)")
    print(f"  Drift: {result2a['drift']:.4f}")
    print(f"  Drift Ratio: {result2a['drift_ratio']:.4f}")
    print(f"  Context Loss: {result2a['context_loss']:.4f}")
    print(f"  Hallucination: {result2a['is_hallucination']}")

    # Inconsistent with FAKE context (factually TRUE but contradicts context)
    result2b = engine.compute_drift(prompt2, " Paris.")
    print(f"\nResponse: 'Paris' (Factually true, but CONTRADICTS the fake context)")
    print(f"  Drift: {result2b['drift']:.4f}")
    print(f"  Drift Ratio: {result2b['drift_ratio']:.4f}")
    print(f"  Context Loss: {result2b['context_loss']:.4f}")
    print(f"  Hallucination: {result2b['is_hallucination']}")

    sep2 = result2b['drift'] - result2a['drift']
    print(f"\nSeparation (Paris - Watermelon): {sep2:.4f}")

    print("\n" + "=" * 70)
    print("TEST 3: RAG-Style Context (Multiple Facts)")
    print("=" * 70)

    # Rich RAG-style context
    context3 = """
Retrieved Documents:
[Doc 1] Company Report 2024: TechCorp announced revenue of $5.2 billion in Q3.
[Doc 2] The CEO of TechCorp is Sarah Johnson, appointed in 2022.
[Doc 3] TechCorp headquarters is located in Austin, Texas.

Based on the above documents, answer the following question.
Question: What was TechCorp's Q3 revenue?
Answer: According to the documents, TechCorp's Q3 revenue was"""

    # Consistent
    result3a = engine.compute_drift(context3, " $5.2 billion.")
    print(f"\nResponse: '$5.2 billion' (Matches document)")
    print(f"  Drift: {result3a['drift']:.4f}")
    print(f"  Drift Ratio: {result3a['drift_ratio']:.4f}")
    print(f"  Context Loss: {result3a['context_loss']:.4f}")
    print(f"  Hallucination: {result3a['is_hallucination']}")

    # Fabricated number
    result3b = engine.compute_drift(context3, " $8.7 billion.")
    print(f"\nResponse: '$8.7 billion' (Fabricated - not in documents)")
    print(f"  Drift: {result3b['drift']:.4f}")
    print(f"  Drift Ratio: {result3b['drift_ratio']:.4f}")
    print(f"  Context Loss: {result3b['context_loss']:.4f}")
    print(f"  Hallucination: {result3b['is_hallucination']}")

    sep3 = result3b['drift'] - result3a['drift']
    print(f"\nSeparation (Fabricated - True): {sep3:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Compute drift ratio separations (more robust metric)
    sep1_ratio = result1b['drift_ratio'] - result1a['drift_ratio']
    sep2_ratio = result2b['drift_ratio'] - result2a['drift_ratio']
    sep3_ratio = result3b['drift_ratio'] - result3a['drift_ratio']

    print(f"\nTest 1 (Standard Facts):     Drift Sep = {sep1:.4f}, Ratio Sep = {sep1_ratio:.4f}")
    print(f"Test 2 (Fabricated Facts):   Drift Sep = {sep2:.4f}, Ratio Sep = {sep2_ratio:.4f}")
    print(f"Test 3 (RAG-Style Context):  Drift Sep = {sep3:.4f}, Ratio Sep = {sep3_ratio:.4f}")

    # Check drift ratio separation (more meaningful than raw drift)
    all_positive = sep1_ratio > 0 and sep2_ratio > 0 and sep3_ratio > 0
    avg_sep_ratio = (sep1_ratio + sep2_ratio + sep3_ratio) / 3
    avg_sep_drift = (sep1 + sep2 + sep3) / 3

    print(f"\nAverage Drift Separation: {avg_sep_drift:.4f}")
    print(f"Average Ratio Separation: {avg_sep_ratio:.4f}")

    if all_positive and avg_sep_ratio > 0.1:
        print(f"\n[PASS] Optimized TTT is working!")
        print(f"       All ratio separations positive, avg = {avg_sep_ratio:.4f}")
        print(f"       The predictor successfully distinguishes context violations.")
    elif all_positive and avg_sep_ratio > 0.01:
        print(f"\n[PARTIAL] Separation exists but may be marginal.")
        print(f"          Try increasing epochs or adjusting LR.")
    else:
        print(f"\n[FAIL] Optimized TTT may not be effective.")
        print(f"       Check pretrained prior or context quality.")

    # Cleanup
    engine.cleanup()


if __name__ == "__main__":
    main()
