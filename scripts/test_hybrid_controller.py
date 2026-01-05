#!/usr/bin/env python3
"""
Test Hybrid Controller (v13.0) - The Production Solution.

This script proves that the symbolic entity check solves the "Paris vs London"
problem that neural methods (JEPA predictor, embeddings) cannot handle.

Key Test Cases:
1. Standard Facts: Paris (correct) vs London (wrong)
2. Fabricated Facts: Watermelon (context-consistent) vs Paris (context-violation)
3. RAG Numbers: $5.2B (correct) vs $8.7B (hallucinated)

Usage:
    python scripts/test_hybrid_controller.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig

# Configuration
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def test_symbolic_only():
    """Test the symbolic overlap function directly."""
    print("=" * 70)
    print("SYMBOLIC OVERLAP TEST (Direct Function)")
    print("=" * 70)

    from ag_sar.measures.symbolic import compute_context_overlap, compute_numeric_consistency

    # Test 1: Paris vs London
    context1 = "Geography Facts: The capital of France is Paris. It is known for the Eiffel Tower."

    score_paris, details_paris = compute_context_overlap("Paris", context1)
    score_london, details_london = compute_context_overlap("London", context1)

    print(f"\nContext: '{context1[:50]}...'")
    print(f"Response 'Paris':  Score = {score_paris:.2f}, Violations = {details_paris['violations']}")
    print(f"Response 'London': Score = {score_london:.2f}, Violations = {details_london['violations']}")
    print(f"Separation: {score_paris - score_london:.2f} (target: > 0)")

    # Test 2: RAG Numbers
    context2 = "[Doc 1] TechCorp announced revenue of $5.2 billion in Q3 2024."

    score_correct, details_correct = compute_numeric_consistency("$5.2 billion", context2)
    score_wrong, details_wrong = compute_numeric_consistency("$8.7 billion", context2)

    print(f"\nContext: '{context2}'")
    print(f"Response '$5.2B':  Score = {score_correct:.2f}, Violations = {details_correct['violations']}")
    print(f"Response '$8.7B':  Score = {score_wrong:.2f}, Violations = {details_wrong['violations']}")
    print(f"Separation: {score_correct - score_wrong:.2f} (target: > 0)")

    # Test 3: Multiple entities
    context3 = "The meeting was attended by CEO Sarah Johnson and CFO Michael Chen."

    score_ok, details_ok = compute_context_overlap(
        "Sarah Johnson presented the quarterly results.", context3
    )
    score_bad, details_bad = compute_context_overlap(
        "John Smith presented the quarterly results.", context3
    )

    print(f"\nContext: '{context3}'")
    print(f"Response 'Sarah Johnson...': Score = {score_ok:.2f}, Violations = {details_ok['violations']}")
    print(f"Response 'John Smith...':   Score = {score_bad:.2f}, Violations = {details_bad['violations']}")
    print(f"Separation: {score_ok - score_bad:.2f} (target: > 0)")


def test_full_hybrid():
    """Test the full Hybrid Controller with model."""
    print("\n" + "=" * 70)
    print("FULL HYBRID CONTROLLER TEST (With Llama Model)")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Initialize AG-SAR with Hybrid Controller
    config = AGSARConfig(
        enable_hybrid_controller=True,
        enable_jepa_monitor=False,  # Disable JEPA predictor
        enable_intrinsic_detection=False,  # Disable Truth Vector for this test
        symbolic_weight=0.6,
        jepa_weight=0.4,
    )
    engine = AGSAR(model, tokenizer, config)

    # Test 1: Paris vs London
    print("\n" + "-" * 50)
    print("TEST 1: Standard Facts (Capital of France)")
    print("-" * 50)

    context1 = "Geography Facts: The capital of France is Paris. It is known for the Eiffel Tower."
    query1 = "\nQuestion: What is the capital of France?\nAnswer: The capital of France is"
    prompt1 = context1 + query1

    result1a = engine.compute_hybrid_trust(prompt1, " Paris.")
    result1b = engine.compute_hybrid_trust(prompt1, " London.")

    print(f"\nContext: '{context1[:50]}...'")
    print(f"Response: 'Paris' (Correct)")
    print(f"  Trust Score:    {result1a['trust_score']:.4f}")
    print(f"  Neural Trust:   {result1a['neural_trust']:.4f}")
    print(f"  Symbolic Score: {result1a['symbolic_score']:.4f}")
    print(f"  VETO:           {result1a['veto_triggered']}")
    print(f"  Hallucination:  {result1a['is_hallucination']}")

    print(f"\nResponse: 'London' (Wrong - SHOULD TRIGGER VETO)")
    print(f"  Trust Score:    {result1b['trust_score']:.4f}")
    print(f"  Neural Trust:   {result1b['neural_trust']:.4f}")
    print(f"  Symbolic Score: {result1b['symbolic_score']:.4f}")
    print(f"  VETO:           {result1b['veto_triggered']}")
    print(f"  Hallucination:  {result1b['is_hallucination']}")

    sep1 = result1a['trust_score'] - result1b['trust_score']
    print(f"\nTrust Separation (Paris - London): {sep1:.4f}")
    print(f"VETO WORKING: {result1b['veto_triggered'] and not result1a['veto_triggered']}")

    # Test 2: Fabricated Facts
    print("\n" + "-" * 50)
    print("TEST 2: Fabricated Facts (Watermelon Capital)")
    print("-" * 50)

    context2 = "Alternative Geography: In this universe, the capital of France is Watermelon."
    query2 = "\nQuestion: What is the capital of France?\nAnswer: The capital of France is"
    prompt2 = context2 + query2

    result2a = engine.compute_hybrid_trust(prompt2, " Watermelon.")
    result2b = engine.compute_hybrid_trust(prompt2, " Paris.")

    print(f"\nContext: '{context2[:60]}...'")
    print(f"Response: 'Watermelon' (Context-Consistent)")
    print(f"  Trust Score:    {result2a['trust_score']:.4f}")
    print(f"  Neural Trust:   {result2a['neural_trust']:.4f}")
    print(f"  Symbolic Score: {result2a['symbolic_score']:.4f}")
    print(f"  VETO:           {result2a['veto_triggered']}")
    print(f"  Hallucination:  {result2a['is_hallucination']}")

    print(f"\nResponse: 'Paris' (Context-Violation - SHOULD TRIGGER VETO)")
    print(f"  Trust Score:    {result2b['trust_score']:.4f}")
    print(f"  Neural Trust:   {result2b['neural_trust']:.4f}")
    print(f"  Symbolic Score: {result2b['symbolic_score']:.4f}")
    print(f"  VETO:           {result2b['veto_triggered']}")
    print(f"  Hallucination:  {result2b['is_hallucination']}")

    sep2 = result2a['trust_score'] - result2b['trust_score']
    print(f"\nTrust Separation (Watermelon - Paris): {sep2:.4f}")
    print(f"VETO WORKING: {result2b['veto_triggered'] and not result2a['veto_triggered']}")

    # Test 3: RAG Numbers
    print("\n" + "-" * 50)
    print("TEST 3: RAG-Style Numbers")
    print("-" * 50)

    context3 = """
Retrieved Documents:
[Doc 1] Company Report 2024: TechCorp announced revenue of $5.2 billion in Q3.
[Doc 2] The CEO of TechCorp is Sarah Johnson, appointed in 2022.

Question: What was TechCorp's Q3 revenue?
Answer: According to the documents, TechCorp's Q3 revenue was"""

    result3a = engine.compute_hybrid_trust(context3, " $5.2 billion.")
    result3b = engine.compute_hybrid_trust(context3, " $8.7 billion.")

    print(f"Response: '$5.2 billion' (Correct)")
    print(f"  Trust Score:    {result3a['trust_score']:.4f}")
    print(f"  Neural Trust:   {result3a['neural_trust']:.4f}")
    print(f"  Numeric Score:  {result3a['numeric_score']:.4f}")
    print(f"  VETO:           {result3a['veto_triggered']}")
    print(f"  Hallucination:  {result3a['is_hallucination']}")

    print(f"\nResponse: '$8.7 billion' (Hallucinated - SHOULD TRIGGER VETO)")
    print(f"  Trust Score:    {result3b['trust_score']:.4f}")
    print(f"  Neural Trust:   {result3b['neural_trust']:.4f}")
    print(f"  Numeric Score:  {result3b['numeric_score']:.4f}")
    print(f"  VETO:           {result3b['veto_triggered']}")
    print(f"  Hallucination:  {result3b['is_hallucination']}")

    sep3 = result3a['trust_score'] - result3b['trust_score']
    print(f"\nTrust Separation (Correct - Wrong): {sep3:.4f}")
    print(f"VETO WORKING: {result3b['veto_triggered'] and not result3a['veto_triggered']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: UNIVERSAL VETO ENGINE")
    print("=" * 70)

    # Check veto status
    veto1 = result1b['veto_triggered'] and not result1a['veto_triggered']
    veto2 = result2b['veto_triggered'] and not result2a['veto_triggered']
    veto3 = result3b['veto_triggered'] and not result3a['veto_triggered']

    print(f"\nTest 1 (Paris vs London):     Sep = {sep1:.4f}, VETO = {veto1}")
    print(f"Test 2 (Watermelon vs Paris): Sep = {sep2:.4f}, VETO = {veto2}")
    print(f"Test 3 ($5.2B vs $8.7B):      Sep = {sep3:.4f}, VETO = {veto3}")

    # Calculate average separation
    avg_sep = (sep1 + sep2 + sep3) / 3
    veto_count = sum([veto1, veto2, veto3])

    print(f"\nAverage Trust Separation: {avg_sep:.4f}")
    print(f"Veto Triggers (correct): {veto_count}/3")

    if veto_count == 3 and avg_sep > 0.3:
        print(f"\n[PASS] Universal Veto Engine is working!")
        print(f"       All hallucinations triggered VETO.")
        print(f"       Average separation: {avg_sep:.4f}")
    elif veto_count >= 2:
        print(f"\n[PARTIAL] Most cases caught by VETO ({veto_count}/3).")
        print(f"          Review symbolic detection for edge cases.")
    else:
        print(f"\n[FAIL] Veto system needs adjustment ({veto_count}/3).")
        print(f"       Check symbolic overlap logic.")

    # Cleanup
    engine.cleanup()


def main():
    # First test symbolic functions directly (no GPU needed)
    test_symbolic_only()

    # Then test full hybrid with model
    test_full_hybrid()


if __name__ == "__main__":
    main()
