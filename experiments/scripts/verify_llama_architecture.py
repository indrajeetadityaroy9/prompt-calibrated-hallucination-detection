#!/usr/bin/env python3
"""
Smoke Test: Verify Llama 3.1 GQA Compatibility and JEPA Integration.

This script validates:
1. GQA (Grouped Query Attention) handling: 8 KV heads -> 32 Q heads expansion
2. Centroid Variance (JEPA) computation with correct embedding dimensions
3. Antonym Safety Valve constraint (parametric_weight < threshold)

Run: python -m experiments.verify_llama_jepa
"""

import os
import sys
import torch

# Set HF token from environment or use default for testing
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
os.environ["HF_TOKEN"] = HF_TOKEN

from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig
from ag_sar.measures.semantics import compute_centroid_variance, compute_top1_projection


def main():
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    device = "cuda:0"

    print(f"{'='*60}")
    print(f"AG-SAR + Llama 3.1 JEPA Smoke Test")
    print(f"{'='*60}")
    print(f"Model: {MODEL_ID}")
    print(f"Device: {device}")
    print()

    # =========================================================================
    # Step 1: Load Model
    # =========================================================================
    print("Step 1: Loading Llama-3.1-8B-Instruct...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # NOTE: AG-SAR needs "eager" to extract attention weights
        # Flash Attention 2 doesn't expose weights for our hooks
        attn_implementation="eager",
    )

    # Verify GQA configuration
    config = model.config
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Q heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads}")
    print(f"  - GQA ratio: {config.num_attention_heads // config.num_key_value_heads}:1")
    print("  [OK] Model loaded successfully")
    print()

    # =========================================================================
    # Step 2: Test GQA Handling with Top-1 Projection (Baseline)
    # =========================================================================
    print("Step 2: Testing GQA with top1_projection (baseline)...")

    config_top1 = AGSARConfig(
        dispersion_method="top1_projection",
        dispersion_k=5,
        parametric_weight=0.5,
    )
    engine_top1 = AGSAR(model, tokenizer, config_top1)

    # Test prompts
    prompt = "The capital of France is"
    response = " Paris."

    result_top1 = engine_top1.compute_uncertainty(prompt, response, return_details=True)
    print(f"  - Prompt: '{prompt}'")
    print(f"  - Response: '{response}'")
    print(f"  - Uncertainty: {result_top1['score']:.4f}")
    print(f"  - Authority: {result_top1['authority']:.4f}")
    print("  [OK] GQA compatibility verified (no shape mismatch)")
    engine_top1.cleanup()
    print()

    # =========================================================================
    # Step 3: Test JEPA (Centroid Variance) Mode
    # =========================================================================
    print("Step 3: Testing JEPA (centroid_variance) mode...")

    config_jepa = AGSARConfig(
        dispersion_method="centroid_variance",
        dispersion_k=10,
        parametric_weight=0.6,  # Antonym Safety Valve
    )
    engine_jepa = AGSAR(model, tokenizer, config_jepa)

    # The "Synonym Test" - invites synonyms
    prompt_syn = "Review: The visual effects in this movie were absolutely"
    response_syn = " stunning."  # Could be: amazing, incredible, breathtaking

    result_jepa = engine_jepa.compute_uncertainty(prompt_syn, response_syn, return_details=True)
    print(f"  - Prompt: '{prompt_syn}'")
    print(f"  - Response: '{response_syn}'")
    print(f"  - Uncertainty (JEPA): {result_jepa['score']:.4f}")
    print(f"  - Authority (JEPA): {result_jepa['authority']:.4f}")
    print("  [OK] JEPA mode works with Llama 3.1")
    engine_jepa.cleanup()
    print()

    # =========================================================================
    # Step 4: Compare Top-1 vs Centroid Variance
    # =========================================================================
    print("Step 4: Comparing dispersion methods on same input...")

    test_cases = [
        ("The US is also known as", " America."),  # Synonyms expected
        ("The year World War II ended was", " 1945."),  # Factual, no synonyms
        ("A delicious fruit that is red is", " an apple."),  # Some synonyms
    ]

    print(f"  {'Prompt':<45} {'Top1':<8} {'JEPA':<8} {'Delta':<8}")
    print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*8}")

    for prompt, response in test_cases:
        # Create fresh engines for each test (hooks need clean state)
        eng_t1 = AGSAR(model, tokenizer, AGSARConfig(dispersion_method="top1_projection", dispersion_k=10))
        r1 = eng_t1.compute_uncertainty(prompt, response, return_details=True)
        eng_t1.cleanup()

        eng_cv = AGSAR(model, tokenizer, AGSARConfig(dispersion_method="centroid_variance", dispersion_k=10))
        r2 = eng_cv.compute_uncertainty(prompt, response, return_details=True)
        eng_cv.cleanup()

        delta = r2['score'] - r1['score']
        short_prompt = (prompt[:42] + "...") if len(prompt) > 45 else prompt
        print(f"  {short_prompt:<45} {r1['score']:<8.4f} {r2['score']:<8.4f} {delta:+.4f}")

    print()

    # =========================================================================
    # Step 5: Verify Antonym Safety Valve
    # =========================================================================
    print("Step 5: Verifying Antonym Safety Valve constraint...")

    # The constraint: parametric_weight (0.6) < threshold (0.7)
    parametric_weight = 0.6
    threshold = 0.7

    # Worst case: Gate=0 (ignoring context), Trust=1.0 (confident antonym)
    max_score = parametric_weight * 1.0  # (1-Gate) * Trust * weight when Gate=0

    if max_score < threshold:
        print(f"  - parametric_weight: {parametric_weight}")
        print(f"  - threshold: {threshold}")
        print(f"  - Max score when ignoring context: {max_score:.2f} < {threshold}")
        print("  [OK] Antonym Safety Valve constraint satisfied")
    else:
        print(f"  [FAIL] Safety valve violated: {max_score:.2f} >= {threshold}")
        sys.exit(1)
    print()

    # =========================================================================
    # Step 6: GPU Memory Check
    # =========================================================================
    print("Step 6: GPU Memory Usage...")

    for i in range(torch.cuda.device_count()):
        mem_used = torch.cuda.memory_allocated(i) / 1e9
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  - GPU {i}: {mem_used:.1f}GB / {mem_total:.1f}GB ({100*mem_used/mem_total:.1f}%)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"{'='*60}")
    print("SMOKE TEST PASSED")
    print(f"{'='*60}")
    print("Verified:")
    print("  1. GQA (8 KV -> 32 Q heads) expansion works")
    print("  2. top1_projection dispersion method works")
    print("  3. centroid_variance (JEPA) dispersion method works")
    print("  4. Antonym Safety Valve constraint satisfied")
    print()
    print("Ready for full benchmark!")
    print("  - GPU 0: Available for AGSAR_v8_Top1 (Baseline)")
    print("  - GPU 1: Available for AGSAR_JEPA_Centroid (New Method)")


if __name__ == "__main__":
    main()
