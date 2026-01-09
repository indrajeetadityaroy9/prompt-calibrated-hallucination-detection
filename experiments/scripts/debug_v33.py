#!/usr/bin/env python3
"""
Debug v3.3 components to understand signal directions with complexity ratio.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from experiments.preflight import check_installation
check_installation()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig


def analyze_components(agsar, prompt, response):
    """Extract all v3.3 component values for analysis."""
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.measures.entropy import compute_varentropy, compute_token_entropy, compute_epiplexity
    from ag_sar.ops import compute_authority_flow_recursive

    # First calibrate on prompt
    agsar.calibrate_on_prompt(prompt)
    calibration = agsar._calibration

    # Tokenize
    input_ids, attention_mask, response_start = agsar._tokenize(prompt, response)
    input_ids = input_ids.to(agsar._compute_device)
    attention_mask = attention_mask.to(agsar._compute_device)

    # Forward pass
    Q_stack, K_stack, _, model_output = agsar._adapter.extract(input_ids, attention_mask)

    # Get captured states
    last_layer = agsar._semantic_layer_indices[-1]
    attn = agsar._adapter.capture.attention_weights.get(last_layer)
    h_attn = agsar._adapter.capture.attn_outputs.get(last_layer)
    h_block = agsar._adapter.capture.block_outputs.get(last_layer)

    if attn is None:
        raise RuntimeError("Attention weights not captured")

    attn = attn.to(agsar._compute_device)
    h_attn = h_attn.to(agsar._compute_device)
    h_block = h_block.to(agsar._compute_device)
    logits = model_output.logits.to(agsar._compute_device)
    embed_matrix = agsar._embed_matrix.to(agsar._compute_device)

    # 1. Varentropy (absolute signal)
    varentropy = compute_varentropy(logits, attention_mask)

    # 2. Absolute Cognitive Load (τ = 5.0)
    tau = 5.0
    E_t = compute_epiplexity(varentropy, tau=tau)

    # 3. Authority Flow (no structural gain)
    gamma_t = torch.ones_like(E_t)
    A_t = compute_authority_flow_recursive(attn, response_start, gamma_t, attention_mask)

    # 4. Semantic Consistency
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    consistency = (1.0 - D_t).clamp(0.0, 1.0)

    # 5. Epistemic Weight (Sigmoid-Centered on Varentropy)
    lambda_struct = 2.0
    centered_varentropy = varentropy - tau
    epistemic_weight = torch.sigmoid(centered_varentropy * lambda_struct)

    # 6. Master Equation: Score = A × (1-D) × E^λ
    score = A_t * consistency * epistemic_weight

    # Extract response-only values
    resp_start = response_start
    return {
        'tau': tau,
        'varentropy_response': varentropy[:, resp_start:].mean().item(),
        'epiplexity_ratio': E_t[:, resp_start:].mean().item(),
        'epistemic_weight': epistemic_weight[:, resp_start:].mean().item(),
        'authority_flow': A_t[:, resp_start:].mean().item(),
        'dispersion': D_t[:, resp_start:].mean().item(),
        'consistency': consistency[:, resp_start:].mean().item(),
        'final_score': score[:, resp_start:].mean().item(),
        'uncertainty': (1.0 - score[:, resp_start:]).mean().item(),
    }


def main():
    print("=" * 70)
    print("AG-SAR v3.3 Component Debug - Complexity Ratio Analysis")
    print("=" * 70)

    # Load model
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize AG-SAR v3.3
    config = AGSARConfig()
    agsar = AGSAR(model, tokenizer, config)

    # Load HaluEval QA
    from experiments.data import HaluEvalDataset
    dataset = HaluEvalDataset(num_samples=50, seed=42, variant="qa")
    dataset.load()

    # Collect signals by label
    hall_signals = defaultdict(list)
    fact_signals = defaultdict(list)

    for sample in tqdm(dataset, desc="Analyzing samples"):
        try:
            signals = analyze_components(agsar, sample.prompt, sample.response)
            target = hall_signals if sample.label == 1 else fact_signals
            for key, value in signals.items():
                target[key].append(value)
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Print comparison
    print("\n" + "=" * 70)
    print("v3.3 COMPONENT ANALYSIS - Mean Values by Label")
    print("=" * 70)
    print(f"\n{'Component':<22} {'Hallucinated':>14} {'Factual':>14} {'Direction':>12} {'Expected':>10}")
    print("-" * 75)

    expected = {
        'tau': '???',                    # Universal constant (5.0)
        'varentropy_response': 'H < F',  # Hall has low cognitive load
        'epiplexity_ratio': 'H < F',     # Hall: V_resp / τ < 1.0
        'epistemic_weight': 'H < F',     # E^4 crushes for Hall
        'authority_flow': 'H < F',       # Ideally, but inverted in data
        'dispersion': 'H > F',           # Ideally, but inverted in data
        'consistency': 'H < F',          # Ideally, but inverted in data
        'final_score': 'H < F',          # Critical!
        'uncertainty': 'H > F',          # Critical!
    }

    inversions = []
    for key in ['tau', 'varentropy_response', 'epiplexity_ratio',
                'epistemic_weight', 'authority_flow', 'dispersion', 'consistency',
                'final_score', 'uncertainty']:
        hall_mean = np.mean(hall_signals[key]) if hall_signals[key] else 0
        fact_mean = np.mean(fact_signals[key]) if fact_signals[key] else 0

        direction = "H > F" if hall_mean > fact_mean else "H < F"
        exp = expected.get(key, '?')
        match = "OK" if direction == exp or exp == '???' else "INVERTED"

        if match == "INVERTED":
            inversions.append(key)

        print(f"{key:<22} {hall_mean:>14.4f} {fact_mean:>14.4f} {direction:>12} {exp:>10} {match}")

    # Key insight: Check if E^λ is doing its job
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Is E^λ compensating for inverted A and (1-D)?")
    print("=" * 70)

    hall_A = np.mean(hall_signals['authority_flow'])
    fact_A = np.mean(fact_signals['authority_flow'])
    hall_C = np.mean(hall_signals['consistency'])
    fact_C = np.mean(fact_signals['consistency'])
    hall_E2 = np.mean(hall_signals['epistemic_weight'])
    fact_E2 = np.mean(fact_signals['epistemic_weight'])

    print(f"\n  A × (1-D) for Hall: {hall_A:.4f} × {hall_C:.4f} = {hall_A * hall_C:.4f}")
    print(f"  A × (1-D) for Fact: {fact_A:.4f} × {fact_C:.4f} = {fact_A * fact_C:.4f}")
    print(f"  → Without E^λ: {'Hall > Fact' if hall_A * hall_C > fact_A * fact_C else 'Fact > Hall'}")
    print(f"\n  E^λ for Hall: {hall_E2:.4f}")
    print(f"  E^λ for Fact: {fact_E2:.4f}")
    print(f"\n  Final Score Hall: {hall_A * hall_C * hall_E2:.4f}")
    print(f"  Final Score Fact: {fact_A * fact_C * fact_E2:.4f}")
    print(f"  → With E^λ: {'Hall > Fact (PROBLEM!)' if hall_A * hall_C * hall_E2 > fact_A * fact_C * fact_E2 else 'Fact > Hall (GOOD!)'}")

    if inversions:
        print(f"\n INVERTED SIGNALS: {inversions}")
    else:
        print("\n All signals behave as expected!")

    agsar.cleanup()

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
