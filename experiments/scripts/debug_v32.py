#!/usr/bin/env python3
"""
Debug v3.2 components to understand signal directions.
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
    """Extract all v3.2 component values for analysis."""
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.measures.entropy import compute_varentropy, compute_token_entropy, compute_epiplexity_absolute
    from ag_sar.ops import compute_authority_flow_recursive, compute_structural_gain

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

    # 1. Absolute Epiplexity
    varentropy = compute_varentropy(logits, attention_mask)
    E_t = compute_epiplexity_absolute(varentropy, tau_var=3.0)

    # 2. Structural Gain
    gamma_t = compute_structural_gain(E_t, lambda_struct=2.0)

    # 3. Authority Flow
    A_t = compute_authority_flow_recursive(attn, response_start, gamma_t, attention_mask)

    # 4. Absolute Gate
    cos_sim = F.cosine_similarity(h_attn, h_block, dim=-1)
    G_t = 0.5 * (1.0 + cos_sim)

    # 5. Semantic Consistency
    D_t = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    consistency = (1.0 - D_t * 2.0).clamp(0.0, 1.0)

    # 6. Master Equation
    score = G_t * A_t + (1.0 - G_t) * consistency

    # 7. Emergence Override
    is_emergent = (G_t < 0.5) & (E_t > 0.7)
    score_override = torch.where(is_emergent, consistency, score)

    # 8. Loop Safety
    entropy = compute_token_entropy(logits, attention_mask)
    final_score = torch.where(entropy < 0.01, torch.zeros_like(score_override), score_override)

    # Extract response-only values
    resp_start = response_start
    return {
        'varentropy': varentropy[:, resp_start:].mean().item(),
        'epiplexity': E_t[:, resp_start:].mean().item(),
        'gamma': gamma_t[:, resp_start:].mean().item(),
        'authority_flow': A_t[:, resp_start:].mean().item(),
        'cos_sim': cos_sim[:, resp_start:].mean().item(),
        'gate': G_t[:, resp_start:].mean().item(),
        'dispersion': D_t[:, resp_start:].mean().item(),
        'consistency': consistency[:, resp_start:].mean().item(),
        'score_before_override': score[:, resp_start:].mean().item(),
        'score_after_override': score_override[:, resp_start:].mean().item(),
        'final_score': final_score[:, resp_start:].mean().item(),
        'uncertainty': (1.0 - final_score[:, resp_start:]).mean().item(),
        'emergence_rate': is_emergent[:, resp_start:].float().mean().item(),
    }


def main():
    print("=" * 70)
    print("AG-SAR v3.2 Component Debug")
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

    # Initialize AG-SAR v3.2
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
    print("COMPONENT ANALYSIS - Mean Values by Label")
    print("=" * 70)
    print(f"\n{'Component':<22} {'Hallucinated':>14} {'Factual':>14} {'Direction':>12} {'Expected':>10}")
    print("-" * 75)

    expected = {
        'varentropy': 'H > F',
        'epiplexity': 'H > F',
        'gamma': 'H > F',
        'authority_flow': 'H < F',
        'cos_sim': 'H < F',
        'gate': 'H < F',
        'dispersion': 'H > F',
        'consistency': 'H < F',
        'score_before_override': 'H < F',
        'score_after_override': 'H < F',
        'final_score': 'H < F',
        'uncertainty': 'H > F',
        'emergence_rate': '???',
    }

    inversions = []
    for key in ['varentropy', 'epiplexity', 'gamma', 'authority_flow', 'cos_sim',
                'gate', 'dispersion', 'consistency', 'score_before_override',
                'score_after_override', 'final_score', 'uncertainty', 'emergence_rate']:
        hall_mean = np.mean(hall_signals[key]) if hall_signals[key] else 0
        fact_mean = np.mean(fact_signals[key]) if fact_signals[key] else 0

        direction = "H > F" if hall_mean > fact_mean else "H < F"
        exp = expected.get(key, '?')
        match = "OK" if direction == exp or exp == '???' else "INVERTED"

        if match == "INVERTED":
            inversions.append(key)

        print(f"{key:<22} {hall_mean:>14.4f} {fact_mean:>14.4f} {direction:>12} {exp:>10} {match}")

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
