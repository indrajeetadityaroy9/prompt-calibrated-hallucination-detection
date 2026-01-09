#!/usr/bin/env python3
"""
Diagnostic script to trace AG-SAR component signals and identify inversion sources.

This script computes each component individually and tracks:
1. Authority Flow (recursive attention from prompt)
2. Gate (MLP-attention agreement)
3. Trust (dispersion + varentropy-based)
4. Epiplexity (normalized varentropy)
5. Dispersion (semantic consistency)
6. Varentropy (confidence stability)
7. Final Score

For each component, we compare hallucinated vs factual samples to identify
which component(s) are inverted from expected direction.
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
from experiments.data import HaluEvalDataset


def compute_all_components(agsar, prompt, response):
    """
    Compute all AG-SAR components individually for debugging.

    Returns dict with all intermediate signals.
    """
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.measures.entropy import compute_varentropy, compute_token_entropy, compute_epiplexity
    from ag_sar.ops import (
        compute_authority_flow_vectorized,
        compute_authority_flow_recursive,
        fused_stability_gate,
        compute_mlp_divergence,
    )

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
    h_attn = h_attn.to(agsar._compute_device) if h_attn is not None else None
    h_block = h_block.to(agsar._compute_device) if h_block is not None else None
    logits = model_output.logits.to(agsar._compute_device)
    embed_matrix = agsar._embed_matrix.to(agsar._compute_device)

    # Calibration
    calib = agsar._calibration or {}

    # ===== COMPONENT 1: Authority Flow (vectorized) =====
    authority_flow = compute_authority_flow_vectorized(attn, response_start, attention_mask)

    # ===== COMPONENT 2: Raw Divergence =====
    cos_sim = F.cosine_similarity(h_attn, h_block, dim=-1)
    raw_divergence = 1.0 - cos_sim

    # ===== COMPONENT 3: Gate (Z-score based) =====
    div_mu = calib.get('divergence_mu', 0.5)
    div_sigma = calib.get('divergence_sigma', 0.1)
    z_score = (raw_divergence - div_mu) / (div_sigma + 1e-8)
    gate = torch.sigmoid(-2.0 * z_score)
    gate = torch.where(z_score < -1.0, torch.ones_like(gate), gate)

    # ===== COMPONENT 4: Dispersion =====
    dispersion = compute_semantic_dispersion(logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95)

    # ===== COMPONENT 5: Varentropy =====
    varentropy = compute_varentropy(logits, attention_mask)

    # ===== COMPONENT 6: Epiplexity =====
    epi_mu = calib.get('epiplexity_mu', 0.0)
    epi_sigma = calib.get('epiplexity_sigma', 1.0)
    epiplexity = compute_epiplexity(varentropy, epi_mu, epi_sigma)

    # ===== COMPONENT 7: Token Entropy =====
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)

    # ===== COMPONENT 8: Trust (base) =====
    base_trust = (1.0 - dispersion).clamp(0.0, 1.0)

    # ===== COMPONENT 9: Conflict Detection =====
    is_conflict = gate < 0.5
    is_emergent = epiplexity > 0.8

    # ===== COMPONENT 10: Emergence-Gated Trust =====
    trust = base_trust.clone()
    trust = torch.where(is_conflict & (~is_emergent), torch.zeros_like(trust), trust)
    trust = torch.where(is_emergent, torch.ones_like(trust), trust)

    # ===== COMPONENT 11: Final Score =====
    final_score = gate * authority_flow + (1.0 - gate) * trust

    # Extract response tokens only
    response_flow = authority_flow[:, response_start:].mean().item()
    response_gate = gate[:, response_start:].mean().item()
    response_dispersion = dispersion[:, response_start:].mean().item()
    response_varentropy = varentropy[:, response_start:].mean().item()
    response_epiplexity = epiplexity[:, response_start:].mean().item()
    response_entropy = entropy[:, response_start:].mean().item()
    response_trust = trust[:, response_start:].mean().item()
    response_divergence = raw_divergence[:, response_start:].mean().item()
    response_score = final_score[:, response_start:].mean().item()

    # Conflict/emergence stats
    conflict_rate = is_conflict[:, response_start:].float().mean().item()
    emergence_rate = is_emergent[:, response_start:].float().mean().item()

    return {
        'authority_flow': response_flow,
        'gate': response_gate,
        'divergence': response_divergence,
        'dispersion': response_dispersion,
        'varentropy': response_varentropy,
        'epiplexity': response_epiplexity,
        'entropy': response_entropy,
        'base_trust': (1.0 - response_dispersion),
        'trust': response_trust,
        'score': response_score,
        'uncertainty': 1.0 - response_score,
        'conflict_rate': conflict_rate,
        'emergence_rate': emergence_rate,
    }


def main():
    print("=" * 70)
    print("AG-SAR Component Diagnostic - Tracing Inversion Sources")
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

    # Initialize AG-SAR v3
    config = AGSARConfig(version=3, semantic_layers=4)
    agsar = AGSAR(model, tokenizer, config)

    # Load HaluEval QA dataset
    print("\nLoading HaluEval QA dataset...")
    dataset = HaluEvalDataset(num_samples=100, seed=42, variant="qa")
    dataset.load()

    stats = dataset.get_statistics()
    print(f"Samples: {stats['total_samples']} (Hall: {stats['hallucinated']}, Fact: {stats['factual']})")

    # Collect component signals by label
    hall_signals = defaultdict(list)
    fact_signals = defaultdict(list)

    # Also collect AG-SAR API outputs for comparison
    agsar_hall_scores = []
    agsar_fact_scores = []

    print("\nComputing all components for each sample...")
    for sample in tqdm(dataset, desc="Analyzing samples"):
        try:
            # Calibrate on prompt
            agsar.calibrate_on_prompt(sample.prompt)

            # Get all component signals (our diagnostic)
            signals = compute_all_components(agsar, sample.prompt, sample.response)

            # Also get the official AG-SAR uncertainty score
            official_uncertainty = agsar.compute_uncertainty(sample.prompt, sample.response)

            target = hall_signals if sample.label == 1 else fact_signals
            for key, value in signals.items():
                target[key].append(value)

            # Track official scores
            if sample.label == 1:
                agsar_hall_scores.append(official_uncertainty)
            else:
                agsar_fact_scores.append(official_uncertainty)

        except Exception as e:
            print(f"Error: {e}")
            continue

    # Compute statistics
    print("\n" + "=" * 70)
    print("COMPONENT ANALYSIS - Mean Values by Label")
    print("=" * 70)
    print(f"\n{'Component':<20} {'Hallucinated':>15} {'Factual':>15} {'Direction':>12} {'Expected':>12}")
    print("-" * 75)

    # Expected directions based on AG-SAR theory:
    # - authority_flow: Hall < Fact (hallucinations ignore context)
    # - gate: Hall < Fact (hallucinations override attention)
    # - dispersion: Hall > Fact (hallucinations are semantically confused)
    # - varentropy: Hall > Fact (hallucinations have unstable confidence)
    # - trust: Hall < Fact (hallucinations have low trust)
    # - score: Hall < Fact (hallucinations have low authority)
    # - uncertainty: Hall > Fact (hallucinations have high uncertainty)

    expected = {
        'authority_flow': 'H < F',
        'gate': 'H < F',
        'divergence': 'H > F',
        'dispersion': 'H > F',
        'varentropy': 'H > F',
        'epiplexity': 'H > F',
        'entropy': 'H > F',
        'base_trust': 'H < F',
        'trust': 'H < F',
        'score': 'H < F',
        'uncertainty': 'H > F',
        'conflict_rate': 'H > F',
        'emergence_rate': 'H < F',
    }

    inversions = []

    for key in ['authority_flow', 'gate', 'divergence', 'dispersion', 'varentropy',
                'epiplexity', 'entropy', 'base_trust', 'trust', 'score', 'uncertainty',
                'conflict_rate', 'emergence_rate']:
        hall_mean = np.mean(hall_signals[key])
        fact_mean = np.mean(fact_signals[key])

        if hall_mean > fact_mean:
            direction = "H > F"
        else:
            direction = "H < F"

        exp = expected.get(key, '?')
        match = "✓" if direction == exp else "✗ INVERTED"

        if direction != exp:
            inversions.append(key)

        print(f"{key:<20} {hall_mean:>15.4f} {fact_mean:>15.4f} {direction:>12} {exp:>8} {match}")

    # Summary
    print("\n" + "=" * 70)
    print("INVERSION SUMMARY")
    print("=" * 70)

    if inversions:
        print(f"\nInverted components ({len(inversions)}):")
        for inv in inversions:
            print(f"  - {inv}")

        print("\nDiagnosis:")
        if 'authority_flow' in inversions:
            print("  * CRITICAL: Authority flow is inverted!")
            print("    This means hallucinations ATTEND to context MORE than factual.")
            print("    AG-SAR's core assumption (hallucinations ignore context) is violated.")
            print("    Possible cause: HaluEval hallucinations are 'reasoning errors' not 'context ignoring'")
    else:
        print("\nNo inversions detected! All components behave as expected.")

    # Additional analysis: correlation between components
    print("\n" + "=" * 70)
    print("COMPONENT CORRELATIONS WITH LABEL")
    print("=" * 70)

    all_signals = {k: hall_signals[k] + fact_signals[k] for k in hall_signals.keys()}
    all_labels = [1] * len(hall_signals['score']) + [0] * len(fact_signals['score'])

    print(f"\n{'Component':<20} {'Correlation':>12} {'Direction':>15}")
    print("-" * 50)

    for key in ['authority_flow', 'gate', 'dispersion', 'trust', 'score', 'uncertainty']:
        corr = np.corrcoef(all_signals[key], all_labels)[0, 1]
        direction = "Hall++" if corr > 0 else "Factual++"
        print(f"{key:<20} {corr:>12.4f} {direction:>15}")

    # Compute actual AUROC from token-mean scores
    from sklearn.metrics import roc_auc_score

    print("\n" + "=" * 70)
    print("AUROC FROM COMPONENT MEANS")
    print("=" * 70)
    print("\nThese are AUROCs from using token-mean values directly:")

    for key in ['authority_flow', 'gate', 'dispersion', 'trust', 'score', 'uncertainty']:
        try:
            auroc = roc_auc_score(all_labels, all_signals[key])
            # For metrics where higher = factual, invert
            if key in ['authority_flow', 'gate', 'trust', 'score']:
                auroc = 1 - auroc  # Invert because higher score = factual
            print(f"  {key:<20}: AUROC = {auroc:.4f}")
        except Exception as e:
            print(f"  {key:<20}: Error - {e}")

    # Compare with official AG-SAR output
    print("\n" + "=" * 70)
    print("OFFICIAL AG-SAR API OUTPUT COMPARISON")
    print("=" * 70)

    print(f"\nDiagnostic uncertainty (token-mean):")
    print(f"  Hallucinated mean: {np.mean(hall_signals['uncertainty']):.4f}")
    print(f"  Factual mean:      {np.mean(fact_signals['uncertainty']):.4f}")

    print(f"\nOfficial AG-SAR uncertainty (aggregated):")
    print(f"  Hallucinated mean: {np.mean(agsar_hall_scores):.4f}")
    print(f"  Factual mean:      {np.mean(agsar_fact_scores):.4f}")

    # AUROC comparison
    all_official = agsar_hall_scores + agsar_fact_scores
    official_auroc = roc_auc_score(all_labels, all_official)
    diagnostic_auroc = roc_auc_score(all_labels, all_signals['uncertainty'])

    print(f"\nAUROC comparison:")
    print(f"  Diagnostic (token-mean uncertainty):  {diagnostic_auroc:.4f}")
    print(f"  Official AG-SAR (aggregated):         {official_auroc:.4f}")

    if official_auroc < diagnostic_auroc:
        print(f"\n  ISSUE: Official AUROC is {diagnostic_auroc - official_auroc:.4f} lower than diagnostic!")
        print("  This suggests the aggregation step is HURTING performance.")
    else:
        print(f"\n  OK: Official AUROC is {official_auroc - diagnostic_auroc:.4f} higher than diagnostic")

    # Cleanup
    agsar.cleanup()

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
