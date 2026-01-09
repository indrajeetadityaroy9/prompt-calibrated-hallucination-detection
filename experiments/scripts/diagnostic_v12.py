#!/usr/bin/env python3
"""
Diagnostic script to analyze v12 signal values and identify bugs.

Checks:
1. Varentropy and dispersion values for facts vs halls
2. JSD signal strength
3. d_signal (dispersion shield) values
4. Effect of threshold choices
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig


def analyze_signals(agsar, prompt, response, is_hall):
    """Analyze raw signal values for a single sample."""
    from ag_sar.measures.entropy import compute_varentropy
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.ops import compute_logit_divergence_jsd

    # Get inputs
    input_ids, attention_mask, response_start = agsar._tokenize(prompt, response)
    input_ids = input_ids.to(agsar._compute_device)
    attention_mask = attention_mask.to(agsar._compute_device)

    # Forward pass
    Q_stack, K_stack, _, model_output = agsar._adapter.extract(input_ids, attention_mask)

    last_layer = agsar._semantic_layer_indices[-1]
    h_attn = agsar._adapter.capture.attn_outputs.get(last_layer).to(agsar._compute_device)
    h_block = agsar._adapter.capture.block_outputs.get(last_layer).to(agsar._compute_device)
    logits = model_output.logits.to(agsar._compute_device)
    embed_matrix = agsar._embed_matrix.to(agsar._compute_device)

    # Slice to response
    h_attn_resp = h_attn[:, response_start:, :]
    h_block_resp = h_block[:, response_start:, :]
    logits_resp = logits[:, response_start:, :].contiguous()

    # Compute raw signals
    varentropy = compute_varentropy(logits_resp, attention_mask=None)
    D_t = compute_semantic_dispersion(
        logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    jsd = compute_logit_divergence_jsd(
        h_attn_resp, h_block_resp, embed_matrix, top_k=50
    )

    # Compute derived signals (as in v12)
    varentropy_scale = 3.0
    v_signal = torch.tanh(varentropy / varentropy_scale)

    # Test multiple dispersion thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    d_signals = {}
    for thresh in thresholds:
        d_signals[thresh] = torch.sigmoid((D_t - thresh) * 20.0)

    # Compute risk_internal for different thresholds
    risk_internals = {}
    for thresh in thresholds:
        risk_internals[thresh] = v_signal * d_signals[thresh]

    return {
        'is_hall': is_hall,
        'num_tokens': varentropy.shape[1],
        'varentropy_mean': varentropy.float().mean().item(),
        'varentropy_max': varentropy.float().max().item(),
        'varentropy_p90': torch.quantile(varentropy.float().flatten(), 0.90).item(),
        'dispersion_mean': D_t.float().mean().item(),
        'dispersion_max': D_t.float().max().item(),
        'dispersion_p90': torch.quantile(D_t.float().flatten(), 0.90).item(),
        'jsd_mean': jsd.float().mean().item(),
        'jsd_max': jsd.float().max().item(),
        'jsd_p90': torch.quantile(jsd.float().flatten(), 0.90).item(),
        'v_signal_mean': v_signal.float().mean().item(),
        'v_signal_p90': torch.quantile(v_signal.float().flatten(), 0.90).item(),
        **{f'd_signal_{thresh}_mean': d_signals[thresh].float().mean().item() for thresh in thresholds},
        **{f'risk_int_{thresh}_mean': risk_internals[thresh].float().mean().item() for thresh in thresholds},
        **{f'risk_int_{thresh}_p90': torch.quantile(risk_internals[thresh].float().flatten(), 0.90).item() for thresh in thresholds},
    }


def main():
    print("=" * 80)
    print("V12 Diagnostic Analysis")
    print("=" * 80)

    print("\nLoading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    config = AGSARConfig(version=12)
    agsar = AGSAR(model, tokenizer, config)

    # Test on HaluEval
    print("\n" + "=" * 80)
    print("HALUEVAL QA ANALYSIS")
    print("=" * 80)

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    fact_stats = []
    hall_stats = []

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        prompt = f"Knowledge: {sample['knowledge']}\n\nQuestion: {sample['question']}"
        response = sample['answer']
        is_hall = sample.get('hallucination', 'no').lower() == 'yes'

        try:
            agsar.calibrate_on_prompt(prompt)
            stats = analyze_signals(agsar, prompt, response, is_hall)

            if is_hall:
                hall_stats.append(stats)
            else:
                fact_stats.append(stats)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Print analysis
    print(f"\nSamples: {len(fact_stats)} facts, {len(hall_stats)} halls")

    print("\n--- RAW SIGNALS (Mean across samples) ---")
    print(f"{'Signal':<25} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 60)

    metrics = ['varentropy_mean', 'varentropy_p90', 'dispersion_mean', 'dispersion_p90',
               'jsd_mean', 'jsd_p90', 'v_signal_mean', 'v_signal_p90']
    for m in metrics:
        fact_val = np.mean([s[m] for s in fact_stats])
        hall_val = np.mean([s[m] for s in hall_stats])
        print(f"{m:<25} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    print("\n--- D_SIGNAL at different thresholds ---")
    print(f"{'Threshold':<15} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 55)
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        key = f'd_signal_{thresh}_mean'
        fact_val = np.mean([s[key] for s in fact_stats])
        hall_val = np.mean([s[key] for s in hall_stats])
        print(f"{thresh:<15.2f} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    print("\n--- RISK_INTERNAL (Mean) at different thresholds ---")
    print(f"{'Threshold':<15} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 55)
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        key = f'risk_int_{thresh}_mean'
        fact_val = np.mean([s[key] for s in fact_stats])
        hall_val = np.mean([s[key] for s in hall_stats])
        print(f"{thresh:<15.2f} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    print("\n--- RISK_INTERNAL (P90) at different thresholds ---")
    print(f"{'Threshold':<15} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 55)
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        key = f'risk_int_{thresh}_p90'
        fact_val = np.mean([s[key] for s in fact_stats])
        hall_val = np.mean([s[key] for s in hall_stats])
        print(f"{thresh:<15.2f} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    # Test on RAGTruth
    print("\n" + "=" * 80)
    print("RAGTRUTH QA ANALYSIS")
    print("=" * 80)

    dataset = load_dataset("flowaicom/RAGTruth_test", split="qa")

    fact_stats = []
    hall_stats = []

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        prompt = sample["prompt"]
        response = sample["response"]
        is_hall = sample["score"] == 1

        try:
            agsar.calibrate_on_prompt(prompt)
            stats = analyze_signals(agsar, prompt, response, is_hall)

            if is_hall:
                hall_stats.append(stats)
            else:
                fact_stats.append(stats)
        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # Print analysis
    print(f"\nSamples: {len(fact_stats)} facts, {len(hall_stats)} halls")

    print("\n--- RAW SIGNALS (Mean across samples) ---")
    print(f"{'Signal':<25} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 60)

    for m in metrics:
        fact_val = np.mean([s[m] for s in fact_stats]) if fact_stats else 0
        hall_val = np.mean([s[m] for s in hall_stats]) if hall_stats else 0
        print(f"{m:<25} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    print("\n--- D_SIGNAL at different thresholds ---")
    print(f"{'Threshold':<15} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 55)
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        key = f'd_signal_{thresh}_mean'
        fact_val = np.mean([s[key] for s in fact_stats]) if fact_stats else 0
        hall_val = np.mean([s[key] for s in hall_stats]) if hall_stats else 0
        print(f"{thresh:<15.2f} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    print("\n--- RISK_INTERNAL (Mean) at different thresholds ---")
    print(f"{'Threshold':<15} {'Facts':>12} {'Halls':>12} {'Gap':>12}")
    print("-" * 55)
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        key = f'risk_int_{thresh}_mean'
        fact_val = np.mean([s[key] for s in fact_stats]) if fact_stats else 0
        hall_val = np.mean([s[key] for s in hall_stats]) if hall_stats else 0
        print(f"{thresh:<15.2f} {fact_val:>12.4f} {hall_val:>12.4f} {hall_val - fact_val:>12.4f}")

    agsar.cleanup()

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("""
Key observations to check:
1. Is dispersion_mean higher for halls than facts on HaluEval?
2. Is JSD higher for halls than facts on RAGTruth?
3. What dispersion threshold gives best separation on HaluEval?
4. Is the current threshold=0.1 optimal?
""")


if __name__ == "__main__":
    main()
