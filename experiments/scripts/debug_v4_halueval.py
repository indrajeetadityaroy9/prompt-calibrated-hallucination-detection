#!/usr/bin/env python3
"""
Debug v4 on actual HaluEval QA samples to understand the discrimination gap.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar.measures.entropy import compute_varentropy
from ag_sar.measures.semantics import compute_semantic_dispersion
from ag_sar.ops import compute_authority_flow_recursive


def analyze_sample(model, tokenizer, embed_matrix, prompt, response, label, idx):
    """Analyze a single sample and return key statistics."""
    # Tokenize
    prompt_enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    response_enc = tokenizer(response, return_tensors='pt', add_special_tokens=False)

    input_ids = torch.cat([prompt_enc['input_ids'], response_enc['input_ids']], dim=1)
    attention_mask = torch.ones_like(input_ids)
    prompt_length = prompt_enc['input_ids'].size(1)

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    response_len = input_ids.size(1) - prompt_length

    # Forward pass
    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)

    logits = outputs.logits
    attn = outputs.attentions[-1]

    # Compute components
    varentropy = compute_varentropy(logits, attention_mask)
    # Safe Harbor: V < 2.5 → penalty = 0
    V_excess = torch.nn.functional.relu(varentropy - 2.5)
    V_penalty = torch.tanh(V_excess / 3.0)

    dispersion = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    consistency = (1.0 - dispersion).clamp(0.0, 1.0)

    gamma = torch.ones_like(varentropy)
    A_t = compute_authority_flow_recursive(attn, prompt_length, gamma, attention_mask)

    # Final score
    score = A_t * (1.0 - V_penalty) * consistency
    score = score.clamp(0.0, 1.0)

    # Response portion stats
    resp_score = score[0, prompt_length:].cpu().numpy()
    resp_V = varentropy[0, prompt_length:].cpu().numpy()
    resp_A = A_t[0, prompt_length:].cpu().numpy()
    resp_D = dispersion[0, prompt_length:].cpu().numpy()

    min_score = resp_score.min()
    mean_score = resp_score.mean()

    # Find worst token
    worst_idx = np.argmin(resp_score)
    worst_token = tokenizer.decode([input_ids[0, prompt_length + worst_idx].item()])

    return {
        "idx": idx,
        "label": label,
        "response_len": response_len,
        "min_score": min_score,
        "mean_score": mean_score,
        "uncertainty_min": 1.0 - min_score,
        "uncertainty_mean": 1.0 - mean_score,
        "V_max": resp_V.max(),
        "V_mean": resp_V.mean(),
        "A_min": resp_A.min(),
        "D_max": resp_D.max(),
        "worst_token": worst_token,
        "response_preview": response[:50],
    }


def main():
    print("Loading model...")
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
    embed_matrix = model.get_output_embeddings().weight.detach()

    print("Loading HaluEval QA dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # Analyze first 40 samples
    results = []
    for i in range(40):
        sample = dataset[i]
        question = sample["question"]
        answer = sample["answer"]
        is_hallucination = sample["hallucination"] == "yes"
        label = "HALL" if is_hallucination else "FACT"

        prompt = f"Question: {question}\nAnswer:"

        result = analyze_sample(
            model, tokenizer, embed_matrix, prompt,
            answer, label, i
        )
        results.append(result)

        print(f"[{i:2d}] {label:4s} len={result['response_len']:3d} "
              f"U_min={result['uncertainty_min']:.3f} U_mean={result['uncertainty_mean']:.3f} "
              f"V_max={result['V_max']:.2f}")

    # Aggregate statistics
    facts = [r for r in results if r["label"] == "FACT"]
    halls = [r for r in results if r["label"] == "HALL"]

    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)

    print(f"\nFACTS (n={len(facts)}):")
    print(f"  Response length: {np.mean([f['response_len'] for f in facts]):.1f} ± {np.std([f['response_len'] for f in facts]):.1f}")
    print(f"  Uncertainty (min): {np.mean([f['uncertainty_min'] for f in facts]):.3f} ± {np.std([f['uncertainty_min'] for f in facts]):.3f}")
    print(f"  Uncertainty (mean): {np.mean([f['uncertainty_mean'] for f in facts]):.3f} ± {np.std([f['uncertainty_mean'] for f in facts]):.3f}")
    print(f"  V_max: {np.mean([f['V_max'] for f in facts]):.3f} ± {np.std([f['V_max'] for f in facts]):.3f}")

    print(f"\nHALLUCINATIONS (n={len(halls)}):")
    print(f"  Response length: {np.mean([h['response_len'] for h in halls]):.1f} ± {np.std([h['response_len'] for h in halls]):.1f}")
    print(f"  Uncertainty (min): {np.mean([h['uncertainty_min'] for h in halls]):.3f} ± {np.std([h['uncertainty_min'] for h in halls]):.3f}")
    print(f"  Uncertainty (mean): {np.mean([h['uncertainty_mean'] for h in halls]):.3f} ± {np.std([h['uncertainty_mean'] for h in halls]):.3f}")
    print(f"  V_max: {np.mean([h['V_max'] for h in halls]):.3f} ± {np.std([h['V_max'] for h in halls]):.3f}")

    # What matters for discrimination
    print(f"\nDISCRIMINATION GAP:")
    gap_min = np.mean([h['uncertainty_min'] for h in halls]) - np.mean([f['uncertainty_min'] for f in facts])
    gap_mean = np.mean([h['uncertainty_mean'] for h in halls]) - np.mean([f['uncertainty_mean'] for f in facts])
    print(f"  min aggregation: {gap_min:.3f} (Hall - Fact)")
    print(f"  mean aggregation: {gap_mean:.3f} (Hall - Fact)")

    # Show some problematic facts
    print("\n" + "=" * 70)
    print("PROBLEMATIC FACTS (high uncertainty):")
    print("=" * 70)
    high_uncertainty_facts = sorted(facts, key=lambda x: -x['uncertainty_min'])[:5]
    for f in high_uncertainty_facts:
        print(f"\n[{f['idx']}] U_min={f['uncertainty_min']:.3f}, len={f['response_len']}, V_max={f['V_max']:.2f}")
        print(f"    worst_token='{f['worst_token']}', response='{f['response_preview']}...'")


if __name__ == "__main__":
    main()
