#!/usr/bin/env python3
"""
Debug script to investigate v4 token-level scores.

This script examines why min() aggregation is producing poor results.
"""

import os
import sys
import torch
import numpy as np

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig
from ag_sar.measures.authority import compute_semantic_authority_v4
from ag_sar.measures.entropy import compute_varentropy
from ag_sar.measures.semantics import compute_semantic_dispersion


def load_sample_pairs():
    """Load a few fact/hallucination pairs for debugging."""
    return [
        {
            "question": "What is the capital of France?",
            "fact": "Paris",
            "hallucination": "The capital of France is Lyon, which is located in the southern part of the country and is known for its beautiful architecture and rich history.",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "fact": "William Shakespeare",
            "hallucination": "Romeo and Juliet was written by Christopher Marlowe, a contemporary of Shakespeare who was known for his dramatic works including Doctor Faustus.",
        },
    ]


def debug_token_level_scores(
    model,
    tokenizer,
    prompt: str,
    response: str,
    label: str,
):
    """Debug token-level v4 scores."""
    print(f"\n{'='*60}")
    print(f"Label: {label}")
    print(f"Prompt: {prompt[:50]}...")
    print(f"Response: {response[:50]}...")
    print(f"{'='*60}")

    # Tokenize
    prompt_enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    response_enc = tokenizer(response, return_tensors='pt', add_special_tokens=False)

    input_ids = torch.cat([prompt_enc['input_ids'], response_enc['input_ids']], dim=1)
    attention_mask = torch.ones_like(input_ids)
    prompt_length = prompt_enc['input_ids'].size(1)

    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    print(f"\nSequence length: {input_ids.size(1)}")
    print(f"Prompt length: {prompt_length}")
    print(f"Response length: {input_ids.size(1) - prompt_length}")

    # Forward pass
    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)

    logits = outputs.logits
    attn = outputs.attentions[-1]  # Last layer attention

    # Get embedding matrix
    embed_matrix = model.get_output_embeddings().weight.detach()

    # Compute individual components
    varentropy = compute_varentropy(logits, attention_mask)
    V_penalty = torch.tanh(varentropy / 3.0)

    dispersion = compute_semantic_dispersion(
        logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
    )
    consistency = (1.0 - dispersion).clamp(0.0, 1.0)

    # Compute authority flow
    from ag_sar.ops import compute_authority_flow_recursive
    gamma = torch.ones_like(varentropy)
    A_t = compute_authority_flow_recursive(attn, prompt_length, gamma, attention_mask)

    # Compute final score
    score = A_t * (1.0 - V_penalty) * consistency
    score = score.clamp(0.0, 1.0)

    # Get response portion
    response_varentropy = varentropy[0, prompt_length:].cpu().numpy()
    response_V_penalty = V_penalty[0, prompt_length:].cpu().numpy()
    response_dispersion = dispersion[0, prompt_length:].cpu().numpy()
    response_consistency = consistency[0, prompt_length:].cpu().numpy()
    response_authority = A_t[0, prompt_length:].cpu().numpy()
    response_score = score[0, prompt_length:].cpu().numpy()

    # Decode response tokens
    response_tokens = input_ids[0, prompt_length:].cpu().tolist()
    response_text = [tokenizer.decode([t]) for t in response_tokens]

    print(f"\n--- Response Token Analysis ---")
    print(f"{'Idx':<4} {'Token':<15} {'V':<8} {'V_pen':<8} {'D':<8} {'1-D':<8} {'A':<8} {'Score':<8}")
    print("-" * 80)

    for i, (tok, v, vp, d, c, a, s) in enumerate(zip(
        response_text, response_varentropy, response_V_penalty,
        response_dispersion, response_consistency, response_authority, response_score
    )):
        tok_display = tok.replace('\n', '\\n')[:12]
        print(f"{i:<4} {tok_display:<15} {v:<8.3f} {vp:<8.3f} {d:<8.3f} {c:<8.3f} {a:<8.3f} {s:<8.3f}")

    print("-" * 80)
    print(f"\nAggregation Statistics (Response Only):")
    print(f"  Varentropy:   mean={response_varentropy.mean():.3f}, min={response_varentropy.min():.3f}, max={response_varentropy.max():.3f}")
    print(f"  V_penalty:    mean={response_V_penalty.mean():.3f}, min={response_V_penalty.min():.3f}, max={response_V_penalty.max():.3f}")
    print(f"  Dispersion:   mean={response_dispersion.mean():.3f}, min={response_dispersion.min():.3f}, max={response_dispersion.max():.3f}")
    print(f"  Authority:    mean={response_authority.mean():.3f}, min={response_authority.min():.3f}, max={response_authority.max():.3f}")
    print(f"  Score:        mean={response_score.mean():.3f}, min={response_score.min():.3f}, max={response_score.max():.3f}")

    # The key metrics
    min_score = response_score.min()
    mean_score = response_score.mean()
    uncertainty_min = 1.0 - min_score
    uncertainty_mean = 1.0 - mean_score

    print(f"\n  UNCERTAINTY (1 - score):")
    print(f"    min aggregation:  {uncertainty_min:.4f}")
    print(f"    mean aggregation: {uncertainty_mean:.4f}")

    # Find the "worst" token
    worst_idx = np.argmin(response_score)
    print(f"\n  WORST TOKEN (idx={worst_idx}):")
    print(f"    Token: '{response_text[worst_idx]}'")
    print(f"    Varentropy: {response_varentropy[worst_idx]:.3f}")
    print(f"    V_penalty:  {response_V_penalty[worst_idx]:.3f}")
    print(f"    Dispersion: {response_dispersion[worst_idx]:.3f}")
    print(f"    Authority:  {response_authority[worst_idx]:.3f}")
    print(f"    Score:      {response_score[worst_idx]:.3f}")

    return {
        "label": label,
        "min_score": min_score,
        "mean_score": mean_score,
        "worst_token": response_text[worst_idx],
        "worst_idx": worst_idx,
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
        attn_implementation="eager",  # Need full attention weights
    )

    samples = load_sample_pairs()

    all_results = []
    for sample in samples:
        prompt = f"Question: {sample['question']}\nAnswer:"

        # Fact
        result_fact = debug_token_level_scores(
            model, tokenizer, prompt, sample['fact'], "FACT"
        )
        all_results.append(result_fact)

        # Hallucination
        result_hall = debug_token_level_scores(
            model, tokenizer, prompt, sample['hallucination'], "HALLUCINATION"
        )
        all_results.append(result_hall)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in all_results:
        print(f"{r['label']:<15} min_score={r['min_score']:.4f} mean_score={r['mean_score']:.4f} worst='{r['worst_token']}'")


if __name__ == "__main__":
    main()
