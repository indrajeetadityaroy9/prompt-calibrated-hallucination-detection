#!/usr/bin/env python3
"""
RAGTruth Signal Isolation Diagnostic

Confirms the hypothesis about "Dual Nature of Varentropy":
- HaluEval (Confusion): V_hall >> V_fact
- RAGTruth (Simplification): V_hall << V_fact

We need to see:
1. Varentropy Mean: Is Fact > Hall? (Hypothesis: Yes)
2. Authority Mean: Is Fact > Hall? (Does Provenance work?)
3. Dispersion Mean: Is Fact < Hall? (Is Consistency valid?)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ComponentScores:
    """Per-token component scores for a single sample."""
    authority: torch.Tensor
    varentropy: torch.Tensor
    dispersion: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    tokens: List[str]
    response_start: int


def load_model_and_tokenizer(model_name: str):
    """Load model with proper configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    print(f"  Model loaded on {next(model.parameters()).device}")

    return model, tokenizer


def load_ragtruth(task_type: str = "qa", num_samples: int = 50, seed: int = 42):
    """Load RAGTruth samples."""
    from datasets import load_dataset
    import random

    print(f"\nLoading RAGTruth {task_type} ({num_samples} samples)...")

    dataset = load_dataset("flowaicom/RAGTruth_test", split=task_type)
    samples = list(dataset)
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:num_samples]

    # Count labels (score: 1 = hallucinated, 0 = faithful)
    n_hall = sum(1 for s in samples if s.get('score', 0) == 1)
    n_fact = len(samples) - n_hall
    print(f"  Hallucinations: {n_hall}, Facts: {n_fact}")

    return samples


def compute_components(
    model,
    tokenizer,
    prompt: str,
    response: str,
) -> ComponentScores:
    """Compute all component scores for a sample."""
    from ag_sar.modeling import ModelAdapter
    from ag_sar.measures.entropy import compute_varentropy, compute_token_entropy
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.ops import compute_authority_flow_vectorized

    device = next(model.parameters()).device

    # Tokenize prompt and full text
    prompt_enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    response_enc = tokenizer(response, return_tensors='pt', add_special_tokens=False)

    input_ids = torch.cat([prompt_enc['input_ids'], response_enc['input_ids']], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids)

    response_start = prompt_enc['input_ids'].size(1)
    seq_len = input_ids.size(1)

    # Get token strings
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    # Setup hooks
    num_layers = model.config.num_hidden_layers
    semantic_layers = list(range(max(0, num_layers - 4), num_layers))

    adapter = ModelAdapter(
        model=model,
        layers=semantic_layers,
        dtype=torch.bfloat16,
    )
    adapter.register()

    try:
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        embed_matrix = model.get_output_embeddings().weight.detach()

        last_layer = semantic_layers[-1]
        attn_weights = adapter.capture.attention_weights.get(last_layer)

        if attn_weights is None:
            raise RuntimeError("Attention weights not captured")

        # 1. Authority Flow
        authority = compute_authority_flow_vectorized(
            attn_weights, response_start, attention_mask
        )

        # 2. Varentropy
        varentropy = compute_varentropy(logits, attention_mask)

        # 3. Semantic Dispersion
        dispersion = compute_semantic_dispersion(
            logits, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
        )

        # 4. Token Entropy
        entropy = compute_token_entropy(logits, attention_mask)

        # 5. Log Probability
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.zeros(1, seq_len, device=device)
        token_log_probs[:, 1:] = log_probs[:, :-1, :].gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        return ComponentScores(
            authority=authority[0].cpu(),
            varentropy=varentropy[0].cpu(),
            dispersion=dispersion[0].cpu(),
            log_prob=token_log_probs[0].cpu(),
            entropy=entropy[0].cpu(),
            tokens=tokens,
            response_start=response_start,
        )

    finally:
        adapter.cleanup()


def compute_prompt_varentropy(model, tokenizer, prompt: str) -> float:
    """Compute mean varentropy for the prompt only (for complexity matching)."""
    from ag_sar.measures.entropy import compute_varentropy

    device = next(model.parameters()).device

    prompt_enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
    input_ids = prompt_enc['input_ids'].to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(input_ids, attention_mask=attention_mask)

    varentropy = compute_varentropy(outputs.logits, attention_mask)
    return varentropy.mean().item()


def run_signal_analysis(model, tokenizer, samples: List[dict]):
    """Run signal isolation analysis on RAGTruth."""
    print("\n" + "=" * 70)
    print("RAGTRUTH SIGNAL ISOLATION ANALYSIS")
    print("=" * 70)

    # Collect scores
    authority_scores = []
    varentropy_scores = []
    dispersion_scores = []
    logprob_scores = []
    entropy_scores = []
    prompt_varentropy_scores = []
    varentropy_ratios = []
    labels = []

    for i, sample in enumerate(samples):
        label = sample.get('score', 0)  # 1 = hallucinated, 0 = faithful
        labels.append(label)

        prompt = sample['prompt']
        response = sample['response']

        try:
            # Compute prompt varentropy for complexity matching
            prompt_v = compute_prompt_varentropy(model, tokenizer, prompt)
            prompt_varentropy_scores.append(prompt_v)

            scores = compute_components(model, tokenizer, prompt, response)

            # Aggregate over response tokens
            resp_slice = slice(scores.response_start, None)

            resp_v = scores.varentropy[resp_slice].mean().item()
            resp_a = scores.authority[resp_slice].mean().item()
            resp_d = scores.dispersion[resp_slice].mean().item()
            resp_lp = scores.log_prob[resp_slice].mean().item()
            resp_e = scores.entropy[resp_slice].mean().item()

            authority_scores.append(resp_a)
            varentropy_scores.append(resp_v)
            dispersion_scores.append(resp_d)
            logprob_scores.append(resp_lp)
            entropy_scores.append(resp_e)

            # Compute ratio for v6 hypothesis
            ratio = resp_v / (prompt_v + 1e-6)
            varentropy_ratios.append(ratio)

            label_str = "HALL" if label == 1 else "FACT"
            print(f"  [{i+1:3d}/{len(samples)}] {label_str} | "
                  f"A={resp_a:.3f} V={resp_v:.3f} D={resp_d:.3f} "
                  f"V_prompt={prompt_v:.3f} Ratio={ratio:.3f}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(samples)}] ERROR: {str(e)[:50]}")
            labels.pop()
            if prompt_varentropy_scores:
                prompt_varentropy_scores.pop()

    # Compute statistics
    print("\n" + "=" * 70)
    print("SIGNAL STATISTICS")
    print("=" * 70)

    hall_mask = [l == 1 for l in labels]
    fact_mask = [l == 0 for l in labels]

    def print_stats(name, scores, expected_hall_higher=True):
        hall_scores = [s for s, m in zip(scores, hall_mask) if m]
        fact_scores = [s for s, m in zip(scores, fact_mask) if m]

        if not hall_scores or not fact_scores:
            print(f"  {name}: Not enough samples")
            return

        hall_mean, hall_std = np.mean(hall_scores), np.std(hall_scores)
        fact_mean, fact_std = np.mean(fact_scores), np.std(fact_scores)
        diff = hall_mean - fact_mean

        # Compute AUROC
        try:
            auroc = roc_auc_score(labels, scores if expected_hall_higher else [-s for s in scores])
        except:
            auroc = 0.5

        # Determine if inverted
        if expected_hall_higher:
            inverted = hall_mean < fact_mean
        else:
            inverted = hall_mean > fact_mean

        status = "INVERTED!" if inverted else "correct"

        print(f"\n  {name}:")
        print(f"    Hall: {hall_mean:.4f} ± {hall_std:.4f}")
        print(f"    Fact: {fact_mean:.4f} ± {fact_std:.4f}")
        print(f"    Diff: {diff:+.4f} [{status}]")
        print(f"    AUROC: {auroc:.4f}")

    print("\n--- CORE SIGNALS ---")

    print_stats("Authority (expect: Fact > Hall)", authority_scores, expected_hall_higher=False)
    print_stats("Varentropy (expect: Hall > Fact for Confusion regime)", varentropy_scores, expected_hall_higher=True)
    print_stats("Dispersion (expect: Hall > Fact)", dispersion_scores, expected_hall_higher=True)

    print("\n--- COMPLEXITY MATCHING (v6 Hypothesis) ---")

    print_stats("Prompt Varentropy", prompt_varentropy_scores, expected_hall_higher=True)
    print_stats("Response Varentropy", varentropy_scores, expected_hall_higher=True)

    # Key hypothesis: Varentropy Ratio
    hall_ratios = [r for r, m in zip(varentropy_ratios, hall_mask) if m]
    fact_ratios = [r for r, m in zip(varentropy_ratios, fact_mask) if m]

    if hall_ratios and fact_ratios:
        print(f"\n  Varentropy Ratio (V_response / V_prompt):")
        print(f"    Hall: {np.mean(hall_ratios):.4f} ± {np.std(hall_ratios):.4f}")
        print(f"    Fact: {np.mean(fact_ratios):.4f} ± {np.std(fact_ratios):.4f}")
        print(f"    Diff: {np.mean(hall_ratios) - np.mean(fact_ratios):+.4f}")

        # Key insight for v6
        if np.mean(hall_ratios) < np.mean(fact_ratios):
            print(f"\n  *** SIMPLIFICATION REGIME CONFIRMED ***")
            print(f"      Hall ratio < Fact ratio")
            print(f"      Hallucinations are SIMPLER than facts (oversimplification)")
            print(f"      v6 should penalize Ratio << 1.0")
        else:
            print(f"\n  *** CONFUSION REGIME ***")
            print(f"      Hall ratio > Fact ratio")
            print(f"      Hallucinations are MORE COMPLEX than facts (confusion)")

    # Summary for v6 calibration
    print("\n" + "=" * 70)
    print("V6 CALIBRATION DATA")
    print("=" * 70)

    if hall_ratios and fact_ratios:
        print(f"\n  Fact Ratio Mean:  {np.mean(fact_ratios):.4f}")
        print(f"  Hall Ratio Mean:  {np.mean(hall_ratios):.4f}")
        print(f"  Suggested σ for Gaussian: {abs(np.mean(fact_ratios) - np.mean(hall_ratios)) / 2:.4f}")


def main():
    parser = argparse.ArgumentParser(description="RAGTruth Signal Isolation Diagnostic")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="qa",
        choices=["qa", "summarization", "data2txt"],
        help="RAGTruth task type"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RAGTRUTH SIGNAL ISOLATION DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Task: {args.task_type}")
    print(f"Samples: {args.num_samples}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    samples = load_ragtruth(args.task_type, args.num_samples)

    run_signal_analysis(model, tokenizer, samples)


if __name__ == "__main__":
    main()
