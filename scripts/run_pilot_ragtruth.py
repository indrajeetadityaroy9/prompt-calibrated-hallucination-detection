"""
Pilot evaluation on RAGTruth dataset (via HuggingFace).

Runs forced decoding with token-level hallucination labels.

Usage:
    python scripts/run_pilot_ragtruth.py --num-examples 50
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig, TokenSignals
from ag_sar.evaluation.modes import ForcedDecodingEvaluator
from ag_sar.evaluation.metrics import compute_metrics, compute_confidence_interval



def compute_token_risk(signals: TokenSignals) -> float:
    """Risk computation from signals."""
    import math
    # Combine JSD, LCI, inv_margin, and normalized entropy
    norm_entropy = 1 / (1 + math.exp(-0.5 * (signals.entropy - 2)))
    risk = (signals.jsd_cand + signals.lci_cand + signals.inv_margin + norm_entropy) / 4
    return min(1.0, max(0.0, risk))


def align_labels_to_tokens(
    response: str,
    labels: List[Dict],
    tokenizer,
) -> List[int]:
    """
    Align character-level labels to token indices.

    Returns list of 0/1 per token.
    """
    tokens = tokenizer.encode(response, add_special_tokens=False)
    token_labels = [0] * len(tokens)

    if not labels:
        return token_labels

    # Build character-to-token mapping
    char_to_token = {}
    current_char = 0
    for token_idx, token_id in enumerate(tokens):
        token_text = tokenizer.decode([token_id])
        for j in range(len(token_text)):
            if current_char + j < len(response):
                char_to_token[current_char + j] = token_idx
        current_char += len(token_text)

    # Mark hallucinated tokens
    for label in labels:
        if label.get('implicit_true', False):
            continue  # Skip implicit_true

        label_type = label.get('label_type', '')
        if 'Conflict' not in label_type and 'Baseless' not in label_type:
            continue

        start = label.get('start', 0)
        end = label.get('end', start)

        for char_pos in range(start, min(end, len(response))):
            if char_pos in char_to_token:
                token_labels[char_to_token[char_pos]] = 1

    return token_labels


def run_pilot(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_examples: int = 50,
    task_type: str = "QA",
):
    """Run pilot evaluation on RAGTruth."""

    # Load RAGTruth from HuggingFace
    print(f"Loading RAGTruth-processed from HuggingFace...")
    dataset = load_dataset("wandb/RAGTruth-processed", split="train")
    print(f"Total examples: {len(dataset)}")

    # Filter by task type and hallucination presence
    filtered_indices = []
    for i, ex in enumerate(dataset):
        if ex['task_type'] != task_type:
            continue
        # Parse labels
        try:
            labels = json.loads(ex['hallucination_labels']) if ex['hallucination_labels'] else []
        except:
            labels = []
        # Check for non-implicit hallucinations
        has_hallucination = any(
            not l.get('implicit_true', False) and
            ('Conflict' in l.get('label_type', '') or 'Baseless' in l.get('label_type', ''))
            for l in labels
        )
        if has_hallucination:
            filtered_indices.append(i)

    print(f"{task_type} examples with hallucinations: {len(filtered_indices)}")

    # Sample
    import random
    random.seed(42)
    sample_indices = random.sample(filtered_indices, min(num_examples, len(filtered_indices)))
    examples = [dataset[i] for i in sample_indices]
    print(f"Sampled {len(examples)} examples")

    # Load model
    print(f"Loading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"Model loaded on {next(model.parameters()).device}")

    # Initialize detector
    config = DetectorConfig(
        layer_subset="last_third",
        candidate_topk=128,
        lci_topk=10,
    )

    detector = AGSAR(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    evaluator = ForcedDecodingEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        signal_computer=detector.signal_computer,
        candidate_manager=detector.candidate_manager,
    )

    # Run evaluation
    all_token_scores = []
    all_token_labels = []
    all_token_entropy = []  # For baseline comparison
    all_response_scores = []
    all_response_labels = []
    all_entropy_scores = []
    topk_masses = []

    print(f"Running forced decoding evaluation...")
    start_time = time.time()

    for example in tqdm(examples, desc="Evaluating"):
        context = example['context']
        question = example['query']
        response = example['output']

        # Parse labels
        try:
            labels = json.loads(example['hallucination_labels']) if example['hallucination_labels'] else []
        except:
            labels = []

        # Run forced decoding
        try:
            token_signals = evaluator.evaluate(
                context=context,
                question=question,
                response=response,
            )
        except Exception as e:
            print(f"Error: {e}")
            continue

        if not token_signals:
            continue

        # Align labels to tokens
        token_labels = align_labels_to_tokens(response, labels, tokenizer)

        # Match lengths
        min_len = min(len(token_signals), len(token_labels))
        token_labels = token_labels[:min_len]
        token_signals = token_signals[:min_len]

        # Compute token risks
        token_risks = [compute_token_risk(sig) for sig in token_signals]

        # Collect token-level data
        all_token_scores.extend(token_risks)
        all_token_labels.extend(token_labels)
        all_token_entropy.extend([sig.entropy for sig in token_signals])

        # Response-level
        response_risk = max(token_risks) if token_risks else 0.0
        response_label = 1 if any(l == 1 for l in token_labels) else 0
        all_response_scores.append(response_risk)
        all_response_labels.append(response_label)

        # Entropy baseline
        max_entropy = max(sig.entropy for sig in token_signals) if token_signals else 0.0
        all_entropy_scores.append(max_entropy)

        # topk_mass
        for sig in token_signals:
            if sig.topk_mass is not None:
                topk_masses.append(sig.topk_mass)

    elapsed = time.time() - start_time

    # Compute metrics
    print("\n" + "=" * 60)
    print("PILOT EVALUATION RESULTS (RAGTruth)")
    print("=" * 60)

    print(f"\nExamples evaluated: {len(all_response_scores)}")
    print(f"Total tokens: {len(all_token_scores)}")
    print(f"Hallucinated tokens: {sum(all_token_labels)}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/example)")

    # Token-level metrics
    print("\nToken-level metrics (our method):")
    token_metrics = compute_metrics(all_token_scores, all_token_labels)
    print(f"  AUROC: {token_metrics.auroc:.4f}")
    print(f"  AUPRC: {token_metrics.auprc:.4f}")
    print(f"  F1: {token_metrics.f1:.4f}")

    # Token-level entropy baseline
    if len(all_token_entropy) == len(all_token_labels):
        entropy_token_metrics = compute_metrics(all_token_entropy, all_token_labels)
        print("\nToken-level entropy baseline:")
        print(f"  AUROC: {entropy_token_metrics.auroc:.4f}")
        print(f"  AUPRC: {entropy_token_metrics.auprc:.4f}")
        delta = token_metrics.auroc - entropy_token_metrics.auroc
        print(f"\nDelta (ours - entropy): {delta:+.4f} AUROC")
        if delta > 0.05:
            print("  ✓ Meets +0.05 AUROC improvement target")
        elif delta > 0:
            print("  ~ Positive improvement")
        else:
            print("  ✗ No improvement over entropy")

    # Response-level metrics
    print("\nResponse-level metrics:")
    response_metrics = compute_metrics(all_response_scores, all_response_labels)
    print(f"  AUROC: {response_metrics.auroc:.4f}")
    print(f"  AUPRC: {response_metrics.auprc:.4f}")

    # Bootstrap CIs
    print("\nBootstrap 95% CIs (token-level):")
    auroc_ci, auprc_ci = compute_confidence_interval(
        all_token_scores, all_token_labels, n_bootstrap=1000
    )
    print(f"  Token AUROC: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")

    # Sanity checks
    print("\nSanity checks:")
    if topk_masses:
        sorted_masses = sorted(topk_masses)
        median_mass = sorted_masses[len(sorted_masses)//2]
        print(f"  topk_mass: min={min(topk_masses):.3f}, med={median_mass:.3f}, max={max(topk_masses):.3f}")
        if median_mass > 0.9:
            print("  ✓ topk_mass median > 0.9")

    print("\n" + "=" * 60)

    return {
        "token_auroc": token_metrics.auroc,
        "token_auprc": token_metrics.auprc,
        "response_auroc": response_metrics.auroc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGTruth pilot evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-examples", type=int, default=50)
    parser.add_argument("--task", default="QA", choices=["QA", "Summary", "Data2txt"])

    args = parser.parse_args()

    run_pilot(
        model_name=args.model,
        num_examples=args.num_examples,
        task_type=args.task,
    )
