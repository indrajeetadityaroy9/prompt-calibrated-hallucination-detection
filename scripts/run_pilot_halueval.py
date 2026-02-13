"""
Pilot evaluation on HaluEval dataset (via HuggingFace).

Runs forced decoding on HaluEval QA samples to validate the detector.

Usage:
    python scripts/run_pilot_halueval.py --num-examples 100
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import torch
from datasets import load_dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig, TokenSignals
from ag_sar.evaluation.modes import ForcedDecodingEvaluator
from ag_sar.evaluation.metrics import compute_metrics, compute_confidence_interval



def compute_token_risk(signals: TokenSignals, include_entropy: bool = True) -> float:
    """Risk computation from signals.

    Uses normalized signals to combine JSD, LCI, inv_margin, and optionally entropy.
    """
    # All signals are already in [0, 1] range
    components = [signals.jsd_cand, signals.lci_cand, signals.inv_margin]

    if include_entropy:
        # Normalize entropy to [0, 1] using sigmoid (typical range 0-6)
        import math
        norm_entropy = 1 / (1 + math.exp(-0.5 * (signals.entropy - 2)))
        components.append(norm_entropy)

    risk = sum(components) / len(components)
    return min(1.0, max(0.0, risk))


def run_pilot(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_examples: int = 100,
    dataset_config: str = "qa_samples",
):
    """Run pilot evaluation on HaluEval."""

    # Load HaluEval dataset
    print(f"Loading HaluEval/{dataset_config} from HuggingFace...")
    dataset = load_dataset("pminervini/HaluEval", dataset_config, split="data")
    print(f"Loaded {len(dataset)} total examples")

    # Sample balanced set
    print(f"Sampling {num_examples} examples...")

    # Get indices for positive and negative examples
    pos_indices = [i for i, ex in enumerate(dataset) if ex['hallucination'] == 'yes']
    neg_indices = [i for i, ex in enumerate(dataset) if ex['hallucination'] == 'no']

    print(f"Dataset: {len(pos_indices)} positive, {len(neg_indices)} negative")

    # Sample balanced
    import random
    random.seed(42)
    n_each = num_examples // 2
    sample_pos = random.sample(pos_indices, min(n_each, len(pos_indices)))
    sample_neg = random.sample(neg_indices, min(n_each, len(neg_indices)))
    sample_indices = sample_pos + sample_neg
    random.shuffle(sample_indices)

    examples = [dataset[i] for i in sample_indices]
    print(f"Sampled {len(examples)} examples ({len(sample_pos)} pos, {len(sample_neg)} neg)")

    # Load model
    print(f"Loading model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True,
    )
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
    all_response_scores = []
    all_response_scores_no_ent = []
    all_response_labels = []
    all_entropy_scores = []
    topk_masses = []
    signal_stats = {
        'jsd_cand': [], 'lci_cand': [], 'var_logp_cand': [],
        'entropy': [], 'inv_margin': []
    }

    print(f"Running forced decoding evaluation on {len(examples)} examples...")
    start_time = time.time()

    for i, example in enumerate(tqdm(examples, desc="Evaluating")):
        # Extract fields based on config
        if dataset_config == "qa_samples":
            context = example['knowledge']
            question = example['question']
            response = example['answer']
            label = 1 if example['hallucination'] == 'yes' else 0
        elif dataset_config == "qa":
            context = example['knowledge']
            question = example['question']
            # Use hallucinated answer half the time, right answer other half
            if example.get('hallucination') == 'yes' or i % 2 == 0:
                response = example.get('hallucinated_answer', example.get('right_answer', ''))
                label = 1
            else:
                response = example.get('right_answer', '')
                label = 0
        else:
            print(f"Unsupported config: {dataset_config}")
            continue

        # Run forced decoding
        try:
            token_signals = evaluator.evaluate(
                context=context,
                question=question,
                response=response,
            )
        except Exception as e:
            print(f"Error on example {i}: {e}")
            continue

        if not token_signals:
            continue

        # Compute token risks - both with and without entropy
        token_risks = [compute_token_risk(sig, include_entropy=True) for sig in token_signals]
        token_risks_no_ent = [compute_token_risk(sig, include_entropy=False) for sig in token_signals]

        # Response-level risk (max of token risks)
        response_risk = max(token_risks) if token_risks else 0.0
        response_risk_no_ent = max(token_risks_no_ent) if token_risks_no_ent else 0.0
        all_response_scores.append(response_risk)
        all_response_scores_no_ent.append(response_risk_no_ent)
        all_response_labels.append(label)

        # Entropy-only baseline (max entropy across tokens)
        max_entropy = max(sig.entropy for sig in token_signals) if token_signals else 0.0
        all_entropy_scores.append(max_entropy)

        # Collect signal statistics
        for sig in token_signals:
            signal_stats['jsd_cand'].append(sig.jsd_cand)
            signal_stats['lci_cand'].append(sig.lci_cand)
            signal_stats['var_logp_cand'].append(sig.var_logp_cand)
            signal_stats['entropy'].append(sig.entropy)
            signal_stats['inv_margin'].append(sig.inv_margin)
            if sig.topk_mass is not None:
                topk_masses.append(sig.topk_mass)

    elapsed = time.time() - start_time

    # Compute metrics
    print("\n" + "=" * 60)
    print("PILOT EVALUATION RESULTS (HaluEval)")
    print("=" * 60)

    print(f"\nExamples evaluated: {len(all_response_scores)}")
    print(f"Positive examples: {sum(all_response_labels)}")
    print(f"Negative examples: {len(all_response_labels) - sum(all_response_labels)}")
    print(f"Time elapsed: {elapsed:.1f}s ({elapsed/len(examples):.2f}s/example)")

    # Response-level metrics - Combined (with entropy)
    print("\n1) Combined signals (JSD + LCI + inv_margin + entropy):")
    response_metrics = compute_metrics(all_response_scores, all_response_labels)
    print(f"   AUROC: {response_metrics.auroc:.4f}")
    print(f"   AUPRC: {response_metrics.auprc:.4f}")
    print(f"   F1: {response_metrics.f1:.4f}")

    # Response-level metrics - Without entropy
    print("\n2) Internal signals only (JSD + LCI + inv_margin):")
    no_ent_metrics = compute_metrics(all_response_scores_no_ent, all_response_labels)
    print(f"   AUROC: {no_ent_metrics.auroc:.4f}")
    print(f"   AUPRC: {no_ent_metrics.auprc:.4f}")
    print(f"   F1: {no_ent_metrics.f1:.4f}")

    # Entropy baseline metrics
    print("\n3) Entropy-only baseline:")
    entropy_metrics = compute_metrics(all_entropy_scores, all_response_labels)
    print(f"   AUROC: {entropy_metrics.auroc:.4f}")
    print(f"   AUPRC: {entropy_metrics.auprc:.4f}")

    # Comparison summary
    print("\nComparison Summary:")
    print(f"  Combined vs Entropy-only: {response_metrics.auroc - entropy_metrics.auroc:+.4f} AUROC")
    print(f"  Internal vs Entropy-only: {no_ent_metrics.auroc - entropy_metrics.auroc:+.4f} AUROC")
    print(f"  Combined vs Internal:     {response_metrics.auroc - no_ent_metrics.auroc:+.4f} AUROC")

    if no_ent_metrics.auroc > entropy_metrics.auroc:
        print("  ✓ Internal signals outperform entropy alone!")
    else:
        print("  Note: HaluEval has systematic length differences (hallucinated=longer)")
        print("        This makes entropy a strong proxy for labels.")

    # Confidence intervals
    print("\nBootstrap 95% CIs:")
    auroc_ci, auprc_ci = compute_confidence_interval(
        all_response_scores, all_response_labels, n_bootstrap=1000
    )
    print(f"  Response AUROC: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
    print(f"  Response AUPRC: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]")

    # Signal statistics
    print("\nSignal statistics (across all tokens):")
    for name, values in signal_stats.items():
        if values:
            sorted_vals = sorted(values)
            median = sorted_vals[len(sorted_vals)//2]
            print(f"  {name}: min={min(values):.4f}, med={median:.4f}, max={max(values):.4f}")

    # Sanity checks
    print("\nSanity checks:")
    if topk_masses:
        sorted_masses = sorted(topk_masses)
        median_mass = sorted_masses[len(sorted_masses)//2]
        print(f"  topk_mass: min={min(topk_masses):.3f}, med={median_mass:.3f}, max={max(topk_masses):.3f}")
        low_mass = sum(1 for m in topk_masses if m < 0.8)
        print(f"  topk_mass < 0.8: {low_mass}/{len(topk_masses)} ({100*low_mass/len(topk_masses):.1f}%)")
        if median_mass > 0.9:
            print("  ✓ topk_mass median > 0.9 (candidate approximation valid)")
        else:
            print("  ⚠ topk_mass median <= 0.9 (candidate approximation may be weak)")

    print("\n" + "=" * 60)

    return {
        "response_auroc": response_metrics.auroc,
        "response_auprc": response_metrics.auprc,
        "entropy_auroc": entropy_metrics.auroc,
        "delta_auroc": delta_auroc,
        "auroc_ci": auroc_ci,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HaluEval pilot evaluation")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--config", default="qa_samples",
                       choices=["qa_samples", "qa", "summarization_samples"])

    args = parser.parse_args()

    run_pilot(
        model_name=args.model,
        num_examples=args.num_examples,
        dataset_config=args.config,
    )
