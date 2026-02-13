#!/usr/bin/env python3
"""
Evaluate corrected signals on diverse QA benchmarks.

Datasets:
- TriviaQA: Trivia questions with Wikipedia/web evidence
- SQuAD: Stanford Question Answering Dataset (extractive QA)
- Natural Questions (NQ): Google's real user questions
- BioASQ: Biomedical QA (if available)

For each dataset:
1. Load questions with ground truth answers
2. Generate model answers
3. Determine hallucination via answer matching (EM/F1)
4. Collect per-token signals
5. Evaluate AUROC of corrected vs original signals

Usage:
    python scripts/evaluate_qa_datasets.py --dataset triviaqa --n-samples 100
    python scripts/evaluate_qa_datasets.py --dataset squad --n-samples 100
    python scripts/evaluate_qa_datasets.py --dataset nq --n-samples 100
    python scripts/evaluate_qa_datasets.py --all --n-samples 50
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer



# ============================================================================
# Answer Matching Utilities (from SQuAD evaluation)
# ============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def is_hallucination(prediction: str, ground_truths: List[str], threshold: float = 0.3) -> bool:
    """
    Determine if prediction is a hallucination based on F1 with ground truths.

    Args:
        prediction: Model's generated answer
        ground_truths: List of acceptable ground truth answers
        threshold: F1 threshold below which we consider it a hallucination

    Returns:
        True if hallucination (low F1), False if correct
    """
    if not prediction.strip():
        return True

    max_f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    return max_f1 < threshold


# ============================================================================
# Dataset Loaders
# ============================================================================

def load_triviaqa(n_samples: int = 100) -> List[Dict]:
    """Load TriviaQA dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)

    samples = []
    for i, item in enumerate(dataset):
        if len(samples) >= n_samples:
            break

        question = item["question"]
        # TriviaQA has multiple answer aliases
        answers = item["answer"]["aliases"] + [item["answer"]["value"]]
        answers = list(set(answers))  # Dedupe

        # Get context from search results or Wikipedia
        context = ""
        if item.get("search_results") and item["search_results"].get("search_context"):
            context = item["search_results"]["search_context"][0][:2000]
        elif item.get("entity_pages") and item["entity_pages"].get("wiki_context"):
            context = item["entity_pages"]["wiki_context"][0][:2000]

        samples.append({
            "question": question,
            "answers": answers,
            "context": context,
            "dataset": "triviaqa",
        })

    print(f"Loaded {len(samples)} TriviaQA samples")
    return samples


def load_squad(n_samples: int = 100) -> List[Dict]:
    """Load SQuAD v2 dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading SQuAD v2...")
    dataset = load_dataset("squad_v2", split="validation", trust_remote_code=True)

    samples = []
    for i, item in enumerate(dataset):
        if len(samples) >= n_samples:
            break

        # Skip unanswerable questions for cleaner evaluation
        if not item["answers"]["text"]:
            continue

        question = item["question"]
        answers = list(set(item["answers"]["text"]))
        context = item["context"]

        samples.append({
            "question": question,
            "answers": answers,
            "context": context,
            "dataset": "squad",
        })

    print(f"Loaded {len(samples)} SQuAD samples")
    return samples


def load_natural_questions(n_samples: int = 100) -> List[Dict]:
    """Load Natural Questions dataset from HuggingFace."""
    from datasets import load_dataset

    print("Loading Natural Questions...")
    # NQ is large, use streaming
    dataset = load_dataset("natural_questions", split="validation", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break

        question = item["question"]["text"]

        # Get short answers
        answers = []
        for ann in item["annotations"]:
            for sa in ann.get("short_answers", []):
                if sa.get("text"):
                    answers.append(sa["text"])

        if not answers:
            continue

        answers = list(set(answers))

        # Get document context (first 2000 chars)
        context = item["document"]["text"][:2000] if item.get("document") else ""

        samples.append({
            "question": question,
            "answers": answers,
            "context": context,
            "dataset": "nq",
        })

    print(f"Loaded {len(samples)} Natural Questions samples")
    return samples


def load_bioasq(n_samples: int = 100) -> List[Dict]:
    """
    Load BioASQ dataset.
    Note: BioASQ requires registration. Using pubmed_qa as alternative.
    """
    from datasets import load_dataset

    print("Loading PubMedQA (BioASQ alternative)...")
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break

        question = item["question"]
        # PubMedQA has yes/no/maybe answers, but also has long_answer
        long_answer = item.get("long_answer", "")
        if not long_answer:
            continue

        answers = [long_answer]
        context = " ".join(item.get("context", {}).get("contexts", []))[:2000]

        samples.append({
            "question": question,
            "answers": answers,
            "context": context,
            "dataset": "bioasq",
        })

    print(f"Loaded {len(samples)} BioASQ/PubMedQA samples")
    return samples


DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
    "nq": load_natural_questions,
    "bioasq": load_bioasq,
}


# ============================================================================
# Signal Collection
# ============================================================================

@dataclass
class TokenSignalData:
    """Per-token signal data."""
    position: int
    jsd: float
    entropy_pre: float
    surprise: float
    rank_pre: int
    rank_post: int
    top1_match: bool


@dataclass
class SampleResult:
    """Result for a single sample."""
    question: str
    context: str
    ground_truths: List[str]
    generated_answer: str
    is_hallucination: bool
    f1_score: float
    n_tokens: int
    token_signals: List[Dict]
    # Aggregated signals
    jsd_mean: float
    jsd_max: float
    max_surprise: float
    spike_count: int
    spike_rate: float


def format_qa_prompt(sample: Dict, tokenizer) -> str:
    """Format QA sample into chat prompt."""
    question = sample["question"]
    context = sample.get("context", "")

    if context:
        system = "You are a helpful assistant. Answer the question based on the provided context. Be concise."
        user = f"Context: {context[:1500]}\n\nQuestion: {question}"
    else:
        system = "You are a helpful assistant. Answer the question concisely."
        user = f"Question: {question}"

    # Use simple Llama-3 format instead of apply_chat_template
    # This avoids jinja2 version issues
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def collect_signals_for_sample(
    model,
    tokenizer,
    hook_manager,
    sample: Dict,
    max_new_tokens: int = 50,
    spike_threshold: float = 0.27,
) -> SampleResult:
    """
    Generate answer and collect signals for a single sample.
    """
    from ag_sar.hooks import LayerHiddenStates
    from ag_sar.numerics import safe_softmax, safe_jsd

    device = next(model.parameters()).device
    prompt = format_qa_prompt(sample, tokenizer)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    # Get model components
    lm_head = model.lm_head
    final_norm = model.model.norm
    optimal_layer = 10  # From principled analysis

    token_signals = []

    # Generate token by token
    for step in range(max_new_tokens):
        hook_manager.clear_buffer()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=False, use_cache=False)
            logits = outputs.logits[0, -1, :]

        # Get next token
        next_token = logits.argmax().item()

        # Get layer states
        layer_states = hook_manager.buffer.get_states()

        if optimal_layer in layer_states:
            states = layer_states[optimal_layer]
        elif layer_states:
            states = layer_states[max(layer_states.keys())]
        else:
            states = None

        if states is not None:
            h_pre = states.h_resid_attn
            h_post = states.h_resid_mlp

            if h_pre.dim() == 1:
                h_pre = h_pre.unsqueeze(0)
                h_post = h_post.unsqueeze(0)

            with torch.no_grad():
                h_pre = h_pre.to(dtype=lm_head.weight.dtype, device=device)
                h_post = h_post.to(dtype=lm_head.weight.dtype, device=device)

                h_pre_norm = final_norm(h_pre)
                h_post_norm = final_norm(h_post)

                z_pre = lm_head(h_pre_norm).squeeze(0)
                z_post = lm_head(h_post_norm).squeeze(0)

                p_pre = torch.softmax(z_pre.float(), dim=-1)
                p_post = torch.softmax(z_post.float(), dim=-1)

                # Compute JSD on top-128 candidates
                topk = torch.topk(logits, k=128).indices
                p_pre_cand = safe_softmax(z_pre[topk].float(), dim=-1)
                p_post_cand = safe_softmax(z_post[topk].float(), dim=-1)
                jsd = safe_jsd(p_pre_cand, p_post_cand)

                # Entropy
                entropy_pre = -(p_pre * torch.log(p_pre + 1e-10)).sum().item()

                # Surprise
                surprise = jsd / (entropy_pre + 0.1)

                # Ranks
                rank_pre = (p_pre > p_pre[next_token]).sum().item() + 1
                rank_post = (p_post > p_post[next_token]).sum().item() + 1

                # Top-1 match
                top1_match = (p_pre.argmax().item() == p_post.argmax().item())

                token_signals.append({
                    "position": step,
                    "jsd": jsd,
                    "entropy_pre": entropy_pre,
                    "surprise": surprise,
                    "rank_pre": rank_pre,
                    "rank_post": rank_post,
                    "top1_match": top1_match,
                })

        # Append token
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)

        # Stop on EOS
        if next_token == tokenizer.eos_token_id:
            break

    # Decode answer
    generated_answer = tokenizer.decode(input_ids[0, prompt_len:], skip_special_tokens=True).strip()

    # Compute hallucination label
    ground_truths = sample["answers"]
    max_f1 = max(f1_score(generated_answer, gt) for gt in ground_truths) if ground_truths else 0.0
    is_hall = is_hallucination(generated_answer, ground_truths)

    # Compute aggregated signals
    if token_signals:
        jsds = np.array([t["jsd"] for t in token_signals])
        surprises = np.array([t["surprise"] for t in token_signals])

        jsd_mean = np.mean(jsds)
        jsd_max = np.max(jsds)
        max_surprise = np.max(surprises)

        spike_mask = jsds > spike_threshold
        spike_count = spike_mask.sum()
        spike_rate = spike_count / len(jsds)
    else:
        jsd_mean = jsd_max = max_surprise = 0.0
        spike_count = 0
        spike_rate = 0.0

    return SampleResult(
        question=sample["question"],
        context=sample.get("context", "")[:200],
        ground_truths=ground_truths,
        generated_answer=generated_answer,
        is_hallucination=is_hall,
        f1_score=max_f1,
        n_tokens=len(token_signals),
        token_signals=token_signals,
        jsd_mean=jsd_mean,
        jsd_max=jsd_max,
        max_surprise=max_surprise,
        spike_count=spike_count,
        spike_rate=spike_rate,
    )


def evaluate_dataset(
    model,
    tokenizer,
    hook_manager,
    samples: List[Dict],
    spike_threshold: float = 0.27,
) -> Dict:
    """
    Evaluate corrected signals on a dataset.
    """
    results = []

    for sample in tqdm(samples, desc=f"Evaluating {samples[0]['dataset']}"):
        try:
            result = collect_signals_for_sample(
                model, tokenizer, hook_manager, sample,
                spike_threshold=spike_threshold,
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

    if not results:
        return {"error": "No valid results"}

    # Extract labels and signals
    labels = np.array([r.is_hallucination for r in results]).astype(int)

    # Compute AUROCs
    signal_aurocs = {}

    for signal_name in ["jsd_mean", "jsd_max", "max_surprise", "spike_count", "spike_rate", "n_tokens"]:
        values = np.array([getattr(r, signal_name) for r in results])

        if np.std(values) < 1e-10 or len(np.unique(labels)) < 2:
            auroc = 0.5
        else:
            try:
                auroc = roc_auc_score(labels, values)
            except ValueError:
                auroc = 0.5

        signal_aurocs[signal_name] = auroc

    # Summary stats
    n_hall = labels.sum()
    n_correct = len(labels) - n_hall
    hall_rate = n_hall / len(labels) if labels.size > 0 else 0

    return {
        "dataset": samples[0]["dataset"],
        "n_samples": len(results),
        "n_hallucinations": int(n_hall),
        "n_correct": int(n_correct),
        "hallucination_rate": hall_rate,
        "signal_aurocs": signal_aurocs,
        "best_signal": max(signal_aurocs.items(), key=lambda x: x[1] if x[0] != "n_tokens" else 0),
        "improvement_over_jsd_mean": signal_aurocs.get("max_surprise", 0.5) - signal_aurocs.get("jsd_mean", 0.5),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate corrected signals on QA datasets")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--all", action="store_true", help="Run on all datasets")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", default="results/qa_datasets_eval.json")
    parser.add_argument("--spike-threshold", type=float, default=0.27)
    args = parser.parse_args()

    # Determine which datasets to run
    if args.all:
        datasets_to_run = list(DATASET_LOADERS.keys())
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        parser.error("Must specify --dataset or --all")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    # Setup hooks
    from ag_sar.hooks import HookManager
    n_layers = len(model.model.layers)
    hook_manager = HookManager(model=model, layer_indices=list(range(n_layers)))
    hook_manager.install()

    all_results = {}

    try:
        for dataset_name in datasets_to_run:
            print(f"\n{'='*60}")
            print(f"EVALUATING: {dataset_name.upper()}")
            print(f"{'='*60}")

            # Load dataset
            try:
                samples = DATASET_LOADERS[dataset_name](n_samples=args.n_samples)
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
                all_results[dataset_name] = {"error": str(e)}
                continue

            # Evaluate
            result = evaluate_dataset(
                model, tokenizer, hook_manager, samples,
                spike_threshold=args.spike_threshold,
            )
            all_results[dataset_name] = result

            # Print results
            print(f"\nResults for {dataset_name}:")
            print(f"  Samples: {result.get('n_samples', 0)}")
            print(f"  Hallucination rate: {result.get('hallucination_rate', 0):.2%}")
            print(f"  Signal AUROCs:")
            for sig, auroc in result.get("signal_aurocs", {}).items():
                marker = "***" if auroc > 0.65 else ("**" if auroc > 0.55 else "")
                print(f"    {sig:20s}: {auroc:.4f} {marker}")
            print(f"  Best: {result.get('best_signal', ('N/A', 0))}")
            print(f"  Improvement over jsd_mean: {result.get('improvement_over_jsd_mean', 0):+.4f}")

    finally:
        hook_manager.remove()

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for dataset_name, result in all_results.items():
        if "error" in result:
            print(f"{dataset_name}: ERROR - {result['error']}")
        else:
            best = result.get("best_signal", ("N/A", 0))
            print(f"{dataset_name}: best={best[0]} (AUROC={best[1]:.4f}), "
                       f"improvement={result.get('improvement_over_jsd_mean', 0):+.4f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
