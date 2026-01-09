#!/usr/bin/env python3
"""
AG-SAR Signal Isolation Diagnostic

Answers 5 critical questions about HaluEval QA performance:

1. Signal Isolation Check - Individual component AUROCs
2. Token-Level Dynamics - Per-token arrays for hallucinated samples
3. Instruction Tuning Factor - Prompt template verification
4. Length Bias - Length distribution analysis
5. Ground Truth Integrity - Manual sample inspection

Usage:
    python -m experiments.scripts.diagnostic_signal_isolation

    # Use 8B for faster iteration:
    python -m experiments.scripts.diagnostic_signal_isolation --model meta-llama/Llama-3.1-8B-Instruct

    # Use 70B for production:
    python -m experiments.scripts.diagnostic_signal_isolation --model meta-llama/Llama-3.1-70B-Instruct
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# Set HF token before imports
os.environ['HF_TOKEN'] = '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'

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class ComponentScores:
    """Per-token component scores for a single sample."""
    authority: torch.Tensor      # (S,) Authority flow from prompt
    varentropy: torch.Tensor     # (S,) Varentropy per token
    dispersion: torch.Tensor     # (S,) Semantic dispersion per token
    log_prob: torch.Tensor       # (S,) Log probability per token
    entropy: torch.Tensor        # (S,) Entropy per token
    tokens: List[str]            # Token strings
    response_start: int          # Index where response begins


def load_model_and_tokenizer(model_name: str):
    """Load model with proper configuration."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.environ['HF_TOKEN']
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        token=os.environ['HF_TOKEN']
    )
    model.eval()

    print(f"  Model loaded on {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


def load_halueval_qa(num_samples: int = 100, seed: int = 42):
    """Load HaluEval QA samples."""
    from datasets import load_dataset
    import random

    print(f"\nLoading HaluEval QA ({num_samples} samples)...")

    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")
    samples = list(dataset)
    random.seed(seed)
    random.shuffle(samples)
    samples = samples[:num_samples]

    # Count labels
    n_hall = sum(1 for s in samples if s.get('hallucination', 'no').lower() == 'yes')
    n_fact = len(samples) - n_hall
    print(f"  Hallucinations: {n_hall}, Facts: {n_fact}")

    return samples


def format_prompt_for_llama31(knowledge: str, question: str, answer: str, tokenizer) -> Tuple[str, str]:
    """
    Format HaluEval sample using Llama 3.1 chat template.

    Returns (full_text, prompt_only) for proper boundary detection.
    """
    # Build the prompt with knowledge context
    user_content = f"""Based on the following knowledge, answer the question.

Knowledge: {knowledge}

Question: {question}"""

    # Create messages for chat template
    messages = [
        {"role": "user", "content": user_content}
    ]

    # Get prompt with generation prompt (before answer)
    prompt_only = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Full text includes the answer
    full_text = prompt_only + answer

    return full_text, prompt_only


def compute_components(
    model,
    tokenizer,
    prompt: str,
    response: str,
    full_text: str,
    prompt_only: str,
) -> ComponentScores:
    """
    Compute all component scores for a sample.

    This bypasses the master equation to get raw signals.
    """
    from ag_sar.modeling import ModelAdapter
    from ag_sar.measures.entropy import compute_varentropy, compute_token_entropy
    from ag_sar.measures.semantics import compute_semantic_dispersion
    from ag_sar.ops import compute_authority_flow_vectorized

    device = next(model.parameters()).device

    # Tokenize
    prompt_enc = tokenizer(prompt_only, return_tensors='pt', add_special_tokens=False)
    full_enc = tokenizer(full_text, return_tensors='pt', add_special_tokens=False)

    input_ids = full_enc['input_ids'].to(device)
    attention_mask = torch.ones_like(input_ids)

    response_start = prompt_enc['input_ids'].size(1)
    seq_len = input_ids.size(1)

    # Get token strings
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    # Setup hooks to capture attention
    num_layers = model.config.num_hidden_layers
    semantic_layers = list(range(max(0, num_layers - 4), num_layers))

    adapter = ModelAdapter(
        model=model,
        layers=semantic_layers,
        dtype=torch.bfloat16,
    )
    adapter.register()

    try:
        # Forward pass
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits

        # Get embedding matrix for dispersion
        embed_matrix = model.get_output_embeddings().weight.detach()

        # Get attention from last semantic layer
        last_layer = semantic_layers[-1]
        attn_weights = adapter.capture.attention_weights.get(last_layer)

        if attn_weights is None:
            raise RuntimeError("Attention weights not captured")

        # 1. Authority Flow (raw, no gating)
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

        # 5. Log Probability (negative = uncertainty)
        log_probs = F.log_softmax(logits, dim=-1)
        # Shift: log_prob[i] = log P(token[i] | context)
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


def compute_auroc(scores: List[float], labels: List[int], higher_is_hall: bool = True) -> float:
    """
    Compute AUROC for a component.

    Args:
        scores: List of aggregated scores per sample
        labels: List of labels (1=hallucination, 0=fact)
        higher_is_hall: If True, higher score means more likely hallucination
                       If False, flip the sign
    """
    if len(set(labels)) < 2:
        return 0.5

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    if not higher_is_hall:
        scores_arr = -scores_arr

    try:
        return roc_auc_score(labels_arr, scores_arr)
    except:
        return 0.5


def run_question_1(model, tokenizer, samples: List[dict]) -> Dict[str, float]:
    """
    Question 1: Signal Isolation Check

    Compute AUROC for each component individually.
    """
    print("\n" + "=" * 70)
    print("QUESTION 1: Signal Isolation Check")
    print("=" * 70)
    print("Computing individual component AUROCs on HaluEval QA...")
    print("(Higher AUROC = component correctly identifies hallucinations)\n")

    # Collect scores
    authority_scores = []
    varentropy_scores = []
    dispersion_scores = []
    logprob_scores = []
    entropy_scores = []
    labels = []

    for i, sample in enumerate(samples):
        label = 1 if sample.get('hallucination', 'no').lower() == 'yes' else 0
        labels.append(label)

        knowledge = sample['knowledge']
        question = sample['question']
        answer = sample['answer']

        full_text, prompt_only = format_prompt_for_llama31(
            knowledge, question, answer, tokenizer
        )

        try:
            scores = compute_components(
                model, tokenizer,
                f"Knowledge: {knowledge}\n\nQuestion: {question}",
                answer,
                full_text,
                prompt_only,
            )

            # Aggregate over response tokens only (mean)
            resp_slice = slice(scores.response_start, None)

            authority_scores.append(scores.authority[resp_slice].mean().item())
            varentropy_scores.append(scores.varentropy[resp_slice].mean().item())
            dispersion_scores.append(scores.dispersion[resp_slice].mean().item())
            logprob_scores.append(scores.log_prob[resp_slice].mean().item())
            entropy_scores.append(scores.entropy[resp_slice].mean().item())

            label_str = "HALL" if label == 1 else "FACT"
            print(f"  [{i+1:3d}/{len(samples)}] {label_str} | "
                  f"A={authority_scores[-1]:.3f} V={varentropy_scores[-1]:.3f} "
                  f"D={dispersion_scores[-1]:.3f} LogP={logprob_scores[-1]:.2f}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(samples)}] ERROR: {str(e)[:50]}")
            # Remove label if we couldn't process
            labels.pop()

    # Compute AUROCs
    print("\n" + "-" * 70)
    print("INDIVIDUAL COMPONENT AUROCs:")
    print("-" * 70)

    results = {}

    # Authority: Higher authority = more grounded = LESS hallucination
    # So if AUROC < 0.5, it means hallucinations have HIGHER authority (inverted!)
    auroc_a = compute_auroc(authority_scores, labels, higher_is_hall=True)
    results['authority'] = auroc_a
    direction_a = "INVERTED!" if auroc_a < 0.5 else "correct" if auroc_a > 0.5 else "random"
    print(f"  Authority (A):     AUROC = {auroc_a:.4f}  [{direction_a}]")
    print(f"                     (Higher A should mean LESS hallucination)")

    # Varentropy: Higher varentropy = more uncertainty = MORE hallucination expected
    # But our hypothesis is: Hall has LOW V, Fact has HIGH V (INVERTED)
    auroc_v = compute_auroc(varentropy_scores, labels, higher_is_hall=True)
    results['varentropy'] = auroc_v
    direction_v = "INVERTED!" if auroc_v < 0.5 else "correct" if auroc_v > 0.5 else "random"
    print(f"  Varentropy (V):    AUROC = {auroc_v:.4f}  [{direction_v}]")
    print(f"                     (If <0.5: Hall has LOW V, Fact has HIGH V)")

    # Dispersion: Higher dispersion = less consistent = MORE hallucination expected
    auroc_d = compute_auroc(dispersion_scores, labels, higher_is_hall=True)
    results['dispersion'] = auroc_d
    direction_d = "INVERTED!" if auroc_d < 0.5 else "correct" if auroc_d > 0.5 else "random"
    print(f"  Dispersion (D):    AUROC = {auroc_d:.4f}  [{direction_d}]")
    print(f"                     (Higher D should mean MORE hallucination)")

    # Log Probability: Lower log prob = less confident = MORE hallucination expected
    # Note: We use higher_is_hall=False because lower logprob = more uncertainty
    auroc_lp = compute_auroc(logprob_scores, labels, higher_is_hall=False)
    results['logprob'] = auroc_lp
    direction_lp = "INVERTED!" if auroc_lp < 0.5 else "correct" if auroc_lp > 0.5 else "random"
    print(f"  LogProb (P):       AUROC = {auroc_lp:.4f}  [{direction_lp}]")
    print(f"                     (Lower P should mean MORE hallucination)")

    # Entropy: Higher entropy = more uncertain = MORE hallucination expected
    auroc_e = compute_auroc(entropy_scores, labels, higher_is_hall=True)
    results['entropy'] = auroc_e
    direction_e = "INVERTED!" if auroc_e < 0.5 else "correct" if auroc_e > 0.5 else "random"
    print(f"  Entropy (H):       AUROC = {auroc_e:.4f}  [{direction_e}]")

    # Summary statistics
    print("\n" + "-" * 70)
    print("COMPONENT STATISTICS (mean ± std):")
    print("-" * 70)

    hall_mask = [l == 1 for l in labels]
    fact_mask = [l == 0 for l in labels]

    for name, scores in [
        ("Authority", authority_scores),
        ("Varentropy", varentropy_scores),
        ("Dispersion", dispersion_scores),
        ("LogProb", logprob_scores),
        ("Entropy", entropy_scores),
    ]:
        hall_scores = [s for s, m in zip(scores, hall_mask) if m]
        fact_scores = [s for s, m in zip(scores, fact_mask) if m]

        if hall_scores and fact_scores:
            hall_mean, hall_std = np.mean(hall_scores), np.std(hall_scores)
            fact_mean, fact_std = np.mean(fact_scores), np.std(fact_scores)
            diff = hall_mean - fact_mean

            print(f"  {name:12s}: Hall={hall_mean:7.3f}±{hall_std:.3f}  "
                  f"Fact={fact_mean:7.3f}±{fact_std:.3f}  "
                  f"Δ={diff:+.3f}")

    return results


def run_question_2(model, tokenizer, samples: List[dict], num_examples: int = 3):
    """
    Question 2: Token-Level Dynamics

    Print per-token arrays for hallucinated samples.
    """
    print("\n" + "=" * 70)
    print("QUESTION 2: Token-Level Dynamics")
    print("=" * 70)
    print("Showing per-token Varentropy and Authority for hallucinated samples.\n")

    # Find hallucinated samples
    hall_samples = [(i, s) for i, s in enumerate(samples)
                    if s.get('hallucination', 'no').lower() == 'yes'][:num_examples]

    # Find factual samples for comparison
    fact_samples = [(i, s) for i, s in enumerate(samples)
                    if s.get('hallucination', 'no').lower() == 'no'][:num_examples]

    for label_name, sample_list in [("HALLUCINATION", hall_samples), ("FACT", fact_samples)]:
        print(f"\n{'='*70}")
        print(f"  {label_name} SAMPLES")
        print(f"{'='*70}")

        for idx, sample in sample_list:
            print(f"\n--- Sample {idx} ({label_name}) ---")
            print(f"Knowledge: {sample['knowledge'][:100]}...")
            print(f"Question: {sample['question']}")
            print(f"Answer: {sample['answer']}")

            full_text, prompt_only = format_prompt_for_llama31(
                sample['knowledge'],
                sample['question'],
                sample['answer'],
                tokenizer
            )

            try:
                scores = compute_components(
                    model, tokenizer,
                    f"Knowledge: {sample['knowledge']}\n\nQuestion: {sample['question']}",
                    sample['answer'],
                    full_text,
                    prompt_only,
                )

                print(f"\nResponse tokens (starting at position {scores.response_start}):")
                print("-" * 80)
                print(f"{'Pos':>4} {'Token':>15} {'Varentropy':>12} {'Authority':>12} {'LogProb':>10} {'Entropy':>10}")
                print("-" * 80)

                # Show response tokens
                for i in range(scores.response_start, len(scores.tokens)):
                    token = scores.tokens[i]
                    v = scores.varentropy[i].item()
                    a = scores.authority[i].item()
                    lp = scores.log_prob[i].item()
                    e = scores.entropy[i].item()

                    # Highlight high/low varentropy
                    v_flag = "**" if v > 5.0 else "  " if v > 3.0 else "!!"

                    # Clean token for display
                    token_disp = repr(token)[1:-1][:15]

                    print(f"{i:4d} {token_disp:>15} {v:12.4f}{v_flag} {a:12.4f} {lp:10.4f} {e:10.4f}")

                # Aggregates
                resp_v = scores.varentropy[scores.response_start:].mean().item()
                resp_a = scores.authority[scores.response_start:].mean().item()
                max_v = scores.varentropy[scores.response_start:].max().item()
                min_v = scores.varentropy[scores.response_start:].min().item()

                print("-" * 80)
                print(f"Response Aggregates: V_mean={resp_v:.4f} V_max={max_v:.4f} V_min={min_v:.4f} A_mean={resp_a:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")


def run_question_3(tokenizer, samples: List[dict]):
    """
    Question 3: Instruction Tuning Factor

    Verify prompt template matches Llama 3.1 expectations.
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: Instruction Tuning Factor")
    print("=" * 70)

    print("\n1. Expected Llama 3.1 Instruct Format:")
    print("-" * 50)

    # Show expected format
    test_messages = [{"role": "user", "content": "Test question"}]
    expected = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    print(expected)

    print("\n2. HaluEval Prompt Format (what we're using):")
    print("-" * 50)

    sample = samples[0]
    full_text, prompt_only = format_prompt_for_llama31(
        sample['knowledge'],
        sample['question'],
        sample['answer'],
        tokenizer
    )
    print(prompt_only)

    print("\n3. Template Match Analysis:")
    print("-" * 50)

    # Check for key components
    checks = [
        ("<|begin_of_text|>", "BOS token"),
        ("<|start_header_id|>user<|end_header_id|>", "User header"),
        ("<|start_header_id|>assistant<|end_header_id|>", "Assistant header"),
        ("<|eot_id|>", "End-of-turn token"),
    ]

    for token, desc in checks:
        present = token in prompt_only
        status = "✓" if present else "✗ MISSING!"
        print(f"  {status} {desc}: {token}")

    print("\n4. Potential Issues:")
    print("-" * 50)

    # HaluEval was generated with GPT-3.5, not Llama
    print("  - HaluEval was generated with GPT-3.5/4, NOT Llama")
    print("  - Answers may use different phrasing than Llama would generate")
    print("  - Knowledge context format differs from Llama's training distribution")
    print("  - Consider: Are we evaluating Llama's uncertainty or GPT's writing style?")


def run_question_4(samples: List[dict], tokenizer):
    """
    Question 4: Length Bias

    Analyze length distribution differences.
    """
    print("\n" + "=" * 70)
    print("QUESTION 4: Length Bias Analysis")
    print("=" * 70)

    hall_lengths = []
    fact_lengths = []

    for sample in samples:
        answer = sample['answer']
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        length = len(tokens)

        if sample.get('hallucination', 'no').lower() == 'yes':
            hall_lengths.append(length)
        else:
            fact_lengths.append(length)

    print("\nResponse Length Distribution (in tokens):")
    print("-" * 50)

    if hall_lengths and fact_lengths:
        hall_mean, hall_std = np.mean(hall_lengths), np.std(hall_lengths)
        fact_mean, fact_std = np.mean(fact_lengths), np.std(fact_lengths)

        print(f"  Hallucinations: {hall_mean:.1f} ± {fact_std:.1f} tokens (n={len(hall_lengths)})")
        print(f"  Facts:          {fact_mean:.1f} ± {fact_std:.1f} tokens (n={len(fact_lengths)})")
        print(f"  Difference:     {hall_mean - fact_mean:+.1f} tokens")

        print(f"\n  Hall range: [{min(hall_lengths)}, {max(hall_lengths)}]")
        print(f"  Fact range: [{min(fact_lengths)}, {max(fact_lengths)}]")

        # Length correlation with potential bias
        if abs(hall_mean - fact_mean) > 5:
            print(f"\n  ⚠️  SIGNIFICANT LENGTH BIAS DETECTED!")
            print(f"      Consider using 'first_k_tokens' or 'length_normalized' aggregation")
        else:
            print(f"\n  ✓ Length distributions are similar")


def run_question_5(samples: List[dict], model, tokenizer, num_inspect: int = 5):
    """
    Question 5: Ground Truth Integrity

    Manual inspection of high-uncertainty facts and low-uncertainty hallucinations.
    """
    print("\n" + "=" * 70)
    print("QUESTION 5: Ground Truth Integrity")
    print("=" * 70)
    print("Inspecting samples for label quality issues...\n")

    # Compute scores for all samples
    sample_scores = []

    for i, sample in enumerate(samples[:50]):  # First 50 for speed
        label = 1 if sample.get('hallucination', 'no').lower() == 'yes' else 0

        full_text, prompt_only = format_prompt_for_llama31(
            sample['knowledge'],
            sample['question'],
            sample['answer'],
            tokenizer
        )

        try:
            scores = compute_components(
                model, tokenizer,
                f"Knowledge: {sample['knowledge']}\n\nQuestion: {sample['question']}",
                sample['answer'],
                full_text,
                prompt_only,
            )

            resp_v = scores.varentropy[scores.response_start:].mean().item()
            resp_a = scores.authority[scores.response_start:].mean().item()

            sample_scores.append({
                'idx': i,
                'sample': sample,
                'label': label,
                'varentropy': resp_v,
                'authority': resp_a,
            })
        except:
            pass

    # Find suspicious samples
    print("1. Potential FALSE NEGATIVES (Hallucinations labeled as Facts):")
    print("   (Facts with unusually LOW varentropy - model is 'confidently wrong')")
    print("-" * 70)

    facts = [s for s in sample_scores if s['label'] == 0]
    facts_sorted = sorted(facts, key=lambda x: x['varentropy'])

    for s in facts_sorted[:num_inspect]:
        print(f"\n  Sample {s['idx']} (labeled FACT, V={s['varentropy']:.4f}, A={s['authority']:.4f})")
        print(f"  Q: {s['sample']['question']}")
        print(f"  A: {s['sample']['answer'][:200]}...")
        print(f"  K: {s['sample']['knowledge'][:150]}...")

    print("\n\n2. Potential FALSE POSITIVES (Facts labeled as Hallucinations):")
    print("   (Hallucinations with unusually HIGH varentropy - model is 'appropriately uncertain')")
    print("-" * 70)

    halls = [s for s in sample_scores if s['label'] == 1]
    halls_sorted = sorted(halls, key=lambda x: -x['varentropy'])

    for s in halls_sorted[:num_inspect]:
        print(f"\n  Sample {s['idx']} (labeled HALL, V={s['varentropy']:.4f}, A={s['authority']:.4f})")
        print(f"  Q: {s['sample']['question']}")
        print(f"  A: {s['sample']['answer'][:200]}...")
        print(f"  K: {s['sample']['knowledge'][:150]}...")

    print("\n\n3. Most Confident Hallucinations (model is very wrong):")
    print("-" * 70)

    halls_confident = sorted(halls, key=lambda x: x['varentropy'])

    for s in halls_confident[:num_inspect]:
        print(f"\n  Sample {s['idx']} (labeled HALL, V={s['varentropy']:.4f} LOW!)")
        print(f"  Q: {s['sample']['question']}")
        print(f"  A: {s['sample']['answer'][:200]}...")


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Signal Isolation Diagnostic")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use (default: Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of questions to run (default: all)"
    )

    args = parser.parse_args()
    questions = [int(q) for q in args.questions.split(",")]

    print("=" * 70)
    print("AG-SAR SIGNAL ISOLATION DIAGNOSTIC")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples: {args.num_samples}")
    print(f"Questions: {questions}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model)

    # Load data
    samples = load_halueval_qa(num_samples=args.num_samples)

    # Run requested questions
    results = {}

    if 1 in questions:
        results['q1'] = run_question_1(model, tokenizer, samples)

    if 2 in questions:
        run_question_2(model, tokenizer, samples, num_examples=2)

    if 3 in questions:
        run_question_3(tokenizer, samples)

    if 4 in questions:
        run_question_4(samples, tokenizer)

    if 5 in questions:
        run_question_5(samples, model, tokenizer, num_inspect=3)

    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    if 'q1' in results:
        print("\nKey Findings (Question 1):")
        for comp, auroc in results['q1'].items():
            status = "INVERTED" if auroc < 0.5 else "correct" if auroc > 0.55 else "weak"
            print(f"  {comp:12s}: AUROC={auroc:.4f} ({status})")


if __name__ == "__main__":
    main()
