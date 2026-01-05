#!/usr/bin/env python3
"""
RAGTruth Dataset Preparation Script.

Downloads wandb/RAGTruth-processed and formats prompts with Llama-3.1 chat template
for optimal AG-SAR evaluation. Saves to local JSONL for reproducibility.

Key Features:
- Applies Llama-3.1-Instruct chat template for consistent formatting
- Filters refusals (samples where model incorrectly refused)
- Supports task-type filtering (QA, Summary, Data2txt)
- Produces balanced statistics report

Usage:
    python scripts/prepare_ragtruth.py                    # All tasks
    python scripts/prepare_ragtruth.py --task-type qa     # QA only
    python scripts/prepare_ragtruth.py --task-type summary --num-samples 500
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set HF token
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
os.environ["HF_TOKEN"] = HF_TOKEN


# Llama-3.1-Instruct chat template
LLAMA_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. Only use information from the context to answer. If the context doesn't contain the answer, say so."""

def format_llama_prompt(context: str, query: str, task_type: str) -> str:
    """
    Format prompt using Llama-3.1-Instruct chat template.

    Args:
        context: The reference context/document
        query: The question or instruction
        task_type: 'QA', 'Summary', or 'Data2txt'

    Returns:
        Formatted prompt string
    """
    # Task-specific instructions
    if task_type == "Summary":
        user_content = f"""Summarize the following document:

{context}"""
    elif task_type == "Data2txt":
        user_content = f"""Convert the following data to natural language:

{context}

{query}"""
    else:  # QA
        user_content = f"""Context: {context}

Question: {query}

Answer the question based only on the provided context."""

    # Llama-3.1 chat format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{LLAMA_SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Prepare RAGTruth Dataset")
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["qa", "summary", "data2text", "all"],
        default="all",
        help="Task type filter (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum samples per task type (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--no-filter-refusals",
        action="store_true",
        help="Don't filter incorrect refusals",
    )
    args = parser.parse_args()

    # Map task type
    task_type_map = {
        "qa": "QA",
        "summary": "Summary",
        "data2text": "Data2txt",
        "all": None,
    }
    target_task = task_type_map.get(args.task_type)

    print("=" * 60)
    print("RAGTruth Dataset Preparation")
    print("=" * 60)
    print(f"Task Filter: {args.task_type}")
    print(f"Max Samples: {args.num_samples or 'all'}")
    print(f"Filter Refusals: {not args.no_filter_refusals}")
    print()

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("Loading wandb/RAGTruth-processed...")
    dataset = load_dataset("wandb/RAGTruth-processed", split="test", token=HF_TOKEN)
    print(f"Loaded {len(dataset)} total samples")
    print()

    # Filter by task type
    if target_task:
        dataset = dataset.filter(lambda x: x.get("task_type") == target_task)
        print(f"After task filter ({target_task}): {len(dataset)} samples")

    # Process samples
    samples = []
    stats = {
        "total": 0,
        "filtered_refusals": 0,
        "hallucinated": 0,
        "faithful": 0,
        "by_task": Counter(),
        "by_model": Counter(),
        "by_hallucination_type": Counter(),
    }

    print("\nProcessing samples...")
    for row in tqdm(dataset, desc="Processing"):
        stats["total"] += 1

        # Refusal filtering
        quality = row.get("quality", "good")
        if not args.no_filter_refusals and quality == "incorrect_refusal":
            stats["filtered_refusals"] += 1
            continue

        # Extract fields
        task_type = row.get("task_type", "QA")
        query = row.get("query", "")
        context = row.get("context", "")
        response = row.get("output", "")
        model = row.get("model", "unknown")

        # Determine hallucination label
        labels = row.get("hallucination_labels_processed", {})
        evident_conflict = labels.get("evident_conflict", 0)
        baseless_info = labels.get("baseless_info", 0)
        is_hallucination = (evident_conflict > 0) or (baseless_info > 0)
        label = 1 if is_hallucination else 0

        # Format prompt with Llama template
        prompt = format_llama_prompt(context, query, task_type)

        # Create sample
        sample = {
            "id": f"ragtruth_{stats['total']}",
            "prompt": prompt,
            "response": response,
            "label": label,
            "task_type": task_type,
            "model": model,
            "quality": quality,
            "evident_conflict": evident_conflict,
            "baseless_info": baseless_info,
        }
        samples.append(sample)

        # Update stats
        if label == 1:
            stats["hallucinated"] += 1
            if evident_conflict > 0:
                stats["by_hallucination_type"]["evident_conflict"] += 1
            if baseless_info > 0:
                stats["by_hallucination_type"]["baseless_info"] += 1
        else:
            stats["faithful"] += 1

        stats["by_task"][task_type] += 1
        stats["by_model"][model] += 1

        # Check sample limit
        if args.num_samples and len(samples) >= args.num_samples:
            break

    # Save to JSONL
    suffix = f"_{args.task_type}" if args.task_type != "all" else ""
    output_file = output_dir / f"ragtruth_clean{suffix}.jsonl"

    print(f"\nSaving to {output_file}...")
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total Raw Samples: {stats['total']}")
    print(f"Filtered Refusals: {stats['filtered_refusals']}")
    print(f"Final Samples: {len(samples)}")
    print()
    print(f"Label Distribution:")
    print(f"  - Hallucinated (1): {stats['hallucinated']} ({100*stats['hallucinated']/len(samples):.1f}%)")
    print(f"  - Faithful (0): {stats['faithful']} ({100*stats['faithful']/len(samples):.1f}%)")
    print()
    print("By Task Type:")
    for task, count in sorted(stats["by_task"].items()):
        print(f"  - {task}: {count}")
    print()
    print("By Source Model:")
    for model, count in sorted(stats["by_model"].items(), key=lambda x: -x[1])[:5]:
        print(f"  - {model}: {count}")
    print()
    print("Hallucination Types:")
    for htype, count in stats["by_hallucination_type"].items():
        print(f"  - {htype}: {count}")
    print()

    # Class imbalance warning
    imbalance = abs(stats["hallucinated"] - stats["faithful"]) / len(samples)
    if imbalance > 0.3:
        print("WARNING: Significant class imbalance detected!")
        print("  RAGTruth has more faithful than hallucinated samples.")
        print("  AUPRC is the key metric for imbalanced data.")
    print()

    print(f"Output saved to: {output_file}")
    print("\nReady for benchmark!")
    print(f"  python scripts/launch_ragtruth_h2h.py --data-file {output_file}")


if __name__ == "__main__":
    main()
