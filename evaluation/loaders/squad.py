"""SQuAD v2 dataset loader."""

from typing import Dict, List


def load_squad(n_samples: int = 100) -> List[Dict]:
    from datasets import load_dataset
    print("Loading SQuAD v2...")
    dataset = load_dataset("squad_v2", split="validation", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        if not item["answers"]["text"]:
            continue
        samples.append({
            "question": item["question"],
            "answers": list(set(item["answers"]["text"])),
            "context": item["context"],
            "dataset": "squad",
        })

    print(f"Loaded {len(samples)} SQuAD samples")
    return samples
