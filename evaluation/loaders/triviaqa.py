"""TriviaQA dataset loader."""

from typing import Dict, List


def load_triviaqa(n_samples: int = 100, max_context_chars: int = 2000) -> List[Dict]:
    from datasets import load_dataset
    print("Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc", split="validation", trust_remote_code=True)

    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        question = item["question"]
        answers = list(set(item["answer"]["aliases"] + [item["answer"]["value"]]))

        context = ""
        if item.get("search_results") and item["search_results"].get("search_context"):
            ctx_list = item["search_results"]["search_context"]
            if ctx_list:
                context = ctx_list[0][:max_context_chars]
        if not context and item.get("entity_pages") and item["entity_pages"].get("wiki_context"):
            ctx_list = item["entity_pages"]["wiki_context"]
            if ctx_list:
                context = ctx_list[0][:max_context_chars]

        if not context:
            continue

        samples.append({"question": question, "answers": answers, "context": context, "dataset": "triviaqa"})

    print(f"Loaded {len(samples)} TriviaQA samples")
    return samples
