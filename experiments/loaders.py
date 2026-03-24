def load_triviaqa(n_samples: int, max_context_chars: int) -> list[dict]:
    from datasets import load_dataset
    print("Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc", split="validation")

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


def load_squad(n_samples: int, max_context_chars: int = 0) -> list[dict]:
    from datasets import load_dataset
    print("Loading SQuAD v2...")
    dataset = load_dataset("squad_v2", split="validation")

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
