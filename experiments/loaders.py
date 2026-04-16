from datasets import load_dataset, Dataset


def _extract_triviaqa_context(item: dict, max_chars: int) -> str:
    for key, subkey in [("search_results", "search_context"), ("entity_pages", "wiki_context")]:
        ctx_list = (item.get(key) or {}).get(subkey)
        if ctx_list and ctx_list[0]:
            return ctx_list[0][:max_chars]
    return ""


def _normalize_triviaqa(ds: Dataset, max_context_chars: int) -> list[dict]:
    return [
        {
            "question": row["question"],
            "answers": list(set(row["answer"]["aliases"] + [row["answer"]["value"]])),
            "context": _extract_triviaqa_context(row, max_context_chars),
        }
        for row in ds
    ]


def _normalize_squad(ds: Dataset) -> list[dict]:
    return [
        {
            "question": row["question"],
            "answers": list(set(row["answers"]["text"])),
            "context": row["context"],
        }
        for row in ds
    ]


def load_triviaqa(n_samples: int, max_context_chars: int) -> list[dict]:
    print("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc", split="validation")
    ds = ds.filter(lambda x: _extract_triviaqa_context(x, max_context_chars) != "")
    if len(ds) > n_samples:
        ds = ds.select(range(n_samples))
    samples = _normalize_triviaqa(ds, max_context_chars)
    del ds
    print(f"Loaded {len(samples)} TriviaQA samples")
    return samples


def load_squad(n_samples: int, max_context_chars: int) -> list[dict]:
    print("Loading SQuAD v2...")
    ds = load_dataset("squad_v2", split="validation")
    ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
    if len(ds) > n_samples:
        ds = ds.select(range(n_samples))
    samples = _normalize_squad(ds)
    del ds
    print(f"Loaded {len(samples)} SQuAD samples")
    return samples
