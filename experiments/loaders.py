from functools import partial

from datasets import load_dataset

_SCHEMA = {"question", "answers", "context"}


def _triviaqa_context(item, max_chars):
    for key, subkey in [("search_results", "search_context"), ("entity_pages", "wiki_context")]:
        ctx = (item.get(key) or {}).get(subkey)
        if ctx and ctx[0]:
            return ctx[0][:max_chars]
    return ""


def _map_triviaqa(example, max_chars):
    return {"question": example["question"], "answers": list(set(example["answer"]["aliases"] + [example["answer"]["value"]])), "context": _triviaqa_context(example, max_chars)}


def _map_squad(example):
    return {"question": example["question"], "answers": list(set(example["answers"]["text"])), "context": example["context"]}


def load_samples(name, n_samples, max_context_chars, streaming=False):
    print(f"Loading {name}...")
    if name == "triviaqa":
        ds = load_dataset("trivia_qa", "rc", split="validation", streaming=streaming)
        ds = ds.filter(lambda x: _triviaqa_context(x, max_context_chars) != "")
        map_fn = partial(_map_triviaqa, max_chars=max_context_chars)
    else:
        ds = load_dataset("squad_v2", split="validation", streaming=streaming)
        ds = ds.filter(lambda x: len(x["answers"]["text"]) > 0)
        map_fn = _map_squad

    drop = [c for c in ds.column_names if c not in _SCHEMA]

    if streaming:
        ds = ds.map(map_fn, remove_columns=drop)
        samples = list(ds.take(n_samples))
    else:
        ds = ds.select(range(n_samples))
        ds = ds.map(map_fn, remove_columns=drop)
        samples = list(ds)

    print(f"Loaded {len(samples)} {name} samples")
    return samples
