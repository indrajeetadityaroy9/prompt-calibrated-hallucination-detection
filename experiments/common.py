"""Shared experiment utilities — model loading, dataset dispatch, output."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loaders import load_triviaqa, load_squad

from .schema import ModelConfig

DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
}

_VALID_DTYPES = {"float16", "bfloat16", "float32"}


def load_model(config: ModelConfig, seed: int) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load HF model and tokenizer."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    token = os.environ.get("HF_TOKEN")
    if config.torch_dtype not in _VALID_DTYPES:
        raise ValueError(f"Invalid torch_dtype: {config.torch_dtype!r}. Expected one of {_VALID_DTYPES}")
    torch_dtype = getattr(torch, config.torch_dtype)

    print(f"Loading model: {config.name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=token,
        attn_implementation=config.attn_implementation,
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    return model, tokenizer


def load_dataset(name: str, n_samples: int, max_context_chars: int) -> list[dict]:
    """Dispatch to dataset-specific loader."""
    loader = DATASET_LOADERS[name]
    if name == "triviaqa":
        return loader(n_samples=n_samples, max_context_chars=max_context_chars)
    return loader(n_samples=n_samples)


def save_results(data: dict, path: str) -> None:
    """Write JSON results to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {path}")
