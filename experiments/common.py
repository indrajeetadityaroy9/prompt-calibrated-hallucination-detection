"""Shared experiment utilities — model loading, dataset dispatch, output."""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loaders import load_triviaqa, load_squad

from .schema import ModelConfig

DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
}


def load_model(config: ModelConfig, seed: int) -> Tuple:
    """Load HF model and tokenizer."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    token = os.environ.get("HF_TOKEN")
    torch_dtype = getattr(torch, config.torch_dtype)

    print(f"Loading model: {config.name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.name, token=token)
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


def load_dataset(name: str, n_samples: int, max_context_chars: int) -> List[Dict]:
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
