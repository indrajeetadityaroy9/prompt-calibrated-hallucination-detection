import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.loaders import load_samples
from experiments.schema import ModelConfig

PROMPT_TEMPLATE = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"


def load_model(config: ModelConfig, seed: int) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    token = os.environ.get("HF_TOKEN")
    print(f"Loading model: {config.name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.name, token=token)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.name, dtype=getattr(torch, config.dtype), device_map="auto", token=token, attn_implementation=config.attn_implementation)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model, tokenizer


def save_results(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Results saved to {path}")
