"""
Input tokenization for AG-SAR detection.

Builds the "Context: {ctx}\\n\\nQuestion: {q}\\n\\nAnswer:" prompt format
and returns token positions for context span identification.
"""

from typing import Any, Tuple
import torch
from torch import Tensor


def build_input(
    tokenizer: Any,
    context: str,
    question: str,
    device: torch.device,
) -> Tuple[Tensor, int, int, int]:
    """
    Build input_ids from context and question.

    Returns (input_ids, context_start, context_end, prompt_len).
    """
    prefix = tokenizer.encode("Context: ", add_special_tokens=False)
    ctx_tokens = tokenizer.encode(context, add_special_tokens=False) if context else []
    sep = tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
    q_tokens = tokenizer.encode(question, add_special_tokens=False)
    suffix = tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

    bos = tokenizer.bos_token_id
    bos_len = 1 if bos is not None else 0

    context_start = bos_len + len(prefix)
    context_end = context_start + len(ctx_tokens)

    tokens = (([bos] if bos is not None else [])
              + prefix + ctx_tokens + sep + q_tokens + suffix)

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return input_ids, context_start, context_end, len(tokens)
