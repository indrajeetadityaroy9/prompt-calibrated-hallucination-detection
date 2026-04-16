import re
import string
from collections import Counter

import numpy as np

from src.numerics import _otsu_internals

_PUNC = set(string.punctuation)


def normalize_answer(s: str) -> str:
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', ''.join(ch for ch in s.lower() if ch not in _PUNC)).split())


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    num_same = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if num_same == 0:
        return 0.0
    return 2 * num_same / (len(pred_tokens) + len(gt_tokens))


def extract_short_answer(text: str) -> str:
    text = text.strip()
    for sep in ["\n\n", "\nExplanation:", "\nQuestion:", "\nNote:", "\nContext:"]:
        if sep in text:
            text = text[:text.index(sep)].strip()
    if len(text.split()) > 20:
        for end in [". ", ".\n"]:
            if end in text:
                text = text[:text.index(end) + 1].strip()
                break
    return text


def compute_adaptive_f1_threshold(f1_scores: list[float]) -> float:
    values = np.asarray(f1_scores, dtype=float)
    best_idx, sorted_vals, _, _ = _otsu_internals(values)
    return float(0.5 * (sorted_vals[best_idx] + sorted_vals[best_idx + 1]))


def max_f1_score(prediction: str, ground_truths: list[str]) -> float:
    short = extract_short_answer(prediction)
    return max(max(compute_f1(prediction, gt) for gt in ground_truths), max(compute_f1(short, gt) for gt in ground_truths))
