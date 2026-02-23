"""
Answer matching utilities for QA evaluation.

Standard SQuAD-style F1 matching with answer extraction for verbose model responses.
"""

import re
import string
from collections import Counter
from typing import List

F1_HALLUCINATION_THRESHOLD = 0.3


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_short_answer(text: str) -> str:
    """Extract the core answer from a verbose model response.

    Chat models often generate 'Answer\\n\\nExplanation: ...' or repeat the question.
    Extract just the first meaningful line/sentence.
    """
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


def compute_adaptive_f1_threshold(f1_scores: List[float]) -> float:
    """Otsu threshold on F1 scores to separate correct from hallucinated.

    Falls back to F1_HALLUCINATION_THRESHOLD if too few samples or
    if Otsu returns an implausible value outside [0.1, 0.7].
    """
    if len(f1_scores) < 20:
        return F1_HALLUCINATION_THRESHOLD
    from ag_sar.numerics import otsu_threshold
    threshold = otsu_threshold(f1_scores)
    if threshold < 0.1 or threshold > 0.7:
        return F1_HALLUCINATION_THRESHOLD
    return threshold


def max_f1_score(prediction: str, ground_truths: List[str]) -> float:
    if not prediction.strip() or not ground_truths:
        return 0.0
    short = extract_short_answer(prediction)
    raw_f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    short_f1 = max(compute_f1(short, gt) for gt in ground_truths)
    return max(raw_f1, short_f1)
