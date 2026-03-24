import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
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
    from ag_sar.numerics import otsu_threshold
    return otsu_threshold(f1_scores)


def max_f1_score(prediction: str, ground_truths: list[str]) -> float:
    short = extract_short_answer(prediction)
    raw_f1 = max(compute_f1(prediction, gt) for gt in ground_truths)
    short_f1 = max(compute_f1(short, gt) for gt in ground_truths)
    return max(raw_f1, short_f1)
