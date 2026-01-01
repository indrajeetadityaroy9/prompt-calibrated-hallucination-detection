"""ROUGE score computation for text similarity."""


def compute_rouge_l(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L score between prediction and reference.

    ROUGE-L measures longest common subsequence (LCS) similarity,
    useful for evaluating factual consistency between generated
    responses and reference answers.

    Args:
        prediction: Generated/predicted text
        reference: Reference/ground truth text

    Returns:
        ROUGE-L F-measure score between 0 and 1
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Fallback: simple overlap ratio when rouge_score not installed
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if len(pred_tokens) == 0:
            return 0.0
        overlap = len(pred_tokens & ref_tokens)
        return overlap / max(len(pred_tokens), len(ref_tokens))
