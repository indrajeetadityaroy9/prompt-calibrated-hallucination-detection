"""
POS-tag correlation metrics for mechanistic verification.

Tests whether AG-SAR relevance scores correlate with
information-carrying parts of speech (nouns, verbs, etc.)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.stats import spearmanr
import re

# Lazy-loaded spaCy model
_nlp = None


def _get_nlp():
    """Lazy load spaCy model (expensive, only load once)."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Run: python -m spacy download en_core_web_sm")
    return _nlp


def clean_gpt2_token(token: str) -> str:
    """
    Clean GPT-2 token for plain text comparison.

    GPT-2 uses:
    - 'Ġ' (Unicode 0x0120) as space prefix
    - 'Ċ' (Unicode 0x010A) as newline

    Args:
        token: GPT-2 tokenized string

    Returns:
        Cleaned token suitable for spaCy comparison
    """
    # Replace GPT-2 special characters
    cleaned = token.replace('Ġ', '').replace('Ċ', '')
    return cleaned


def detokenize_gpt2(tokens: List[str]) -> str:
    """
    Convert GPT-2 tokens back to readable text.

    Args:
        tokens: List of GPT-2 tokens

    Returns:
        Reconstructed text string
    """
    text = ''
    for token in tokens:
        if token.startswith('Ġ'):
            # Space-prefixed token
            text += ' ' + token[1:]
        elif token.startswith('Ċ'):
            # Newline-prefixed token
            text += '\n' + token[1:]
        else:
            text += token
    return text.strip()


def get_pos_tags(text: str, tokenizer=None) -> List[Tuple[str, str]]:
    """
    Get POS tags for text using spaCy.

    Args:
        text: Input text
        tokenizer: Optional tokenizer for alignment

    Returns:
        List of (token, pos_tag) tuples
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]


def align_tokens_to_pos(
    gpt2_tokens: List[str],
    content_pos: Optional[List[str]] = None
) -> List[str]:
    """
    Align GPT-2 tokens to POS tags by character offset matching.

    This handles:
    - Subword tokenization (GPT-2 splits words into BPE pieces)
    - Space markers (Ġ prefix in GPT-2)
    - Punctuation handling

    Args:
        gpt2_tokens: List of GPT-2 tokens
        content_pos: POS tags to check for

    Returns:
        List of POS tags (one per GPT-2 token)
    """
    if content_pos is None:
        content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

    nlp = _get_nlp()

    # Reconstruct text from tokens
    text = detokenize_gpt2(gpt2_tokens)
    doc = nlp(text)

    # Build character offset to spaCy token mapping
    char_to_spacy = {}
    for spacy_token in doc:
        for i in range(spacy_token.idx, spacy_token.idx + len(spacy_token.text)):
            char_to_spacy[i] = spacy_token

    # Map each GPT-2 token to POS by finding its position in text
    pos_tags = []
    char_offset = 0

    for gpt2_token in gpt2_tokens:
        # Account for space prefix
        if gpt2_token.startswith('Ġ'):
            char_offset += 1  # Skip the space
            clean_token = gpt2_token[1:]
        elif gpt2_token.startswith('Ċ'):
            char_offset += 1  # Skip the newline
            clean_token = gpt2_token[1:]
        else:
            clean_token = gpt2_token

        # Find the spaCy token at this position
        if char_offset in char_to_spacy:
            spacy_token = char_to_spacy[char_offset]
            pos_tags.append(spacy_token.pos_)
        else:
            # Fallback: can't align (punctuation edge case)
            pos_tags.append('X')  # Unknown

        char_offset += len(clean_token)

    return pos_tags


def get_content_word_mask(
    tokens: List[str],
    content_pos: Optional[List[str]] = None
) -> List[bool]:
    """
    Create binary mask for content words.

    Args:
        tokens: List of GPT-2 tokens
        content_pos: POS tags considered "content" (default: NOUN, VERB, ADJ, ADV)

    Returns:
        Boolean mask (True = content word)
    """
    if content_pos is None:
        content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']

    pos_tags = align_tokens_to_pos(tokens, content_pos)
    return [pos in content_pos for pos in pos_tags]


def compute_pos_correlation(
    tokens: List[str],
    relevance_scores: List[float],
    content_pos: Optional[List[str]] = None
) -> Tuple[float, float]:
    """
    Compute Spearman correlation between relevance and content word mask.

    Args:
        tokens: List of tokens
        relevance_scores: Relevance scores per token
        content_pos: POS tags considered "content"

    Returns:
        Tuple of (correlation, p_value)
    """
    content_mask = get_content_word_mask(tokens, content_pos)

    # Convert to numeric
    content_numeric = np.array([1.0 if m else 0.0 for m in content_mask])
    relevance = np.array(relevance_scores)

    # Ensure same length
    min_len = min(len(content_numeric), len(relevance))
    content_numeric = content_numeric[:min_len]
    relevance = relevance[:min_len]

    if len(content_numeric) < 3:
        return 0.0, 1.0

    correlation, p_value = spearmanr(relevance, content_numeric)
    return correlation, p_value


def compute_pos_mass_distribution(
    tokens: List[str],
    relevance_scores: List[float]
) -> Dict[str, float]:
    """
    Compute how relevance mass is distributed across POS categories.

    Args:
        tokens: List of GPT-2 tokens
        relevance_scores: Relevance scores per token

    Returns:
        Dict mapping POS category to total relevance mass
    """
    # Get aligned POS tags for each token
    pos_tags = align_tokens_to_pos(tokens)

    # Normalize relevance to sum to 1
    relevance = np.array(relevance_scores)
    relevance = relevance / (relevance.sum() + 1e-10)

    # Collect mass by POS
    pos_mass: Dict[str, float] = {}

    min_len = min(len(pos_tags), len(relevance))
    for i in range(min_len):
        pos = pos_tags[i]
        if pos not in pos_mass:
            pos_mass[pos] = 0.0
        pos_mass[pos] += relevance[i]

    return pos_mass


def compute_special_token_mass(
    tokens: List[str],
    relevance_scores: List[float],
    special_tokens: Optional[List[str]] = None
) -> float:
    """
    Compute relevance mass on special/punctuation tokens.

    Args:
        tokens: List of GPT-2 tokens
        relevance_scores: Relevance scores
        special_tokens: List of special tokens (default: common ones)

    Returns:
        Total mass on special tokens (should be low for good relevance)
    """
    if special_tokens is None:
        special_tokens = [
            '<|endoftext|>', '<s>', '</s>', '<pad>', '<unk>',
            '.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']',
            '"', "'", '`', '...', '--',
            # GPT-2 versions with space prefix
            'Ġ.', 'Ġ,', 'Ġ!', 'Ġ?', 'Ġ;', 'Ġ:', 'Ġ-', 'Ġ(', 'Ġ)',
            'Ġ[', 'Ġ]', 'Ġ"', "Ġ'", 'Ġ`'
        ]

    # Normalize relevance
    relevance = np.array(relevance_scores)
    relevance = relevance / (relevance.sum() + 1e-10)

    special_mass = 0.0
    for i, token in enumerate(tokens):
        if i < len(relevance):
            # Check both raw token and cleaned version
            cleaned = clean_gpt2_token(token).strip()
            if token in special_tokens or cleaned in special_tokens:
                special_mass += relevance[i]

    return special_mass


def compute_stop_word_mass(
    tokens: List[str],
    relevance_scores: List[float]
) -> float:
    """
    Compute relevance mass on stop words.

    Uses NLTK stop word list.

    Args:
        tokens: List of GPT-2 tokens
        relevance_scores: Relevance scores per token

    Returns:
        Total mass on stop words
    """
    import nltk
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        stop_words = set(nltk.corpus.stopwords.words('english'))

    # Normalize relevance
    relevance = np.array(relevance_scores)
    relevance = relevance / (relevance.sum() + 1e-10)

    stop_mass = 0.0
    for i, token in enumerate(tokens):
        if i < len(relevance):
            # Clean GPT-2 token before checking
            cleaned = clean_gpt2_token(token).lower().strip()
            if cleaned in stop_words:
                stop_mass += relevance[i]

    return stop_mass


def analyze_relevance_distribution(
    tokens: List[str],
    relevance_scores: List[float]
) -> Dict[str, float]:
    """
    Comprehensive analysis of relevance distribution.

    Returns:
        Dict with various relevance distribution metrics
    """
    pos_mass = compute_pos_mass_distribution(tokens, relevance_scores)

    # Aggregate by category
    content_pos = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']
    function_pos = ['DET', 'ADP', 'CCONJ', 'SCONJ', 'AUX', 'PRON']

    content_mass = sum(pos_mass.get(p, 0) for p in content_pos)
    function_mass = sum(pos_mass.get(p, 0) for p in function_pos)
    punct_mass = pos_mass.get('PUNCT', 0)

    special_mass = compute_special_token_mass(tokens, relevance_scores)
    stop_mass = compute_stop_word_mass(tokens, relevance_scores)

    correlation, p_value = compute_pos_correlation(tokens, relevance_scores)

    return {
        'content_word_mass': content_mass,
        'function_word_mass': function_mass,
        'punctuation_mass': punct_mass,
        'special_token_mass': special_mass,
        'stop_word_mass': stop_mass,
        'pos_correlation': correlation,
        'pos_correlation_pvalue': p_value,
        'pos_breakdown': pos_mass
    }
