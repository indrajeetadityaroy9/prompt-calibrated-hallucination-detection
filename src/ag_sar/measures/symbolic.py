"""
Symbolic Entity Overlap Measure.

Solves the "Paris vs London" problem that neural embeddings miss.
When the model generates a Proper Noun that does NOT exist in the
source context, this is a strong signal of hallucination.

Key Insight: Neural methods detect CATEGORICAL violations (fruit vs city).
Symbolic methods detect VALUE violations (wrong city vs right city).

Usage:
    from ag_sar.measures.symbolic import compute_context_overlap

    context = "The capital of France is Paris."
    response = "London"

    overlap = compute_context_overlap(response, context)
    # overlap = 0.0 (London not in context -> violation)
"""

import re
import string
from typing import Set, Tuple, List


# Common words to exclude from entity detection
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "now",
    "and", "but", "or", "if", "because", "until", "while", "although",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "yes", "no", "answer", "question", "according",
    "based", "given", "following", "above", "below", "document", "documents",
    "retrieved", "context", "source", "information", "data", "report",
}


def _clean_token(token: str) -> str:
    """Remove punctuation and lowercase a token."""
    return token.translate(str.maketrans('', '', string.punctuation)).lower()


def _extract_entities(text: str) -> List[str]:
    """
    Extract potential named entities from text.

    Uses simple heuristics:
    1. Capitalized words (proper nouns)
    2. Words with numbers (dates, amounts)
    3. All-caps words (acronyms)

    Returns list of entity strings (lowercase for matching).
    """
    entities = []

    # Split into sentences to handle sentence-initial capitalization
    sentences = re.split(r'[.!?]+', text)

    for sentence in sentences:
        words = sentence.split()

        # Special case: very short responses (1-3 words) - treat ALL capitalized words as entities
        is_short_response = len(words) <= 3

        for i, word in enumerate(words):
            # Skip empty words
            if not word:
                continue

            # Strip leading/trailing punctuation for checking
            word_stripped = word.strip(string.punctuation)
            if not word_stripped:
                continue

            # Clean version for checking
            clean = _clean_token(word)

            # Skip stopwords
            if clean in STOPWORDS:
                continue

            # Skip very short tokens
            if len(clean) < 2:
                continue

            # Check if it's a potential entity
            is_entity = False

            # For short responses, ANY capitalized word is treated as entity
            if is_short_response and word_stripped[0].isupper():
                is_entity = True

            # Capitalized word not at sentence start (for longer text)
            elif i > 0 and word_stripped[0].isupper():
                is_entity = True

            # First word that looks like a proper noun (not common words)
            elif i == 0 and word_stripped[0].isupper() and clean not in STOPWORDS:
                # Check if it's likely a proper noun (not a common starter)
                common_starters = {"the", "a", "an", "this", "that", "it", "i", "we", "they", "yes", "no"}
                if clean not in common_starters:
                    is_entity = True

            # Contains numbers (dates, amounts, IDs)
            if any(c.isdigit() for c in word):
                is_entity = True

            # All caps (acronyms)
            if word_stripped.isupper() and len(word_stripped) > 1:
                is_entity = True

            if is_entity:
                # Store the cleaned version for matching
                entities.append(clean)

    return entities


def _extract_context_tokens(context: str) -> Set[str]:
    """Extract all meaningful tokens from context for matching."""
    tokens = set()

    for word in context.split():
        clean = _clean_token(word)
        if clean and len(clean) >= 2:
            tokens.add(clean)

    return tokens


def compute_context_overlap(
    response_text: str,
    context_text: str,
    strict: bool = False,
) -> Tuple[float, dict]:
    """
    Check if named entities in the response exist in the context.

    This is the key symbolic check that neural methods miss.
    If the response mentions "London" but the context only mentions "Paris",
    this is a strong hallucination signal regardless of embedding similarity.

    Args:
        response_text: The generated response to check
        context_text: The source context (prompt/retrieved docs)
        strict: If True, any violation = 0.0. If False, use gradual penalty.

    Returns:
        Tuple of (overlap_score, details_dict)
        - overlap_score: 1.0 = all entities found, 0.0 = violations found
        - details_dict: Contains entity lists and violation info

    Example:
        >>> context = "The capital of France is Paris. It has the Eiffel Tower."
        >>> response = "The capital is London."
        >>> score, details = compute_context_overlap(response, context)
        >>> score
        0.0
        >>> details['violations']
        ['london']
    """
    # Extract entities from response
    response_entities = _extract_entities(response_text)

    # Extract all tokens from context
    context_tokens = _extract_context_tokens(context_text)

    # Track violations
    violations = []
    found = []

    for entity in response_entities:
        if entity in context_tokens:
            found.append(entity)
        else:
            # Check for partial matches (e.g., "TechCorp" vs "techcorp's")
            partial_match = any(entity in ctx_tok or ctx_tok in entity
                              for ctx_tok in context_tokens if len(ctx_tok) > 3)
            if partial_match:
                found.append(entity)
            else:
                violations.append(entity)

    # Calculate score
    if not response_entities:
        # No entities to check - neutral score
        score = 1.0
    elif strict:
        # Any violation = fail
        score = 0.0 if violations else 1.0
    else:
        # Gradual penalty
        # 0 violations -> 1.0
        # 1 violation -> 0.5
        # 2+ violations -> 0.25
        # 3+ violations -> 0.0
        if len(violations) == 0:
            score = 1.0
        elif len(violations) == 1:
            score = 0.5
        elif len(violations) == 2:
            score = 0.25
        else:
            score = 0.0

    details = {
        "response_entities": response_entities,
        "context_tokens_count": len(context_tokens),
        "found": found,
        "violations": violations,
        "num_violations": len(violations),
    }

    return score, details


def compute_numeric_consistency(
    response_text: str,
    context_text: str,
) -> Tuple[float, dict]:
    """
    Check if numbers in the response match numbers in the context.

    Critical for RAG tasks involving financial data, dates, statistics.

    Args:
        response_text: The generated response
        context_text: The source context

    Returns:
        Tuple of (consistency_score, details_dict)
    """
    # Extract numbers from both texts
    number_pattern = r'\$?[\d,]+\.?\d*%?|\d+(?:st|nd|rd|th)?'

    response_numbers = set(re.findall(number_pattern, response_text))
    context_numbers = set(re.findall(number_pattern, context_text))

    # Normalize numbers for comparison
    def normalize(num_str):
        # Remove $, commas, % for comparison
        return num_str.replace('$', '').replace(',', '').replace('%', '').lower()

    response_normalized = {normalize(n) for n in response_numbers}
    context_normalized = {normalize(n) for n in context_numbers}

    # Find numbers in response that aren't in context
    violations = []
    for num in response_numbers:
        if normalize(num) not in context_normalized:
            violations.append(num)

    # Calculate score
    if not response_numbers:
        score = 1.0  # No numbers to check
    elif not violations:
        score = 1.0  # All numbers found
    else:
        # Penalty based on violation ratio
        violation_ratio = len(violations) / len(response_numbers)
        score = max(0.0, 1.0 - violation_ratio)

    details = {
        "response_numbers": list(response_numbers),
        "context_numbers": list(context_numbers),
        "violations": violations,
    }

    return score, details
