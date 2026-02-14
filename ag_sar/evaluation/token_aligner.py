"""
Token alignment utilities for matching character spans to token positions.

Critical for RAGTruth evaluation where labels are at character level
but signals are computed at token level.
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TokenSpan:
    """A token with its character span."""
    token_id: int
    token_text: str
    char_start: int  # Character start offset
    char_end: int  # Character end offset


@dataclass
class AlignedLabel:
    """A label aligned to tokens."""
    token_start: int  # Token start index
    token_end: int  # Token end index (exclusive)
    label: int  # 0 or 1
    char_start: int  # Original character start
    char_end: int  # Original character end


class TokenAligner:
    """
    Align character-level spans to token positions.

    Uses the tokenizer's offset mapping when available,
    or falls back to heuristic matching.
    """

    def __init__(self, tokenizer):
        """
        Initialize token aligner.

        Args:
            tokenizer: HuggingFace tokenizer
        """
        self.tokenizer = tokenizer

    def tokenize_with_offsets(self, text: str) -> List[TokenSpan]:
        """
        Tokenize text and get character offsets for each token.

        Args:
            text: Input text

        Returns:
            List of TokenSpan with character offsets
        """
        # Use tokenizer with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )

        token_spans = []
        for token_id, (start, end) in zip(
            encoding["input_ids"],
            encoding["offset_mapping"],
        ):
            token_text = self.tokenizer.decode([token_id])
            token_spans.append(TokenSpan(
                token_id=token_id,
                token_text=token_text,
                char_start=start,
                char_end=end,
            ))

        return token_spans

    def align_spans_to_tokens(
        self,
        char_spans: List[Tuple[int, int, int]],
        token_spans: List[TokenSpan],
        strategy: str = "any_overlap",
    ) -> List[AlignedLabel]:
        """
        Align character-level spans to token positions.

        Args:
            char_spans: List of (char_start, char_end, label) tuples
            token_spans: List of TokenSpan from tokenization
            strategy: Alignment strategy:
                - "any_overlap": Token is labeled if any character overlaps
                - "majority_overlap": Token is labeled if >50% overlaps
                - "full_overlap": Token is labeled only if fully contained

        Returns:
            List of AlignedLabel (at token level)
        """
        aligned = []

        for char_start, char_end, label in char_spans:
            # Find tokens that overlap with this span
            token_start = None
            token_end = None

            for i, ts in enumerate(token_spans):
                overlaps = self._check_overlap(
                    (ts.char_start, ts.char_end),
                    (char_start, char_end),
                    strategy,
                )

                if overlaps:
                    if token_start is None:
                        token_start = i
                    token_end = i + 1

            if token_start is not None:
                aligned.append(AlignedLabel(
                    token_start=token_start,
                    token_end=token_end,
                    label=label,
                    char_start=char_start,
                    char_end=char_end,
                ))

        return aligned

    def _check_overlap(
        self,
        token_range: Tuple[int, int],
        span_range: Tuple[int, int],
        strategy: str,
    ) -> bool:
        """
        Check if token overlaps with span according to strategy.

        Args:
            token_range: (start, end) of token
            span_range: (start, end) of span
            strategy: Overlap strategy

        Returns:
            True if token should be labeled
        """
        t_start, t_end = token_range
        s_start, s_end = span_range

        # Calculate overlap
        overlap_start = max(t_start, s_start)
        overlap_end = min(t_end, s_end)
        overlap = max(0, overlap_end - overlap_start)

        token_len = t_end - t_start

        if strategy == "any_overlap":
            return overlap > 0
        elif strategy == "majority_overlap":
            return overlap > token_len / 2
        elif strategy == "full_overlap":
            return overlap >= token_len
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_token_labels(
        self,
        aligned_labels: List[AlignedLabel],
        num_tokens: int,
    ) -> List[int]:
        """
        Convert aligned labels to per-token label array.

        Args:
            aligned_labels: List of AlignedLabel
            num_tokens: Total number of tokens

        Returns:
            List of labels (0 or 1) for each token
        """
        labels = [0] * num_tokens

        for al in aligned_labels:
            for i in range(al.token_start, min(al.token_end, num_tokens)):
                if al.label == 1:
                    labels[i] = 1

        return labels

    def align_response(
        self,
        response: str,
        char_spans: List[Tuple[int, int, int]],
    ) -> Tuple[List[TokenSpan], List[int]]:
        """
        Tokenize response and get aligned token-level labels.

        Args:
            response: Response text
            char_spans: List of (char_start, char_end, label) tuples

        Returns:
            Tuple of (token_spans, token_labels)
        """
        token_spans = self.tokenize_with_offsets(response)
        aligned = self.align_spans_to_tokens(char_spans, token_spans)
        token_labels = self.get_token_labels(aligned, len(token_spans))

        return token_spans, token_labels

    def compute_alignment_statistics(
        self,
        char_spans: List[Tuple[int, int, int]],
        aligned_labels: List[AlignedLabel],
        token_spans: List[TokenSpan],
    ) -> Dict[str, float]:
        """
        Compute statistics about alignment quality.

        Args:
            char_spans: Original character spans
            aligned_labels: Aligned token spans
            token_spans: All tokens

        Returns:
            Dict with alignment statistics
        """
        if not char_spans:
            return {
                "alignment_rate": 1.0,
                "token_coverage": 0.0,
                "avg_expansion_ratio": 1.0,
            }

        # Alignment rate: fraction of char spans that aligned to tokens
        aligned_count = len(aligned_labels)
        alignment_rate = aligned_count / len(char_spans)

        # Token coverage: fraction of tokens that are labeled
        labeled_tokens = set()
        for al in aligned_labels:
            for i in range(al.token_start, al.token_end):
                labeled_tokens.add(i)
        token_coverage = len(labeled_tokens) / len(token_spans) if token_spans else 0

        # Expansion ratio: ratio of token span length to char span length
        expansion_ratios = []
        for cs, al in zip(char_spans, aligned_labels):
            char_len = cs[1] - cs[0]
            token_len = al.token_end - al.token_start
            if char_len > 0:
                expansion_ratios.append(token_len / char_len)

        avg_expansion = (
            sum(expansion_ratios) / len(expansion_ratios)
            if expansion_ratios else 1.0
        )

        return {
            "alignment_rate": alignment_rate,
            "token_coverage": token_coverage,
            "avg_expansion_ratio": avg_expansion,
        }
