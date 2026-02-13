"""
Span merger for grouping high-risk tokens into contiguous spans.

Identifies risky spans (potential hallucinations) from token-level risks.
"""

from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class RiskySpan:
    """A contiguous span of high-risk tokens."""
    start: int  # Start token index (inclusive)
    end: int  # End token index (exclusive)
    token_risks: List[float]  # Risk scores for tokens in span
    max_risk: float  # Maximum risk in span
    mean_risk: float  # Mean risk in span
    text: str = ""  # Decoded text (optional)

    @property
    def length(self) -> int:
        """Number of tokens in span."""
        return self.end - self.start


class SpanMerger:
    """
    Merge high-risk tokens into contiguous spans.

    Algorithm:
    1. Identify tokens above threshold
    2. Merge adjacent high-risk tokens
    3. Optionally extend spans to include nearby tokens
    4. Filter by minimum span length or risk
    """

    def __init__(
        self,
        threshold: float,
        min_span_length: int = 1,
        max_gap: int = 1,
        min_span_risk: Optional[float] = None,
    ):
        """
        Initialize span merger.

        Args:
            threshold: Minimum risk to include token in span (REQUIRED - no default)
            min_span_length: Minimum number of tokens in a span
            max_gap: Maximum gap between high-risk tokens to merge
            min_span_risk: Minimum mean risk to keep span (defaults to threshold)
        """
        self.threshold = threshold
        self.min_span_length = min_span_length
        self.max_gap = max_gap
        self.min_span_risk = min_span_risk if min_span_risk is not None else threshold

    def find_spans(
        self,
        token_risks: List[float],
        token_texts: Optional[List[str]] = None,
    ) -> List[RiskySpan]:
        """
        Find contiguous spans of high-risk tokens.

        Args:
            token_risks: List of per-token risk scores
            token_texts: Optional list of token texts for span decoding

        Returns:
            List of RiskySpan objects
        """
        if not token_risks:
            return []

        # Find indices above threshold
        high_risk_indices = [
            i for i, r in enumerate(token_risks) if r >= self.threshold
        ]

        if not high_risk_indices:
            return []

        # Merge adjacent indices (with gap tolerance)
        spans = []
        current_start = high_risk_indices[0]
        current_end = high_risk_indices[0] + 1

        for i in range(1, len(high_risk_indices)):
            idx = high_risk_indices[i]
            if idx <= current_end + self.max_gap:
                # Extend current span
                current_end = idx + 1
            else:
                # Save current span and start new one
                spans.append((current_start, current_end))
                current_start = idx
                current_end = idx + 1

        # Don't forget the last span
        spans.append((current_start, current_end))

        # Convert to RiskySpan objects and filter
        result = []
        for start, end in spans:
            span_risks = token_risks[start:end]
            mean_risk = sum(span_risks) / len(span_risks)
            max_risk = max(span_risks)

            # Apply filters
            if end - start < self.min_span_length:
                continue
            if mean_risk < self.min_span_risk:
                continue

            # Get text if available
            text = ""
            if token_texts is not None:
                text = "".join(token_texts[start:end])

            result.append(RiskySpan(
                start=start,
                end=end,
                token_risks=span_risks,
                max_risk=max_risk,
                mean_risk=mean_risk,
                text=text,
            ))

        return result
