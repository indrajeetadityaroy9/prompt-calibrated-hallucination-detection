"""
Span merger for grouping high-risk tokens into contiguous spans.

Identifies risky spans (potential hallucinations) from token-level risks.
"""

from typing import List, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ..numerics import EPS


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
        min_span_risk: float = 0.0,
    ):
        """
        Initialize span merger.

        Args:
            threshold: Minimum risk to include token in span (REQUIRED - no default)
            min_span_length: Minimum number of tokens in a span
            max_gap: Maximum gap between high-risk tokens to merge
            min_span_risk: Minimum mean risk to keep span (0.0 = use threshold)
        """
        self.threshold = threshold
        self.min_span_length = min_span_length
        self.max_gap = max_gap
        self.min_span_risk = min_span_risk if min_span_risk > 0 else threshold

    @classmethod
    def adaptive(cls, token_risks: List[float]) -> "SpanMerger":
        """
        Create a SpanMerger with data-driven parameters (no hardcoded floors).

        Threshold: Q3 + 0.5 × IQR (mild Tukey fence).
        Identifies tokens in the upper tail of the risk distribution.
        Falls back to Q3 when IQR < 0.01 (near-uniform risks).

        Reference: Tukey (1977) "Exploratory Data Analysis"

        Args:
            token_risks: Per-token risk scores

        Returns:
            SpanMerger with adaptive parameters
        """
        n = len(token_risks)
        if n == 0:
            return cls(threshold=0.5, min_span_length=1, max_gap=1)

        risks = np.array(token_risks)
        q1 = float(np.percentile(risks, 25))
        q3 = float(np.percentile(risks, 75))
        iqr = q3 - q1

        if iqr < EPS:
            # Near-uniform risks: use Q3 directly
            threshold = q3
        else:
            # Mild Tukey fence (0.5×IQR, not standard 1.5×IQR):
            # Standard 1.5× is designed for Gaussian outlier detection.
            # We use 0.5× for higher sensitivity — hallucination spans
            # are moderate elevations, not extreme outliers.
            threshold = q3 + 0.5 * iqr

        return cls(
            threshold=threshold,
            min_span_length=max(1, n // 100),
            max_gap=max(1, n // 50),
        )

    def find_spans(
        self,
        token_risks: List[float],
        token_texts: List[str] = None,
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
