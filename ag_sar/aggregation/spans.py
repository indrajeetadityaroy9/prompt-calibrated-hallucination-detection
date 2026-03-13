"""
Span merger for grouping high-risk tokens into contiguous spans.

Identifies risky spans (potential hallucinations) from token-level risks.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..numerics import otsu_threshold, EPS


@dataclass
class RiskySpan:
    """A contiguous span of high-risk tokens."""
    start: int
    end: int
    token_risks: list[float]
    max_risk: float
    mean_risk: float

    @property
    def length(self) -> int:
        return self.end - self.start


class SpanMerger:
    """Merge high-risk tokens into contiguous spans via Otsu thresholding."""

    def __init__(self, threshold: float, max_gap: int = 1):
        self.threshold = threshold
        self.max_gap = max_gap

    @classmethod
    def adaptive(cls, token_risks: list[float]) -> SpanMerger:
        """Otsu-based adaptive threshold with expected-gap merging."""
        n = len(token_risks)
        risks = np.array(token_risks)

        total_var = float(np.var(risks))
        if total_var <= EPS:
            threshold = float(np.median(risks))
        else:
            threshold = otsu_threshold(risks)

        n_above = int(np.sum(risks >= threshold))
        max_gap = max(1, n // max(1, n_above))

        return cls(threshold=threshold, max_gap=max_gap)

    def find_spans(self, token_risks: list[float]) -> list[RiskySpan]:
        """Find contiguous spans above threshold, merge with gap tolerance."""
        high_risk_indices = [
            i for i, r in enumerate(token_risks) if r >= self.threshold
        ]

        if not high_risk_indices:
            return []

        spans = []
        current_start = high_risk_indices[0]
        current_end = high_risk_indices[0] + 1

        for i in range(1, len(high_risk_indices)):
            idx = high_risk_indices[i]
            if idx <= current_end + self.max_gap:
                current_end = idx + 1
            else:
                spans.append((current_start, current_end))
                current_start = idx
                current_end = idx + 1

        spans.append((current_start, current_end))

        return [
            RiskySpan(
                start=start,
                end=end,
                token_risks=token_risks[start:end],
                max_risk=max(token_risks[start:end]),
                mean_risk=sum(token_risks[start:end]) / (end - start),
            )
            for start, end in spans
        ]
