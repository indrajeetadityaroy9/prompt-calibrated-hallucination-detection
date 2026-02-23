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
    """Merge high-risk tokens into contiguous spans via bimodality-adaptive thresholding."""

    def __init__(
        self,
        threshold: float,
        min_span_length: int = 1,
        max_gap: int = 1,
        min_span_risk: float = 0.0,
    ):
        self.threshold = threshold
        self.min_span_length = min_span_length
        self.max_gap = max_gap
        self.min_span_risk = min_span_risk if min_span_risk > 0 else threshold

    @classmethod
    def adaptive(cls, token_risks: List[float]) -> "SpanMerger":
        """Bimodality-adaptive percentile threshold. Otsu (1979).

        Interpolates between Otsu split (bimodal) and 95th percentile (unimodal)
        based on bimodality coefficient. Distribution-free: no IQR or Tukey fence.

        B=0 (unimodal): threshold = P95 (conservative, few tokens flagged)
        B=1 (bimodal):  threshold = Otsu split (aggressive, flags the high mode)
        """
        from ..numerics import otsu_threshold as _otsu

        n = len(token_risks)
        if n == 0:
            return cls(threshold=0.5, min_span_length=1, max_gap=1)

        risks = np.array(token_risks)

        # Bimodality coefficient via Otsu
        total_var = float(np.var(risks))
        bimodality = 0.0
        threshold_otsu = float(np.median(risks))
        if total_var > EPS:
            threshold_otsu = _otsu(risks)
            mask = risks >= threshold_otsu
            if mask.any() and (~mask).any():
                w0 = float((~mask).mean())
                w1 = float(mask.mean())
                mu0 = float(risks[~mask].mean())
                mu1 = float(risks[mask].mean())
                bimodality = w0 * w1 * (mu0 - mu1) ** 2 / total_var

        # Adaptive percentile: bimodality controls threshold aggressiveness
        p95 = float(np.percentile(risks, 95))
        if bimodality > EPS and total_var > EPS:
            threshold = bimodality * threshold_otsu + (1.0 - bimodality) * p95
        else:
            threshold = p95

        # Expected-gap-based max_gap
        n_above = int(np.sum(risks >= threshold))
        if n_above > 1:
            expected_gap = max(1, n // n_above)
            max_gap_val = max(1, expected_gap // 2)
        else:
            max_gap_val = 1

        return cls(
            threshold=threshold,
            min_span_length=max(1, n // 100),
            max_gap=max_gap_val,
        )

    def find_spans(
        self,
        token_risks: List[float],
        token_texts: List[str] = None,
    ) -> List[RiskySpan]:
        """Find contiguous spans above threshold, merge with gap tolerance, filter by length/risk."""
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
