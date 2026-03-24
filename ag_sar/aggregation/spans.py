from dataclasses import dataclass
import numpy as np

from ..numerics import otsu_threshold


@dataclass
class RiskySpan:
    start: int
    end: int
    token_risks: list[float]
    max_risk: float
    mean_risk: float

    @property
    def length(self) -> int:
        return self.end - self.start


class SpanMerger:

    def __init__(self, threshold: float, max_gap: int = 1):
        self.threshold = threshold
        self.max_gap = max_gap

    @classmethod
    def adaptive(cls, token_risks: list[float]) -> "SpanMerger":
        risks = np.asarray(token_risks)
        threshold = otsu_threshold(risks)
        n_above = int(np.sum(risks >= threshold))
        max_gap = len(risks) // n_above
        return cls(threshold=threshold, max_gap=max_gap)

    def find_spans(self, token_risks: list[float]) -> list[RiskySpan]:
        risks = np.asarray(token_risks)
        high_risk_indices = np.nonzero(risks >= self.threshold)[0]

        if len(high_risk_indices) == 0:
            return []

        gaps = np.diff(high_risk_indices)
        split_points = np.nonzero(gaps > self.max_gap)[0] + 1
        groups = np.split(high_risk_indices, split_points)

        spans = []
        for g in groups:
            start, end = int(g[0]), int(g[-1]) + 1
            span_risks = risks[start:end]
            spans.append(RiskySpan(
                start=start,
                end=end,
                token_risks=span_risks.tolist(),
                max_risk=float(span_risks.max()),
                mean_risk=float(span_risks.mean()),
            ))
        return spans
