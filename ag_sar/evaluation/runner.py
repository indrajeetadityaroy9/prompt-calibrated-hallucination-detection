"""
Evaluation runner for RAGTruth and HaluEval benchmarks.

Orchestrates evaluation, collects results, and computes metrics.
"""

from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from tqdm import tqdm

import numpy as np

from .modes import EvaluationMode, ForcedDecodingEvaluator, GenerationEvaluator
from .metrics import compute_metrics, compute_span_metrics, MetricsResult
from ..config import DSGConfig, DSGTokenSignals
from ..aggregation.prompt_anchored import PromptAnchoredAggregator


@dataclass
class EvaluationExample:
    """Single evaluation example."""
    id: str
    context: str
    question: str
    response: str
    labels: List[Tuple[int, int, int]]  # List of (start, end, label) tuples
    response_label: int = 0  # Response-level label for HaluEval


@dataclass
class EvaluationResult:
    """Result for a single example."""
    example_id: str
    token_signals: List[DSGTokenSignals]
    token_risks: List[float]
    predicted_spans: List[Tuple[int, int]]
    response_risk: float
    response_predicted: int  # Binary prediction
    ground_truth_label: int = 0


@dataclass
class EvaluationSummary:
    """Summary of evaluation run."""
    mode: EvaluationMode
    num_examples: int
    token_metrics: MetricsResult
    response_metrics: MetricsResult
    span_metrics: Dict[str, float] = None

    # Per-example results (optional)
    example_results: List[EvaluationResult] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print formatted summary of key metrics including selective prediction."""
        r = self.response_metrics
        t = self.token_metrics
        print(f"\n{'='*60}")
        print(f"  DSG Evaluation Summary ({self.mode.value}, n={self.num_examples})")
        print(f"{'='*60}")
        print(f"  Response-level:")
        print(f"    AUROC:       {r.auroc:.4f}")
        print(f"    AUPRC:       {r.auprc:.4f}")
        print(f"    TPR@5%FPR:   {r.tpr_at_5_fpr:.4f}")
        print(f"    F1:          {r.f1:.4f}")
        print(f"  Token-level:")
        print(f"    AUROC:       {t.auroc:.4f}")
        print(f"    AUPRC:       {t.auprc:.4f}")
        print(f"  Selective Prediction:")
        print(f"    AURC:        {r.aurc:.4f}")
        print(f"    E-AURC:      {r.e_aurc:.4f}")
        print(f"    Risk@90:     {r.risk_at_90_coverage:.4f}")
        print(f"  Calibration:")
        print(f"    ECE:         {r.expected_calibration_error:.4f}")
        print(f"    Brier:       {r.brier_score:.4f}")
        if self.span_metrics:
            print(f"  Span-level:")
            print(f"    F1:          {self.span_metrics.get('span_f1', 0):.4f}")
        print(f"{'='*60}\n")


class EvaluationRunner:
    """
    Run evaluation on RAGTruth or HaluEval datasets.

    Supports both forced decoding (for span-level labels) and
    generation (for response-level labels) modes.
    """

    def __init__(
        self,
        detector,
        mode: EvaluationMode = EvaluationMode.FORCED_DECODING,
        risk_threshold: float = 0.5,
    ):
        """
        Initialize evaluation runner.

        Args:
            detector: DSGDetector instance
            mode: Evaluation mode
            risk_threshold: Threshold for span/response flagging
        """
        self.detector = detector
        self.mode = mode
        self.risk_threshold = risk_threshold

        # Initialize evaluators based on mode
        if mode == EvaluationMode.FORCED_DECODING:
            self.evaluator = ForcedDecodingEvaluator(
                model=detector.model,
                tokenizer=detector.tokenizer,
                config=detector.config,
            )
        else:
            self.evaluator = GenerationEvaluator(detector=detector)

    def run_ragtruth(
        self,
        examples: List[EvaluationExample],
        compute_span_metrics: bool = True,
    ) -> EvaluationSummary:
        """
        Run evaluation on RAGTruth dataset.

        RAGTruth requires FORCED_DECODING mode because span labels
        apply to the provided response.

        Args:
            examples: List of EvaluationExample
            compute_span_metrics: Whether to compute span-level metrics

        Returns:
            EvaluationSummary with all metrics
        """
        if self.mode != EvaluationMode.FORCED_DECODING:
            raise ValueError(
                "RAGTruth requires FORCED_DECODING mode because "
                "span labels apply to the provided response."
            )

        all_token_scores = []
        all_token_labels = []
        all_response_scores = []
        all_response_labels = []
        all_predicted_spans = []
        all_gold_spans = []
        example_results = []

        for example in tqdm(examples, desc="Evaluating RAGTruth"):
            # Run forced decoding
            token_signals = self.evaluator.evaluate(
                context=example.context,
                question=example.question,
                response=example.response,
            )

            # Compute token risks using Noisy-OR over DSG signals
            token_risks = self._compute_token_risks(token_signals)

            # Collect token-level scores and labels
            token_labels = self._align_labels_to_tokens(
                example.labels, len(token_signals)
            )
            all_token_scores.extend(token_risks)
            all_token_labels.extend(token_labels)

            # Collect response-level scores and labels
            response_risk = max(token_risks) if token_risks else 0.0
            response_label = 1 if any(l == 1 for l in token_labels) else 0
            all_response_scores.append(response_risk)
            all_response_labels.append(response_label)

            # Find predicted spans
            predicted_spans = self._find_spans(token_risks)
            all_predicted_spans.extend(predicted_spans)

            # Get gold spans
            gold_spans = [(s, e) for s, e, l in example.labels if l == 1]
            all_gold_spans.extend(gold_spans)

            # Store result
            example_results.append(EvaluationResult(
                example_id=example.id,
                token_signals=token_signals,
                token_risks=token_risks,
                predicted_spans=predicted_spans,
                response_risk=response_risk,
                response_predicted=1 if response_risk >= self.risk_threshold else 0,
                ground_truth_label=response_label,
            ))

        # Compute metrics
        token_metrics = compute_metrics(all_token_scores, all_token_labels)
        response_metrics = compute_metrics(all_response_scores, all_response_labels)

        span_metrics_result = None
        if compute_span_metrics and all_gold_spans:
            span_metrics_result = compute_span_metrics(
                all_predicted_spans, all_gold_spans
            )

        return EvaluationSummary(
            mode=self.mode,
            num_examples=len(examples),
            token_metrics=token_metrics,
            response_metrics=response_metrics,
            span_metrics=span_metrics_result,
            example_results=example_results,
        )

    def run_halueval(
        self,
        examples: List[EvaluationExample],
    ) -> EvaluationSummary:
        """
        Run evaluation on HaluEval dataset.

        HaluEval has response-level binary labels.
        Can use either mode, but GENERATION is typical.

        Args:
            examples: List of EvaluationExample

        Returns:
            EvaluationSummary with response-level metrics
        """
        all_response_scores = []
        all_response_labels = []
        example_results = []

        for example in tqdm(examples, desc="Evaluating HaluEval"):
            if self.mode == EvaluationMode.GENERATION:
                # Generate and evaluate
                result = self.evaluator.evaluate(
                    context=example.context,
                    question=example.question,
                )
                token_signals = result.token_signals
                token_risks = result.token_risks
                response_risk = result.response_risk
            else:
                # Forced decoding on provided response
                token_signals = self.evaluator.evaluate(
                    context=example.context,
                    question=example.question,
                    response=example.response,
                )
                token_risks = self._compute_token_risks(token_signals)
                response_risk = max(token_risks) if token_risks else 0.0

            all_response_scores.append(response_risk)
            all_response_labels.append(example.response_label)

            # Store result
            example_results.append(EvaluationResult(
                example_id=example.id,
                token_signals=token_signals,
                token_risks=token_risks,
                predicted_spans=[],
                response_risk=response_risk,
                response_predicted=1 if response_risk >= self.risk_threshold else 0,
                ground_truth_label=example.response_label,
            ))

        # Compute metrics
        response_metrics = compute_metrics(all_response_scores, all_response_labels)

        # Token metrics are not meaningful for HaluEval (no token labels)
        token_metrics = MetricsResult(
            auroc=0.0, auprc=0.0, precision=0.0, recall=0.0,
            f1=0.0, accuracy=0.0, threshold=0.5,
            true_positives=0, false_positives=0,
            true_negatives=0, false_negatives=0,
        )

        return EvaluationSummary(
            mode=self.mode,
            num_examples=len(examples),
            token_metrics=token_metrics,
            response_metrics=response_metrics,
            example_results=example_results,
        )

    def _compute_token_risks(
        self,
        token_signals: List[DSGTokenSignals],
    ) -> List[float]:
        """
        Compute token risks from DSG signals using prompt-anchored z-score + Noisy-OR.

        Uses PromptAnchoredAggregator with prompt_stats from the detector
        for proper calibration (prevents Relativity Trap).
        """
        if not token_signals:
            return []

        response_signals = {
            "cus": np.array([s.cus for s in token_signals]),
            "pos": np.array([s.pos for s in token_signals]),
            "dps": np.array([s.dps for s in token_signals]),
        }

        # Prefer evaluator's prompt stats (fresh from this input's prefill),
        # fall back to detector's stats (from a previous detect() call).
        prompt_stats = (
            getattr(self.evaluator, '_prompt_stats', None)
            or getattr(self.detector, '_prompt_stats', None)
            or {}
        )
        aggregator = PromptAnchoredAggregator(active_signals={"cus", "pos", "dps"})
        result = aggregator.compute_risk(prompt_stats, response_signals)
        return result.token_risks.tolist()

    def _align_labels_to_tokens(
        self,
        labels: List[Tuple[int, int, int]],
        num_tokens: int,
    ) -> List[int]:
        """
        Align span labels to token positions.

        Args:
            labels: List of (start, end, label) tuples (character offsets)
            num_tokens: Number of tokens

        Returns:
            List of per-token labels
        """
        # For now, simple assumption that labels are token indices
        # In practice, need character-to-token alignment
        token_labels = [0] * num_tokens

        for start, end, label in labels:
            for i in range(start, min(end, num_tokens)):
                if label == 1:
                    token_labels[i] = 1

        return token_labels

    def _find_spans(
        self,
        token_risks: List[float],
    ) -> List[Tuple[int, int]]:
        """Find contiguous spans of high-risk tokens."""
        spans = []
        in_span = False
        span_start = 0

        for i, risk in enumerate(token_risks):
            if risk >= self.risk_threshold:
                if not in_span:
                    span_start = i
                    in_span = True
            else:
                if in_span:
                    spans.append((span_start, i))
                    in_span = False

        if in_span:
            spans.append((span_start, len(token_risks)))

        return spans
