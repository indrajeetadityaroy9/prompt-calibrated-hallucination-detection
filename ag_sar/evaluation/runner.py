"""
Evaluation runner for RAGTruth and HaluEval benchmarks.

Orchestrates evaluation, collects results, and computes metrics.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from tqdm import tqdm

from .modes import EvaluationMode, ForcedDecodingEvaluator, GenerationEvaluator
from .metrics import compute_metrics, compute_span_metrics, MetricsResult
from ..config import TokenSignals, DetectorConfig



@dataclass
class EvaluationExample:
    """Single evaluation example."""
    id: str
    context: str
    question: str
    response: str
    labels: List[Tuple[int, int, int]]  # List of (start, end, label) tuples
    response_label: Optional[int] = None  # Response-level label for HaluEval


@dataclass
class EvaluationResult:
    """Result for a single example."""
    example_id: str
    token_signals: List[TokenSignals]
    token_risks: List[float]
    predicted_spans: List[Tuple[int, int]]
    response_risk: float
    response_predicted: int  # Binary prediction
    ground_truth_label: Optional[int] = None


@dataclass
class EvaluationSummary:
    """Summary of evaluation run."""
    mode: EvaluationMode
    num_examples: int
    token_metrics: MetricsResult
    response_metrics: MetricsResult
    span_metrics: Optional[Dict[str, float]] = None

    # Aggregated statistics
    mean_topk_mass: float = 0.0
    min_topk_mass: float = 1.0
    topk_mass_warning_count: int = 0

    # Per-example results (optional)
    example_results: List[EvaluationResult] = field(default_factory=list)


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
            detector: HallucinationDetector instance
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
                signal_computer=detector.jsd_signal,
                candidate_manager=detector.candidate_manager,
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

        topk_masses = []

        for example in tqdm(examples, desc="Evaluating RAGTruth"):
            # Run forced decoding
            token_signals = self.evaluator.evaluate(
                context=example.context,
                question=example.question,
                response=example.response,
            )

            # Compute token risks using Noisy-OR or simple average
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

            # Collect topk_mass
            for ts in token_signals:
                if ts.topk_mass is not None:
                    topk_masses.append(ts.topk_mass)

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

        # Compute topk_mass statistics
        mean_topk_mass = sum(topk_masses) / len(topk_masses) if topk_masses else 0.0
        min_topk_mass = min(topk_masses) if topk_masses else 1.0
        warning_count = sum(
            1 for m in topk_masses
            if m < self.detector.config.min_topk_mass_warning
        )

        return EvaluationSummary(
            mode=self.mode,
            num_examples=len(examples),
            token_metrics=token_metrics,
            response_metrics=response_metrics,
            span_metrics=span_metrics_result,
            mean_topk_mass=mean_topk_mass,
            min_topk_mass=min_topk_mass,
            topk_mass_warning_count=warning_count,
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

        topk_masses = []

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
            all_response_labels.append(example.response_label or 0)

            # Collect topk_mass
            for ts in token_signals:
                if ts.topk_mass is not None:
                    topk_masses.append(ts.topk_mass)

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

        mean_topk_mass = sum(topk_masses) / len(topk_masses) if topk_masses else 0.0
        min_topk_mass = min(topk_masses) if topk_masses else 1.0
        warning_count = sum(
            1 for m in topk_masses
            if m < self.detector.config.min_topk_mass_warning
        )

        return EvaluationSummary(
            mode=self.mode,
            num_examples=len(examples),
            token_metrics=token_metrics,
            response_metrics=response_metrics,
            mean_topk_mass=mean_topk_mass,
            min_topk_mass=min_topk_mass,
            topk_mass_warning_count=warning_count,
            example_results=example_results,
        )

    def _compute_token_risks(
        self,
        token_signals: List[TokenSignals],
    ) -> List[float]:
        """Compute token risks from signals (simple average for now)."""
        risks = []
        for signals in token_signals:
            # TODO: Update for DSG signals (cus, pos, dps)
            risk = signals.jsd_cand
            risks.append(min(1.0, max(0.0, risk)))
        return risks

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
