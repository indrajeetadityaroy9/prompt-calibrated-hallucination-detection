"""
Configuration for the hallucination detector.
"""

from dataclasses import dataclass, field
from typing import List, Union, Optional


@dataclass
class DetectorConfig:
    """
    Configuration for HallucinationDetector.

    Attributes:
        layer_subset: Which layers to hook for signal computation.
            Options: "last_third", "last_quarter", "all", or List[int] for explicit indices.
        candidate_topk: Number of top tokens to include in candidate set for JSD.
        include_prev_topk: Whether to include previous step's top-k in candidate set.
        token_flag_threshold: Threshold for flagging individual tokens as high-risk.
        span_build_threshold: Threshold for including tokens in risk spans.
        response_flag_threshold: Threshold for flagging entire response as high-risk.
        multi_gpu_mode: How to handle 70B models with device_map='auto'.
            Options: "compatible_only" (only hook layers on lm_head device),
                     "allow_transfer" (allow cross-device tensor transfer).
        default_eval_mode: Default evaluation mode.
            Options: "forced_decoding" (stepwise with ground truth),
                     "generation" (argmax decoding).
        min_topk_mass_warning: Warn if topk_mass falls below this threshold.
        compute_topk_mass_every_n: Compute topk_mass every N tokens (for performance).
    """

    # Layer selection
    layer_subset: Union[str, List[int]] = "last_third"

    # Candidate set (for JSD)
    candidate_topk: int = 128
    include_prev_topk: bool = True

    # Thresholds (None = dynamic from distribution, set explicitly for fixed thresholds)
    token_flag_threshold: Optional[float] = None
    span_build_threshold: Optional[float] = None
    response_flag_threshold: Optional[float] = None

    # 70B multi-GPU
    multi_gpu_mode: str = "compatible_only"

    # Evaluation
    default_eval_mode: str = "forced_decoding"

    # Sanity thresholds
    min_topk_mass_warning: float = 0.8

    # Performance tuning
    compute_topk_mass_every_n: int = 1

    # Prompt-Anchored Aggregation
    # If True, uses prompt-anchored normalization + Noisy-OR fusion
    use_prompt_anchored: bool = True
    # Signals to include in aggregation
    # TODO: Update for DSG signals (cus, pos, dps)
    prompt_anchored_signals: tuple = ("jsd",)


    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate layer_subset
        if isinstance(self.layer_subset, str):
            valid_subsets = {"last_third", "last_quarter", "all"}
            if self.layer_subset not in valid_subsets:
                raise ValueError(
                    f"layer_subset must be one of {valid_subsets} or a list of ints, "
                    f"got '{self.layer_subset}'"
                )

        # Validate multi_gpu_mode
        valid_modes = {"compatible_only", "allow_transfer"}
        if self.multi_gpu_mode not in valid_modes:
            raise ValueError(
                f"multi_gpu_mode must be one of {valid_modes}, "
                f"got '{self.multi_gpu_mode}'"
            )

        # Validate eval_mode
        valid_eval_modes = {"forced_decoding", "generation"}
        if self.default_eval_mode not in valid_eval_modes:
            raise ValueError(
                f"default_eval_mode must be one of {valid_eval_modes}, "
                f"got '{self.default_eval_mode}'"
            )

        # Validate thresholds if explicitly set (None = dynamic)
        for name in ["token_flag_threshold", "span_build_threshold", "response_flag_threshold"]:
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass
class TokenSignals:
    """
    Per-token signals.

    TODO: Replace with DSGTokenSignals (cus, pos, dps) during DSG implementation.
    Currently only jsd_cand is populated.
    """

    # Core signal (retained for POS foundation)
    jsd_cand: float = 0.0  # MLP-induced shift (candidate-set JSD)

    # Sanity checks for candidate approximation validity
    candidate_size: int = 0
    emitted_in_candidate: bool = True
    topk_mass: Optional[float] = None  # sum(probs[candidate_set])

    def as_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "jsd_cand": self.jsd_cand,
            "candidate_size": self.candidate_size,
            "emitted_in_candidate": self.emitted_in_candidate,
            "topk_mass": self.topk_mass,
        }


@dataclass
class SpanRisk:
    """Risk information for a contiguous span of tokens."""

    start_token: int  # Start position in generated sequence
    end_token: int  # End position (exclusive)
    text: str  # Decoded text of the span
    risk_score: float  # Aggregated risk for the span
    token_risks: List[float]  # Individual token risks within span


@dataclass
class DetectionResult:
    """Complete detection result for a generated response."""

    # Generated text
    generated_text: str

    # Token-level results
    token_signals: List[TokenSignals]
    token_risks: List[float]  # Aggregated per-token risk scores

    # Span-level results
    risky_spans: List[SpanRisk]

    # Response-level results
    response_risk: float
    is_flagged: bool

    # Metadata
    num_tokens: int
    num_claim_tokens: int
    prompt_length: int

    # Sanity metrics
    mean_topk_mass: Optional[float] = None
    min_topk_mass: Optional[float] = None
