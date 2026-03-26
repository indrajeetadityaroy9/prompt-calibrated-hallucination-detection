from dataclasses import dataclass


@dataclass
class TokenSignals:
    rho: float
    phi: float
    spf: float
    mlp: float
    ent: float


@dataclass
class RiskySpan:
    start: int
    end: int
    peak_cusum: float


@dataclass
class DetectionResult:
    generated_text: str
    token_signals: list[TokenSignals]
    token_risks: list[float]
    cusum_values: list[float]
    risky_spans: list[RiskySpan]
    response_risk: float
    is_flagged: bool
    num_tokens: int
    prompt_length: int
