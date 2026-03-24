from dataclasses import dataclass


@dataclass
class TokenSignals:
    ent: float
    mlp: float
    psp: float
    spt: float
    spectral_gap: float


@dataclass
class DetectionResult:
    generated_text: str
    token_signals: list[TokenSignals]
    token_risks: list[float]
    risky_spans: list
    response_risk: float
    is_flagged: bool
    num_tokens: int
    prompt_length: int
