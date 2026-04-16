from dataclasses import dataclass

from torch import Tensor

SIGNAL_NAMES = ("rho", "phi", "spf", "mlp", "ent")


@dataclass
class LayerHiddenStates:
    h_resid_attn: Tensor
    h_resid_mlp: Tensor


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
