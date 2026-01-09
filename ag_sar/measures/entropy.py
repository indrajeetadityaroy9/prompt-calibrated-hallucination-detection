"""
Token Entropy and Varentropy for uncertainty quantification.

Varentropy (variance of entropy) detects oscillating confidence - a hallucination signal.
"""

from typing import Optional, Tuple
import torch


def compute_token_entropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-token predictive entropy: H = -sum(p * log p)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    raw_entropy = -torch.sum(probs * log_probs, dim=-1)

    # Align: entropy[i] = uncertainty about token[i]
    # In LM semantics, logits[i] predicts token[i+1], so we shift.
    # Position 0 is initialized with its own value (no preceding context).
    entropy = torch.zeros_like(raw_entropy)
    entropy[:, 1:] = raw_entropy[:, :-1]
    entropy[:, 0] = raw_entropy[:, 0]  # Fix: initialize position 0

    if attention_mask is not None:
        entropy = entropy * attention_mask.float()
    return entropy


def compute_varentropy(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute varentropy (variance of entropy) per token.

    High varentropy = oscillating confidence = hallucination signal.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)

    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)
    surprisal = -log_probs
    squared_deviation = (surprisal - entropy) ** 2
    raw_varentropy = torch.sum(probs * squared_deviation, dim=-1)

    # Align: varentropy[i] = uncertainty variance about token[i]
    # Position 0 is initialized with its own value (no preceding context).
    varentropy = torch.zeros_like(raw_varentropy)
    varentropy[:, 1:] = raw_varentropy[:, :-1]
    varentropy[:, 0] = raw_varentropy[:, 0]  # Fix: initialize position 0

    if attention_mask is not None:
        varentropy = varentropy * attention_mask.float()
    return varentropy


def compute_epiplexity(
    varentropy: torch.Tensor,
    tau: float = 5.0,
) -> torch.Tensor:
    """
    Absolute Cognitive Load Metric (v3.4).

    E_t = V_t / τ

    Theory: Varentropy measures "Structural Effort" or "Cognitive Load".
    Research (Entropix, Finzi) suggests reasoning happens at V > 4.0 for Llama-3.
    - V < τ: Rote retrieval (low cognitive load) → Risk of hallucination
    - V ≥ τ: Active synthesis (high cognitive load) → Grounded reasoning

    This is CALIBRATION-FREE. We use a universal threshold τ instead of
    prompt-relative comparison, because prompt complexity doesn't predict
    required answer complexity (simple questions can need complex answers).

    Empirical data (HaluEval):
    - Hallucination: V ≈ 3.8 → E = 0.76
    - Fact: V ≈ 5.2 → E = 1.04

    Args:
        varentropy: Per-token varentropy [B, S]
        tau: Cognitive load threshold (default 5.0 for Llama-3 scale)

    Returns:
        Epiplexity ratio, shape [B, S]. E < 1 = low effort, E ≥ 1 = high effort
    """
    # Raw ratio (no clamping - let authority.py handle via power law)
    return varentropy / tau


def compute_epiplexity_absolute(
    varentropy: torch.Tensor,
    tau_var: float = 3.0,
) -> torch.Tensor:
    """
    Compute epiplexity proxy from varentropy via absolute tanh scaling (v3.2).

    E_t = tanh(V_t / τ_var)

    Universal scale that works across all datasets and models without calibration.
    Based on observed varentropy ranges:
      - V=0 → E=0 (Rote/Fact)
      - V=3 → E≈0.76 (Reasoning)
      - V>5 → E→1.0 (Confusion/High Structure)

    Args:
        varentropy: Per-token varentropy [B, S]
        tau_var: Scaling factor (default 3.0, based on empirical varentropy range)

    Returns:
        Epiplexity proxy in [0, 1], shape [B, S]
    """
    return torch.tanh(varentropy / tau_var)


def compute_gaussian_complexity(
    varentropy: torch.Tensor,
    varentropy_mu: float,
    sigma: float = 0.5,
    epsilon: float = 0.1,
    center: float = 1.0,
) -> torch.Tensor:
    """
    v6 Gaussian Complexity Matching - penalizes complexity deviations from expected ratio.

    Solves the "Goldilocks Information Problem":
    - HaluEval hallucinations: V >> V_prompt (confusion/overcomplexity)
    - RAGTruth hallucinations: V << V_prompt (oversimplification/glibness)
    - Valid facts: V ≈ expected_ratio × V_prompt (complexity preserved)

    Master equation:
        G(R_t) = exp(-(R_t - center)² / 2σ²)
        R_t = V_t / max(μ_prompt, ε)

    When R_t ≈ center (complexity matches expected), G → 1.0 (trusted).
    When R_t deviates from center, G → 0.0 (penalized).

    Args:
        varentropy: Per-token varentropy [B, S]
        varentropy_mu: Mean varentropy from prompt calibration
        sigma: Gaussian tolerance width (default 0.5)
            - σ=0.3: Strict - R_t must be very close to center
            - σ=0.5: Moderate (default) - R_t within ±0.5 of center scores well
            - σ=1.0: Lenient - Wide tolerance for complexity variation
        epsilon: Floor for mu to prevent division artifacts (default 0.1)
        center: Expected complexity ratio (default 1.0)
            - For short-answer QA: center ≈ 0.3-0.5 (response simpler than prompt)
            - For reasoning tasks: center ≈ 1.0 (response matches prompt)
            - For creative tasks: center ≈ 1.2-1.5 (response more complex)

    Returns:
        Gaussian complexity score [B, S] in [0, 1]
    """
    # Safe division: prevent artifacts when prompt has very low varentropy
    mu_safe = max(varentropy_mu, epsilon)

    # Complexity ratio: how does response complexity compare to prompt?
    R_t = varentropy / mu_safe

    # Gaussian penalty: peaks at R_t = center, decays for deviations
    deviation_sq = (R_t - center) ** 2
    return torch.exp(-deviation_sq / (2.0 * sigma ** 2))
