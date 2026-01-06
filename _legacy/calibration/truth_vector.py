"""
Truth Vector Calibration for Intrinsic Hallucination Detection.

Extracts a geometric "truthfulness direction" from the model's residual stream
by contrasting factual vs counterfactual hidden states.

The Truth Vector enables detection of hallucinations WITHOUT external context
by projecting response embeddings onto the calibrated direction.

Usage:
    # Calibration (offline)
    calibrator = TruthVectorCalibrator(model, tokenizer)
    for fact, lie in pairs:
        calibrator.add_pair(fact, lie)
    calibrator.compute_and_save("data/truth_vectors/llama3_8b.pt")

    # Inference (online)
    vector, meta = TruthVectorCalibrator.load("data/truth_vectors/llama3_8b.pt")
    score = compute_intrinsic_score(hidden_state, vector, meta)
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F


@dataclass
class TruthVectorConfig:
    """Configuration for Truth Vector calibration."""

    layer_ratio: float = 0.5  # Which layer to extract from (0.5 = middle)
    normalize: bool = True  # L2-normalize the truth vector


class TruthVectorCalibrator:
    """
    Calibrates a Truth Vector from fact/counterfact pairs.

    Uses Welford's online mean algorithm for memory-efficient accumulation.
    Stores normalization bounds (mu_pos, mu_neg) for proper scaling at inference.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[TruthVectorConfig] = None,
    ):
        """
        Initialize calibrator.

        Args:
            model: HuggingFace model (must support output_hidden_states=True)
            tokenizer: Corresponding tokenizer
            config: Calibration configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TruthVectorConfig()

        # Determine target layer
        num_layers = model.config.num_hidden_layers
        self.target_layer = int(num_layers * self.config.layer_ratio)

        # Welford's online mean storage (memory-safe)
        self.fact_sum: Optional[torch.Tensor] = None
        self.counterfact_sum: Optional[torch.Tensor] = None
        self.n_samples = 0

        # Device detection
        self.device = next(model.parameters()).device

    def _extract_hidden(self, text: str) -> torch.Tensor:
        """
        Extract hidden state from target layer for given text.

        Returns the last token's hidden state, which represents the
        accumulated meaning of the input.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden state from target layer, last token position
        # hidden_states is a tuple of (num_layers + 1,) tensors
        hidden_states = outputs.hidden_states[self.target_layer]  # (1, S, D)
        last_token_hidden = hidden_states[0, -1, :]  # (D,)

        return last_token_hidden.detach()

    def add_pair(self, fact: str, counterfact: str):
        """
        Add a fact/counterfact pair to the calibration set.

        Args:
            fact: True statement (e.g., "Paris is the capital of France")
            counterfact: False but plausible statement (e.g., "Berlin is the capital of France")

        Note: Counterfacts should be PLAUSIBLE, not absurd. "Paris is on Mars" will
        teach nonsense detection, not lie detection.
        """
        h_fact = self._extract_hidden(fact)
        h_lie = self._extract_hidden(counterfact)

        if self.fact_sum is None:
            self.fact_sum = torch.zeros_like(h_fact)
            self.counterfact_sum = torch.zeros_like(h_lie)

        self.fact_sum += h_fact
        self.counterfact_sum += h_lie
        self.n_samples += 1

    def compute_and_save(self, save_path: str):
        """
        Compute the Truth Vector and save with normalization bounds.

        The saved file contains:
        - vector: (D,) normalized truth direction
        - meta: dict with layer_index, n_samples, mu_pos, mu_neg

        Args:
            save_path: Path to save the .pt file
        """
        if self.n_samples == 0:
            raise ValueError("No samples added. Call add_pair() first.")

        mean_fact = self.fact_sum / self.n_samples
        mean_lie = self.counterfact_sum / self.n_samples

        # Compute the truth direction
        diff = mean_fact - mean_lie
        if self.config.normalize:
            vector = F.normalize(diff, p=2, dim=-1)
        else:
            vector = diff

        # Compute normalization bounds
        # These are the expected cosine similarities for facts and lies
        mu_pos = F.cosine_similarity(mean_fact.unsqueeze(0), vector.unsqueeze(0), dim=-1).item()
        mu_neg = F.cosine_similarity(mean_lie.unsqueeze(0), vector.unsqueeze(0), dim=-1).item()

        # Prepare save data
        data = {
            "vector": vector.cpu(),
            "meta": {
                "layer_index": self.target_layer,
                "layer_ratio": self.config.layer_ratio,
                "n_samples": self.n_samples,
                "mu_pos": mu_pos,
                "mu_neg": mu_neg,
                "model_name": getattr(self.model.config, "_name_or_path", "unknown"),
            }
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save(data, save_path)

        print(f"✅ Saved Truth Vector to {save_path}")
        print(f"   Layer: {self.target_layer} (ratio={self.config.layer_ratio})")
        print(f"   Samples: {self.n_samples}")
        print(f"   Bounds: Lie={mu_neg:.4f} | Truth={mu_pos:.4f}")
        print(f"   Gap: {mu_pos - mu_neg:.4f}")

        # Warn if gap is too small
        if abs(mu_pos - mu_neg) < 0.01:
            print("⚠️  WARNING: Gap < 0.01 - calibration may have failed!")
            print("   Try: different layer_ratio, more samples, or better data quality")

        return vector, data["meta"]

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Load a pre-calibrated Truth Vector.

        Args:
            path: Path to the .pt file
            device: Device to load the vector to

        Returns:
            vector: (D,) truth direction tensor
            meta: Dictionary with layer_index, n_samples, mu_pos, mu_neg
        """
        data = torch.load(path, map_location=device)
        return data["vector"], data["meta"]


def compute_intrinsic_score(
    hidden_state: torch.Tensor,
    truth_vector: torch.Tensor,
    meta: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute intrinsic truthfulness score.

    Args:
        hidden_state: (D,) or (B, D) hidden state from target layer
        truth_vector: (D,) calibrated truth direction
        meta: Optional metadata with mu_pos, mu_neg for normalization

    Returns:
        score: Scalar or (B,) in [0, 1] where 1 = truthful, 0 = lie
    """
    # Compute raw cosine similarity
    if hidden_state.dim() == 1:
        raw_sim = F.cosine_similarity(
            hidden_state.unsqueeze(0),
            truth_vector.unsqueeze(0),
            dim=-1
        ).squeeze()
    else:
        raw_sim = F.cosine_similarity(
            hidden_state,
            truth_vector.unsqueeze(0),
            dim=-1
        )

    # Apply normalization if metadata provided
    if meta is not None and "mu_pos" in meta and "mu_neg" in meta:
        mu_pos = meta["mu_pos"]
        mu_neg = meta["mu_neg"]
        denom = mu_pos - mu_neg

        if abs(denom) < 1e-6:
            # Fallback: if vector is collapsed, use raw similarity
            score = (raw_sim + 1) / 2
        else:
            score = (raw_sim - mu_neg) / denom
    else:
        # No normalization - map [-1, 1] to [0, 1]
        score = (raw_sim + 1) / 2

    return score.clamp(0.0, 1.0)
