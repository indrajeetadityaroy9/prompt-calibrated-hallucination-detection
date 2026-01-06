"""
JEPA Latent Predictor for Drift Detection.

A lightweight neural network that learns the "Physics of Truth" -
how thought vectors should evolve in the model's latent space during
coherent, factual text generation.

Architecture:
    Input (4096) -> Bottleneck (1024) -> Output (4096)

The bottleneck forces semantic compression, learning abstract
transition dynamics rather than memorizing specific token pairs.

Usage:
    predictor = JepaPredictor()
    predicted_next = predictor(current_hidden_state)
    drift = F.mse_loss(predicted_next, actual_next)
"""

import torch
import torch.nn as nn


class JepaPredictor(nn.Module):
    """
    A lightweight Latent Predictor for the JEPA architecture.
    Maps: Hidden_State(t) -> Hidden_State(t+1)

    Architecture:
    - Input: 4096 (Llama-3 Hidden Dim)
    - Bottleneck: 1024 (Compression forces semantic learning)
    - Output: 4096 (Predicted Next State)

    The residual connection means we predict the *change* in thought,
    not just the next thought directly. This makes training more stable.
    """

    def __init__(self, input_dim: int = 4096, hidden_dim: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize weights for stability
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the next hidden state given current hidden state.

        Args:
            x: Current hidden state [batch, dim] or [dim]

        Returns:
            Predicted next hidden state (same shape as input)
        """
        # Residual connection: predict the *change* in thought
        # Prediction = Current + Delta
        delta = self.net(x)
        return x + delta

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "JepaPredictor":
        """Load a trained predictor from disk."""
        model = cls()
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
