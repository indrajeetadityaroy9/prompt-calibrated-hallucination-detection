"""
Online JEPA Predictor with Test-Time Training (TTT).

This predictor adapts to the specific context provided in each prompt,
learning the "local physics" of that document's facts rather than
relying on general language patterns.

Key Insight: A general predictor allows any grammatically valid transition
(e.g., "France is watermelon" is grammatically fine). A context-adapted
predictor learns THIS document's facts and flags deviations.

Usage:
    predictor = OnlineJepaPredictor(input_dim=4096)

    # Test-Time Training: Adapt to context
    context_loss = predictor.fit(context_hidden_states, epochs=3)

    # Measure drift on response
    drift = predictor.compute_drift(response_input, response_target)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple


class OnlineJepaPredictor(nn.Module):
    """
    A lightweight MLP predictor with online adaptation capability.

    Architecture: input_dim -> hidden_dim -> input_dim (residual)

    The key feature is the `fit()` method which performs rapid
    Test-Time Training on the context portion of each prompt.
    """

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Architecture matches JepaPredictor for weight compatibility
        # GELU is used for activation (matches pretrained prior)
        # Dropout disabled at inference, but kept for structural compatibility
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),  # Match JepaPredictor for pretrained compatibility
            nn.Dropout(0.0),  # No dropout during TTT (but keeps layer indices aligned)
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize weights
        self._init_weights()

        # Optimizer for online adaptation
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.loss_fn = nn.MSELoss()

        # Track adaptation state
        self._initial_state = None
        self._is_adapted = False

    def _init_weights(self):
        """Xavier initialization for stability."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def save_initial_state(self):
        """Save weights before adaptation for later reset."""
        self._initial_state = {
            k: v.clone() for k, v in self.state_dict().items()
        }

    def reset(self):
        """Reset to initial state (before adaptation)."""
        if self._initial_state is not None:
            self.load_state_dict(self._initial_state)
            # Recreate optimizer with fresh state
            self.optimizer = optim.AdamW(
                self.net.parameters(),
                lr=self.lr,
            )
        else:
            # Reinitialize from scratch
            self._init_weights()
            self.optimizer = optim.AdamW(
                self.net.parameters(),
                lr=self.lr,
            )
        self._is_adapted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict next hidden state with residual connection.

        Args:
            x: Current hidden state (B, D) or (S, D)

        Returns:
            Predicted next state (same shape)
        """
        return x + self.net(x)

    def fit(
        self,
        hidden_states: torch.Tensor,
        epochs: int = 5,
        verbose: bool = False,
    ) -> float:
        """
        Test-Time Training: Rapidly adapt to the specific context.

        This is the key innovation. We train the predictor on the
        context's hidden state transitions so it "memorizes" the
        facts in this specific document.

        Args:
            hidden_states: Hidden states from context (S, D)
            epochs: Number of adaptation epochs (3-5 is usually enough)
            verbose: Print training progress

        Returns:
            Final training loss
        """
        if hidden_states.shape[0] < 2:
            # Need at least 2 tokens for transitions
            return 0.0

        self.train()

        # Create input/target pairs from the context sequence
        # Input: [h_0, ..., h_{T-1}]
        # Target: [h_1, ..., h_T]
        X = hidden_states[:-1].detach().float()  # Detach to stop gradient to LLM
        Y = hidden_states[1:].detach().float()

        final_loss = 0.0

        # Fast adaptation loop
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            pred = self(X)
            loss = self.loss_fn(pred, Y)
            loss.backward()
            self.optimizer.step()
            final_loss = loss.item()

            if verbose:
                print(f"  TTT Epoch {epoch+1}/{epochs}: Loss = {final_loss:.5f}")

        self.eval()
        self._is_adapted = True

        return final_loss

    def compute_drift(
        self,
        input_states: torch.Tensor,
        target_states: torch.Tensor,
    ) -> Tuple[float, Optional[torch.Tensor]]:
        """
        Compute prediction drift on response tokens.

        Args:
            input_states: h_t states (S, D)
            target_states: h_{t+1} states (S, D)

        Returns:
            Tuple of (average_drift, per_token_drift)
        """
        if input_states.shape[0] == 0:
            return 0.0, None

        self.eval()

        with torch.no_grad():
            input_float = input_states.float()
            target_float = target_states.float()

            preds = self(input_float)

            # MSE per token (averaged over hidden dim)
            mse_per_token = F.mse_loss(
                preds, target_float, reduction="none"
            ).mean(dim=-1)

        avg_drift = mse_per_token.mean().item()

        return avg_drift, mse_per_token.cpu()

    @property
    def is_adapted(self) -> bool:
        """Whether the predictor has been adapted to a context."""
        return self._is_adapted
