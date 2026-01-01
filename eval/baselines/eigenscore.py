"""
EigenScore baseline for uncertainty estimation.

Implements a simplified version of EigenScore (Chen et al., 2024).
Core idea: Generate K samples, compute covariance of hidden states,
use eigenvalue decomposition to measure uncertainty.

Complexity: O(K*N) - requires K generations per sample

Reference: "EigenScore: A Method for Quantifying LLM Uncertainty"
"""

from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
import numpy as np


class EigenScore:
    """
    EigenScore baseline.

    Generates K samples, extracts hidden states, computes covariance matrix,
    and uses eigenvalue spread as uncertainty metric.

    Key insight: If hidden states are consistent across generations (low variance),
    model is confident. High variance in hidden space = high uncertainty.

    Complexity: O(K*N) model forward passes per sample.

    Example:
        >>> es = EigenScore(model, tokenizer, k_samples=5)
        >>> uncertainty = es.compute_uncertainty("The capital of France is")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        k_samples: int = 5,
        temperature: float = 0.7,
        use_last_layer: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.k_samples = k_samples
        self.temperature = temperature
        self.use_last_layer = use_last_layer

        # Auto-detect device and dtype from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = next(model.parameters()).dtype
        else:
            self.dtype = dtype

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.inference_mode()
    def generate_samples_with_hidden_states(
        self,
        prompt: str,
        max_new_tokens: int = 20
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate K samples and extract hidden states for each.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Tuple of (responses, hidden_states_list)
            Each hidden_state is the mean-pooled last layer hidden state
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        responses = []
        hidden_states_list = []

        for _ in range(self.k_samples):
            # Generate with output_hidden_states
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Extract response
            response_ids = output.sequences[0, prompt_len:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response.strip())

            # Extract hidden states from generated tokens
            # hidden_states is a tuple of (num_generated_tokens,) tuples of (num_layers, batch, seq, hidden)
            if hasattr(output, 'hidden_states') and output.hidden_states:
                # Get last layer hidden states for each generated token
                token_hidden_states = []
                for step_hidden in output.hidden_states:
                    if step_hidden:
                        # step_hidden is tuple of (num_layers,), each (batch, seq, hidden)
                        # Get last layer, last token
                        last_layer_hidden = step_hidden[-1][:, -1, :]  # (batch, hidden)
                        token_hidden_states.append(last_layer_hidden)

                if token_hidden_states:
                    # Mean pool across generated tokens
                    stacked = torch.stack(token_hidden_states, dim=0)  # (num_tokens, batch, hidden)
                    mean_hidden = stacked.mean(dim=0).squeeze(0)  # (hidden,)
                    hidden_states_list.append(mean_hidden.cpu())
                else:
                    # Fallback: use a zero vector
                    hidden_dim = self.model.config.hidden_size
                    hidden_states_list.append(torch.zeros(hidden_dim))
            else:
                # Fallback: run forward pass to get hidden states
                full_input = self.tokenizer(
                    prompt + response, return_tensors="pt"
                ).to(self.device)
                out = self.model(
                    input_ids=full_input.input_ids,
                    output_hidden_states=True,
                    return_dict=True
                )
                # Get last layer, mean pool over response tokens
                last_hidden = out.hidden_states[-1]  # (batch, seq, hidden)
                response_hidden = last_hidden[:, prompt_len:, :].mean(dim=1)  # (batch, hidden)
                hidden_states_list.append(response_hidden.squeeze(0).cpu())

        return responses, hidden_states_list

    def _compute_eigenscore(self, hidden_states: List[torch.Tensor]) -> float:
        """
        Compute EigenScore from K hidden state vectors.

        EigenScore = trace(Covariance) / λ_max

        This measures the "spread" of hidden states - high spread = high uncertainty.

        Args:
            hidden_states: List of K hidden state vectors, each (hidden_dim,)

        Returns:
            EigenScore (higher = more uncertain)
        """
        if len(hidden_states) < 2:
            return 0.0

        # Stack into matrix: (K, hidden_dim)
        H = torch.stack(hidden_states, dim=0).float().numpy()

        # Center the data
        H_centered = H - H.mean(axis=0, keepdims=True)

        # Compute covariance: (hidden_dim, hidden_dim)
        # For efficiency, compute K x K gram matrix instead
        # Cov = H^T H / (K-1)
        cov = np.cov(H_centered.T)

        if cov.ndim == 0:
            return float(cov)

        # Compute eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.abs(eigenvalues)

            # EigenScore: trace / max_eigenvalue
            # This measures how "spread out" the variance is
            trace = np.sum(eigenvalues)
            max_eig = np.max(eigenvalues)

            if max_eig > 1e-10:
                eigenscore = trace / max_eig
            else:
                eigenscore = 0.0

            return float(eigenscore)
        except np.linalg.LinAlgError:
            return 0.0

    def compute_uncertainty(
        self,
        prompt: str,
        response: Optional[str] = None,
        max_new_tokens: int = 20
    ) -> float:
        """
        Compute EigenScore uncertainty.

        Note: response parameter is ignored - we generate our own samples.

        Args:
            prompt: Input prompt
            response: Ignored (for API compatibility)
            max_new_tokens: Max tokens to generate per sample

        Returns:
            EigenScore (higher = more uncertain)
        """
        # Generate K samples with hidden states
        responses, hidden_states = self.generate_samples_with_hidden_states(
            prompt, max_new_tokens=max_new_tokens
        )

        # Compute EigenScore
        score = self._compute_eigenscore(hidden_states)

        return score

    def compute_uncertainty_with_response(
        self,
        prompt: str,
        response: str
    ) -> float:
        """
        Compute uncertainty for a specific prompt-response pair.

        For fair comparison, we still generate K samples and compute
        EigenScore on the hidden state distribution.

        Args:
            prompt: Input prompt
            response: Given response (used to estimate max_new_tokens)

        Returns:
            EigenScore
        """
        return self.compute_uncertainty(prompt, max_new_tokens=len(response.split()) + 5)

    def batch_compute_uncertainty(
        self,
        prompts: List[str],
        responses: Optional[List[str]] = None,
        max_new_tokens: int = 20
    ) -> List[float]:
        """
        Compute uncertainty for multiple prompts.

        Args:
            prompts: List of prompts
            responses: Ignored
            max_new_tokens: Max tokens to generate

        Returns:
            List of EigenScore values
        """
        return [
            self.compute_uncertainty(p, max_new_tokens=max_new_tokens)
            for p in prompts
        ]
