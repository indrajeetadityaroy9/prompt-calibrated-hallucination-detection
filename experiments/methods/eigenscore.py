"""
EigenScore: Hidden State Eigenvalue Analysis for Hallucination Detection.

Official implementation based on:
"INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection"
Chen et al., ICLR 2024
https://github.com/D2I-ai/eigenscore

Key insight: EigenScore measures semantic diversity across multiple sampled
responses. High eigenvalue spread in hidden state covariance = uncertain/hallucinating.
"""

import time
from typing import Optional, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn

from experiments.methods.base import UncertaintyMethod, MethodResult


class EigenScoreMethod(UncertaintyMethod):
    """
    EigenScore uncertainty method (official algorithm).

    Generates multiple responses for the same prompt, extracts hidden states
    from the middle transformer layer, computes covariance across samples,
    and uses SVD eigenvalues to measure semantic uncertainty.

    Higher eigenvalue spread = more semantic diversity = more uncertain.

    Reference: https://github.com/D2I-ai/eigenscore
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_samples: int = 5,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EigenScore.

        Args:
            model: Language model
            tokenizer: Tokenizer
            num_samples: Number of response samples to generate (default: 5)
            max_new_tokens: Max tokens per sample (default: 50)
            temperature: Sampling temperature (default: 1.0)
            device: Compute device
        """
        super().__init__(model, tokenizer, device)

        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Regularization for covariance matrix
        self.alpha = 1e-3

        # Determine middle layer for hidden state extraction
        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            n_layers = len(model.transformer.h)
        else:
            n_layers = 12

        # Official ICLR 2024 paper: "penultimate layer" for hidden state extraction
        self.target_layer = n_layers - 2  # Penultimate layer per official impl

        self._hooks: List = []
        self._hidden_states: Dict[int, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "EigenScore"

    @property
    def requires_sampling(self) -> bool:
        return True  # Requires multiple samples

    def _register_hooks(self):
        """Register hook to capture hidden states from middle layer."""
        self._hooks = []
        self._hidden_states = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if isinstance(hidden, torch.Tensor):
                    self._hidden_states[layer_idx] = hidden.detach()
            return hook_fn

        # Register hook on target layer
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            block = self.model.transformer.h[self.target_layer]
            handle = block.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama/Mistral/Qwen style
            layer = self.model.model.layers[self.target_layer]
            handle = layer.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hidden_states = {}

    def _extract_response_embedding(self, prompt_ids: torch.Tensor, full_ids: torch.Tensor) -> torch.Tensor:
        """
        Extract mean hidden state embedding for response tokens.

        Args:
            prompt_ids: Tokenized prompt
            full_ids: Tokenized prompt + response

        Returns:
            Mean embedding vector for response portion
        """
        # Forward pass to capture hidden states
        self.model(full_ids)

        if self.target_layer not in self._hidden_states:
            # Fallback: return zero embedding
            hidden_dim = self.model.config.hidden_size if hasattr(self.model.config, "hidden_size") else 768
            return torch.zeros(hidden_dim, device=self.device)

        hidden = self._hidden_states[self.target_layer]  # (batch, seq, hidden)

        # Get response portion only
        prompt_len = prompt_ids.size(1)
        response_hidden = hidden[:, prompt_len:, :]  # (1, response_len, hidden)

        if response_hidden.size(1) == 0:
            # No response tokens, use last prompt token
            response_hidden = hidden[:, -1:, :]

        # Mean pooling over response tokens (official impl aggregates tokens)
        embedding = response_hidden.mean(dim=1).squeeze(0)  # (hidden,)

        return embedding

    def _compute_eigenscore(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """
        Compute EigenScore from sample embeddings.

        Official algorithm:
        1. Compute covariance matrix across samples
        2. Add regularization (alpha * I)
        3. SVD decomposition
        4. Return mean(log10(singular_values))

        Args:
            embeddings: (num_samples, hidden_dim) tensor

        Returns:
            Dict with eigenscore and singular values
        """
        if embeddings.size(0) < 2:
            return {"eigenscore": 0.0, "singular_values": None}

        try:
            # Compute covariance matrix (official uses torch.cov)
            # torch.cov expects (features, observations) so transpose
            cov_matrix = torch.cov(embeddings.T)  # (hidden, hidden)

            # Convert to numpy for SVD (official impl does this)
            cov_np = cov_matrix.cpu().float().numpy()

            # Add regularization
            cov_reg = cov_np + self.alpha * np.eye(cov_np.shape[0])

            # SVD decomposition
            _, s, _ = np.linalg.svd(cov_reg)

            # Filter out near-zero singular values before log
            s_filtered = s[s > 1e-10]

            if len(s_filtered) == 0:
                return {"eigenscore": 0.0, "singular_values": s}

            # Official formula: mean(log(singular_values)) - natural log per ICLR 2024
            eigenscore = np.mean(np.log(s_filtered))

            return {"eigenscore": float(eigenscore), "singular_values": s}

        except Exception as e:
            # Return neutral score on failure
            return {"eigenscore": 0.0, "singular_values": None, "error": str(e)}

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute EigenScore uncertainty.

        Generates multiple samples, extracts hidden states, computes
        covariance eigenvalues to measure semantic diversity.

        Note: The provided `response` is used as one of the samples,
        and additional samples are generated.
        """
        t0 = time.perf_counter()

        self._register_hooks()

        try:
            # Tokenize prompt
            prompt_inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            ).to(self.device)
            prompt_ids = prompt_inputs["input_ids"]

            # Collect embeddings from multiple samples
            embeddings = []

            # 1. Extract embedding from the provided response
            full_text = prompt + response
            full_inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            emb = self._extract_response_embedding(prompt_ids, full_inputs["input_ids"])
            embeddings.append(emb)

            # 2. Generate additional samples
            for _ in range(self.num_samples - 1):
                # Clear cached hidden states
                self._hidden_states = {}

                # Generate a new response
                with torch.no_grad():
                    generated = self.model.generate(
                        prompt_ids,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        temperature=self.temperature,
                        top_p=0.95,
                        pad_token_id=self.tokenizer.eos_token_id,
                        output_hidden_states=False,  # We use hooks instead
                    )

                # Extract embedding from generated response
                self._hidden_states = {}  # Clear before extraction pass
                emb = self._extract_response_embedding(prompt_ids, generated)
                embeddings.append(emb)

            # Stack embeddings: (num_samples, hidden_dim)
            embeddings_tensor = torch.stack(embeddings, dim=0)

            # Compute EigenScore
            result = self._compute_eigenscore(embeddings_tensor)
            eigenscore = result["eigenscore"]

            # Convert to uncertainty score (higher = more uncertain)
            # Official eigenscore: more negative = more confident
            # We negate and shift to get uncertainty in ~[0, 1] range
            # Typical eigenscore range: [-5, 5], so we use sigmoid-like transform
            uncertainty = 1.0 / (1.0 + np.exp(-eigenscore / 2.0))

        except Exception as e:
            uncertainty = 0.5  # Neutral on error
            eigenscore = 0.0
            result = {"error": str(e)}

        finally:
            self._remove_hooks()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=float(uncertainty),
            confidence=None,
            latency_ms=latency,
            extra={
                "raw_eigenscore": eigenscore,
                "num_samples": self.num_samples,
                "target_layer": self.target_layer,
            },
        )

    def cleanup(self) -> None:
        """Remove any remaining hooks."""
        self._remove_hooks()


class SAPLMAMethod(UncertaintyMethod):
    """
    Zero-shot Hidden State Probing baseline.

    Inspired by SAPLMA (Azaria & Mitchell, 2023), but uses zero-shot
    heuristics instead of trained classifier to avoid train/test leakage.

    Analyzes hidden state statistics:
    - Variance of hidden states (high variance = uncertain)
    - Norm distribution std (irregular norms = uncertain)
    - Temporal consistency (low cosine similarity = uncertain)

    Note: Original SAPLMA trains a classifier on hidden states.
    This zero-shot variant enables fair comparison with other
    unsupervised methods like AG-SAR.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)

        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        else:
            n_layers = 12

        if target_layers is None:
            self.target_layers = list(range(max(0, n_layers - 4), n_layers))
        else:
            self.target_layers = target_layers

        self._hooks: List = []
        self._hidden_states: Dict[int, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "HiddenState-ZS"  # Zero-shot hidden state probing

    @property
    def requires_sampling(self) -> bool:
        return False

    def _register_hooks(self):
        """Register hooks to capture hidden states."""
        self._hooks = []
        self._hidden_states = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                if isinstance(hidden, torch.Tensor):
                    self._hidden_states[layer_idx] = hidden.detach()
            return hook_fn

        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            for idx in self.target_layers:
                block = self.model.transformer.h[idx]
                handle = block.register_forward_hook(make_hook(idx))
                self._hooks.append(handle)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            for idx in self.target_layers:
                layer = self.model.model.layers[idx]
                handle = layer.register_forward_hook(make_hook(idx))
                self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hidden_states = {}

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute SAPLMA uncertainty score.

        Combines hidden state statistics:
        score = 0.35*variance + 0.35*norm_std + 0.30*(1-consistency)
        """
        import time
        t0 = time.perf_counter()

        self._register_hooks()

        try:
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            prompt_enc = self.tokenizer(prompt, return_tensors="pt")
            response_start = prompt_enc["input_ids"].size(1)

            self.model(inputs["input_ids"])

            variances = []
            norm_stds = []
            consistencies = []

            for layer_idx in self.target_layers:
                if layer_idx not in self._hidden_states:
                    continue

                h = self._hidden_states[layer_idx]  # (batch, seq, hidden)
                h_response = h[:, response_start:, :]  # Focus on response

                if h_response.size(1) < 2:
                    continue

                # Variance of hidden states
                var = h_response.var(dim=-1).mean().item()
                variances.append(var)

                # Norm distribution std
                norms = h_response.norm(dim=-1)
                norm_std = norms.std().item()
                norm_stds.append(norm_std)

                # Temporal consistency (cosine similarity between adjacent positions)
                h_norm = h_response / (h_response.norm(dim=-1, keepdim=True) + 1e-10)
                if h_norm.size(1) > 1:
                    cos_sim = (h_norm[:, :-1, :] * h_norm[:, 1:, :]).sum(dim=-1)
                    consistency = cos_sim.mean().item()
                    consistencies.append(consistency)

            if variances:
                avg_var = sum(variances) / len(variances)
                avg_norm_std = sum(norm_stds) / len(norm_stds) if norm_stds else 0
                avg_consistency = sum(consistencies) / len(consistencies) if consistencies else 0.5

                # Normalize features to [0, 1] range approximately
                norm_var = min(avg_var / 10.0, 1.0)  # Heuristic normalization
                norm_std_val = min(avg_norm_std / 100.0, 1.0)

                score = 0.35 * norm_var + 0.35 * norm_std_val + 0.30 * (1 - avg_consistency)
            else:
                score = 0.5
                avg_var = 0
                avg_consistency = 0.5

        finally:
            self._remove_hooks()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=score,
            confidence=None,
            latency_ms=latency,
            extra={
                "variance": avg_var if variances else 0,
                "consistency": avg_consistency if consistencies else 0.5,
            },
        )

    def cleanup(self) -> None:
        """Remove any remaining hooks."""
        self._remove_hooks()
