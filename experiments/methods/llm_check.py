"""
LLM-Check: Zero-shot Hallucination Detection via Internal States.

Official implementation based on:
"LLM-Check: Investigating Detection of Hallucinations in Large Language Models"
Sriramanan et al., NeurIPS 2024
https://github.com/GaurangSriramanan/LLM_Check_Hallucination_Detection

Three complementary scoring methods:
1. AttentionScore: Eigenvalues from attention matrix diagonals
2. HiddenScore: SVD of centered hidden state covariance
3. LogitScore: Token-level entropy (similar to predictive entropy)

All methods are zero-shot, single-pass, and compute-efficient.
"""

import time
from typing import Optional, List, Dict
import torch
import torch.nn as nn

from experiments.methods.base import UncertaintyMethod, MethodResult


class LLMCheckAttentionMethod(UncertaintyMethod):
    """
    LLM-Check Attention Score (NeurIPS 2024).

    Computes hallucination score from attention matrix diagonal eigenvalues.
    For causal attention, the diagonal represents self-attention weights.

    Formula: score = sum_heads(mean(log(diag(A))))

    Higher score (less negative) = more confident = less likely hallucination.
    We negate to get uncertainty score.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layer: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)

        # Determine target layer (default: layer 15 or middle)
        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        else:
            n_layers = 12

        if target_layer is None:
            self.target_layer = min(15, n_layers - 1)
        else:
            self.target_layer = target_layer

        self._hooks: List = []
        self._attention_weights: Dict[int, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "LLMCheck-Attn"

    @property
    def requires_sampling(self) -> bool:
        return False

    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        self._hooks = []
        self._attention_weights = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        self._attention_weights[layer_idx] = attn_weights.detach()
            return hook_fn

        # Register hook on target layer's attention
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            block = self.model.transformer.h[self.target_layer]
            handle = block.attn.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama/Mistral/Qwen style
            layer = self.model.model.layers[self.target_layer]
            handle = layer.self_attn.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._attention_weights = {}

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """Compute attention-based hallucination score."""
        t0 = time.perf_counter()

        self._register_hooks()

        try:
            # Tokenize
            prompt_enc = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].size(1)

            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            # Forward pass with attention output
            self.model(inputs["input_ids"], output_attentions=True)

            if self.target_layer not in self._attention_weights:
                # Fallback
                uncertainty = 0.5
            else:
                attn = self._attention_weights[self.target_layer]  # (batch, heads, seq, seq)

                # Focus on response portion
                response_attn = attn[:, :, prompt_len:, prompt_len:]

                if response_attn.size(2) == 0:
                    uncertainty = 0.5
                else:
                    # Compute diagonal eigenvalue score per head
                    # For causal attention, diagonal = self-attention weights
                    eigscore = 0.0
                    num_heads = response_attn.size(1)

                    for h in range(num_heads):
                        head_attn = response_attn[0, h]  # (seq, seq)
                        diag = torch.diagonal(head_attn, 0)
                        # Avoid log(0)
                        diag = torch.clamp(diag, min=1e-10)
                        eigscore += torch.log(diag).mean().item()

                    # Normalize by number of heads
                    eigscore /= num_heads

                    # Convert to uncertainty: more negative eigscore = more uncertain
                    # Typical range: [-3, 0], transform to [0, 1]
                    uncertainty = 1.0 / (1.0 + torch.exp(torch.tensor(eigscore * 2)).item())

        finally:
            self._remove_hooks()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=float(uncertainty),
            confidence=None,
            latency_ms=latency,
            extra={"layer": self.target_layer},
        )

    def cleanup(self) -> None:
        self._remove_hooks()


class LLMCheckHiddenMethod(UncertaintyMethod):
    """
    LLM-Check Hidden Score (NeurIPS 2024).

    Computes hallucination score from SVD of centered hidden state covariance.

    Formula: score = mean(log(SVD(Z^T * J * Z + alpha*I)))
    where J is the centering matrix and Z is hidden states.

    Higher score = more variance = more uncertain.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layer: Optional[int] = None,
        alpha: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)

        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        else:
            n_layers = 12

        if target_layer is None:
            self.target_layer = min(15, n_layers - 1)
        else:
            self.target_layer = target_layer

        self.alpha = alpha
        self._hooks: List = []
        self._hidden_states: Dict[int, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "LLMCheck-Hidden"

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
            block = self.model.transformer.h[self.target_layer]
            handle = block.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layer = self.model.model.layers[self.target_layer]
            handle = layer.register_forward_hook(make_hook(self.target_layer))
            self._hooks.append(handle)

    def _remove_hooks(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hidden_states = {}

    def _centered_svd_score(self, Z: torch.Tensor) -> float:
        """
        Compute centered SVD score.

        Args:
            Z: (seq_len, hidden_dim) hidden states

        Returns:
            Mean log singular value
        """
        n = Z.shape[0]
        if n < 2:
            return 0.0

        # Centering matrix J = I - (1/n) * ones
        Z = Z.float()  # Ensure float for numerical stability
        J = torch.eye(n, device=Z.device) - (1.0 / n) * torch.ones(n, n, device=Z.device)

        # Centered covariance: Sigma = Z^T * J * Z
        Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)

        # Regularization
        Sigma = Sigma + self.alpha * torch.eye(Sigma.shape[0], device=Sigma.device)

        # SVD
        try:
            svdvals = torch.linalg.svdvals(Sigma)
            svdvals = torch.clamp(svdvals, min=1e-10)
            eigscore = torch.log(svdvals).mean().item()
        except Exception:
            eigscore = 0.0

        return eigscore

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """Compute hidden state SVD hallucination score."""
        t0 = time.perf_counter()

        self._register_hooks()

        try:
            prompt_enc = self.tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].size(1)

            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            self.model(inputs["input_ids"])

            if self.target_layer not in self._hidden_states:
                uncertainty = 0.5
                eigscore = 0.0
            else:
                hidden = self._hidden_states[self.target_layer]  # (batch, seq, hidden)
                Z = hidden[0, prompt_len:, :]  # Response portion only

                if Z.size(0) < 2:
                    uncertainty = 0.5
                    eigscore = 0.0
                else:
                    eigscore = self._centered_svd_score(Z)

                    # Higher eigscore = more variance = more uncertain
                    # Typical range: [-5, 10], transform to [0, 1]
                    uncertainty = 1.0 / (1.0 + torch.exp(torch.tensor(-eigscore / 3)).item())

        finally:
            self._remove_hooks()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=float(uncertainty),
            confidence=None,
            latency_ms=latency,
            extra={"raw_eigscore": eigscore, "layer": self.target_layer},
        )

    def cleanup(self) -> None:
        self._remove_hooks()


class LLMCheckLogitMethod(UncertaintyMethod):
    """
    LLM-Check Logit Score (NeurIPS 2024).

    Computes hallucination score from output logit entropy.
    Similar to predictive entropy but with optional top-k filtering.

    Formula: score = mean(-p * log(p)) where p = softmax(logits)

    Higher entropy = more uncertain.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        top_k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, tokenizer, device)
        self.top_k = top_k

    @property
    def name(self) -> str:
        return "LLMCheck-Logit"

    @property
    def requires_sampling(self) -> bool:
        return False

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """Compute logit entropy hallucination score."""
        t0 = time.perf_counter()

        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].size(1)

        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        outputs = self.model(inputs["input_ids"])
        logits = outputs.logits[0]  # (seq, vocab)

        # Focus on response portion
        response_logits = logits[prompt_len:]

        if response_logits.size(0) == 0:
            entropy = 0.5
        else:
            if self.top_k is not None:
                # Top-k entropy
                top_logits = torch.topk(response_logits, self.top_k, dim=-1).values
                probs = torch.softmax(top_logits, dim=-1)
            else:
                probs = torch.softmax(response_logits, dim=-1)

            # Entropy: -sum(p * log(p))
            log_probs = torch.log(probs + 1e-10)
            token_entropy = -(probs * log_probs).sum(dim=-1)
            entropy = token_entropy.mean().item()

            # Normalize to [0, 1] - typical entropy range depends on vocab size
            # For top_k=10: max entropy = log(10) ≈ 2.3
            # For full vocab: max entropy = log(vocab_size) ≈ 10-11
            max_entropy = torch.log(torch.tensor(float(probs.size(-1)))).item()
            entropy = entropy / max_entropy

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=float(entropy),
            confidence=None,
            latency_ms=latency,
            extra={"top_k": self.top_k},
        )

    def cleanup(self) -> None:
        pass
