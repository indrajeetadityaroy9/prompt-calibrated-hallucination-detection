"""
Spectral baseline methods implementing UncertaintyMethod interface.

EigenScore: Uses eigenvalue analysis of attention matrices.
SAPLMA: Uses hidden state statistics.

These are the main "spectral competitor" baselines that analyze
internal model structure (similar to AG-SAR conceptually).
"""

import time
import math
from typing import Optional, List, Dict
import torch
import torch.nn as nn

from experiments.methods.base import UncertaintyMethod, MethodResult


class EigenScoreMethod(UncertaintyMethod):
    """
    EigenScore uncertainty method.

    Analyzes spectral properties of attention matrices:
    - Spectral entropy of eigenvalues
    - Dominant eigenvalue ratio
    - Eigenvalue decay rate

    Higher spectral entropy = more distributed attention = less confident.

    Note: Requires capturing attention weights, which uses hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize EigenScore.

        Args:
            model: Language model
            tokenizer: Tokenizer
            target_layers: Layers to analyze (default: last 4)
            device: Compute device
        """
        super().__init__(model, tokenizer, device)

        # Determine number of layers
        if hasattr(model.config, "num_hidden_layers"):
            n_layers = model.config.num_hidden_layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            n_layers = len(model.transformer.h)
        else:
            n_layers = 12

        if target_layers is None:
            self.target_layers = list(range(max(0, n_layers - 4), n_layers))
        else:
            self.target_layers = target_layers

        self._hooks: List = []
        self._attention_weights: Dict[int, torch.Tensor] = {}

    @property
    def name(self) -> str:
        return "EigenScore"

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

        # Find attention modules
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            # GPT-2 style
            for idx in self.target_layers:
                block = self.model.transformer.h[idx]
                handle = block.attn.register_forward_hook(make_hook(idx))
                self._hooks.append(handle)
        elif hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            # Llama style
            for idx in self.target_layers:
                layer = self.model.model.layers[idx]
                handle = layer.self_attn.register_forward_hook(make_hook(idx))
                self._hooks.append(handle)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._attention_weights = {}

    def _compute_spectral_features(
        self, attn_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute spectral features from attention weights.

        Args:
            attn_weights: (batch, heads, seq, seq) attention weights

        Returns:
            Dict with spectral entropy, dominant ratio, etc.
        """
        # Average over heads and batch
        attn = attn_weights.mean(dim=(0, 1))  # (seq, seq)

        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvalsh(attn)
            eigenvalues = eigenvalues.real.abs()
            eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)

            # Spectral entropy
            mask = eigenvalues > 1e-10
            entropy = -(eigenvalues[mask] * torch.log(eigenvalues[mask] + 1e-10)).sum()
            max_entropy = math.log(len(eigenvalues))
            spectral_entropy = (entropy / max_entropy).item() if max_entropy > 0 else 0

            # Dominant eigenvalue ratio
            sorted_eigs = torch.sort(eigenvalues, descending=True).values
            dominant_ratio = (sorted_eigs[0] / (sorted_eigs.sum() + 1e-10)).item()

            # Spectral norm
            spectral_norm = sorted_eigs[0].item()

            return {
                "spectral_entropy": spectral_entropy,
                "dominant_ratio": dominant_ratio,
                "spectral_norm": spectral_norm,
            }
        except Exception:
            return {
                "spectral_entropy": 0.5,
                "dominant_ratio": 0.5,
                "spectral_norm": 0.5,
            }

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> MethodResult:
        """
        Compute EigenScore uncertainty.

        Combines spectral features into a single score:
        score = 0.4*entropy + 0.3*(1-dominant) + 0.3*norm_decay
        """
        t0 = time.perf_counter()

        self._register_hooks()

        try:
            # Forward pass
            full_text = prompt + response
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            self.model(inputs["input_ids"], output_attentions=True)

            # Aggregate spectral features across layers
            all_features = []
            for layer_idx in self.target_layers:
                if layer_idx in self._attention_weights:
                    features = self._compute_spectral_features(
                        self._attention_weights[layer_idx]
                    )
                    all_features.append(features)

            if all_features:
                # Average features across layers
                avg_entropy = sum(f["spectral_entropy"] for f in all_features) / len(all_features)
                avg_dominant = sum(f["dominant_ratio"] for f in all_features) / len(all_features)

                # Combine into final score
                # High entropy + low dominance = uncertain
                score = 0.4 * avg_entropy + 0.3 * (1 - avg_dominant) + 0.3 * (1 - avg_dominant)
            else:
                score = 0.5

        finally:
            self._remove_hooks()

        latency = (time.perf_counter() - t0) * 1000

        return MethodResult(
            score=score,
            confidence=None,
            latency_ms=latency,
            extra={
                "spectral_entropy": avg_entropy if all_features else 0.5,
                "dominant_ratio": avg_dominant if all_features else 0.5,
            },
        )

    def cleanup(self) -> None:
        """Remove any remaining hooks."""
        self._remove_hooks()


class SAPLMAMethod(UncertaintyMethod):
    """
    SAPLMA (Self-Attention Probing for LLM Accuracy) baseline.

    Analyzes hidden state statistics:
    - Variance of hidden states
    - Norm distribution
    - Temporal consistency (cosine similarity across positions)

    Higher variance + lower consistency = more uncertain.
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
        return "SAPLMA"

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
                norm_std = min(avg_norm_std / 100.0, 1.0)

                score = 0.35 * norm_var + 0.35 * norm_std + 0.30 * (1 - avg_consistency)
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
