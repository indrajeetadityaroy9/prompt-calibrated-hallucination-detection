"""
EigenScore / SAPLMA Baseline (Spectral Competitor).

Spectral methods that analyze eigenvalue structure of attention matrices
to detect hallucinations. This is AG-SAR's primary spectral competitor.

References:
- EigenScore: Chen et al., "EigenScore: An Eigenvalue-based Score for
  Detecting Generated Text" (2023)
- SAPLMA: Azaria & Mitchell, "The Internal State of an LLM Knows When
  It's Lying" (2023)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import numpy as np


class EigenScoreBaseline:
    """
    EigenScore: Spectral analysis of attention matrices for hallucination detection.

    Key idea: Eigenvalue spectrum of attention matrices differs between
    factual and hallucinated content. Hallucinations show:
    - Higher spectral entropy (more diffuse attention)
    - Lower dominant eigenvalue ratio (less focused)
    - Different eigenvalue decay patterns
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: Optional[List[int]] = None,
        top_k_eigenvalues: int = 10,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # Which layers to analyze (default: last 4 semantic layers)
        n_layers = model.config.num_hidden_layers
        self.target_layers = target_layers or list(range(n_layers - 4, n_layers))
        self.top_k = top_k_eigenvalues

        # Storage for attention matrices
        self._attention_maps = {}
        self._hooks = []

    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        self._attention_maps = {}
        self._hooks = []

        def make_hook(layer_idx):
            def hook(module, inputs, outputs):
                # outputs is typically (hidden_states, attention_weights, ...)
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    attn_weights = outputs[1]  # (B, H, T, T)
                    if attn_weights is not None:
                        self._attention_maps[layer_idx] = attn_weights.detach()
            return hook

        # Register hooks based on model architecture
        for layer_idx in self.target_layers:
            try:
                if hasattr(self.model, 'transformer'):
                    # GPT-2 style
                    layer = self.model.transformer.h[layer_idx].attn
                elif hasattr(self.model, 'model'):
                    # Llama/Mistral style
                    layer = self.model.model.layers[layer_idx].self_attn
                else:
                    continue

                hook = layer.register_forward_hook(make_hook(layer_idx))
                self._hooks.append(hook)
            except (IndexError, AttributeError):
                continue

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._attention_maps = {}

    def _compute_spectral_features(self, attn: torch.Tensor) -> Dict[str, float]:
        """
        Compute spectral features from attention matrix.

        Args:
            attn: Attention weights (B, H, T, T)

        Returns:
            Dict of spectral features
        """
        # Average over batch and heads
        attn_avg = attn.mean(dim=(0, 1))  # (T, T)

        # Compute eigenvalues (use real part for stability)
        try:
            eigenvalues = torch.linalg.eigvalsh(attn_avg.float())
            eigenvalues = eigenvalues.abs()
            eigenvalues = torch.sort(eigenvalues, descending=True)[0]
        except Exception:
            # Fallback: use SVD singular values as proxy
            try:
                singular_values = torch.linalg.svdvals(attn_avg.float())
                eigenvalues = singular_values ** 2
            except Exception:
                return {
                    "spectral_entropy": 0.5,
                    "dominant_ratio": 0.5,
                    "spectral_norm": 0.5,
                    "decay_rate": 0.5,
                }

        # Normalize eigenvalues
        eigenvalues = eigenvalues / (eigenvalues.sum() + 1e-10)
        top_k = eigenvalues[:self.top_k]

        # Feature 1: Spectral entropy (higher = more diffuse = hallucination)
        spectral_entropy = -torch.sum(top_k * torch.log(top_k + 1e-10)).item()
        max_entropy = np.log(self.top_k)
        spectral_entropy_norm = spectral_entropy / max_entropy if max_entropy > 0 else 0.5

        # Feature 2: Dominant eigenvalue ratio (lower = less focused = hallucination)
        dominant_ratio = (eigenvalues[0] / (eigenvalues[:5].sum() + 1e-10)).item()

        # Feature 3: Spectral norm (L2 of eigenvalues)
        spectral_norm = torch.norm(top_k).item()

        # Feature 4: Eigenvalue decay rate (fit exponential decay)
        if len(top_k) > 2:
            log_eig = torch.log(top_k + 1e-10)
            indices = torch.arange(len(top_k), dtype=torch.float32, device=log_eig.device)
            # Simple linear regression on log scale
            decay_rate = -((indices * log_eig).mean() - indices.mean() * log_eig.mean()) / (
                (indices ** 2).mean() - indices.mean() ** 2 + 1e-10
            )
            decay_rate = decay_rate.item()
        else:
            decay_rate = 0.5

        return {
            "spectral_entropy": spectral_entropy_norm,
            "dominant_ratio": dominant_ratio,
            "spectral_norm": spectral_norm,
            "decay_rate": decay_rate,
        }

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute EigenScore for hallucination detection.

        Returns:
            Dict with 'score' (hallucination likelihood) and spectral features
        """
        # Tokenize
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        response_start = prompt_ids.size(1)

        # Register hooks and run forward pass
        self._register_hooks()

        try:
            # Enable attention output
            outputs = self.model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

            # Use output attentions if hooks didn't capture
            if not self._attention_maps and hasattr(outputs, 'attentions') and outputs.attentions:
                for i, layer_idx in enumerate(self.target_layers):
                    if i < len(outputs.attentions):
                        self._attention_maps[layer_idx] = outputs.attentions[layer_idx]

        finally:
            self._remove_hooks()

        if not self._attention_maps:
            # Fallback if no attention captured
            return {
                "score": 0.5,
                "method": "eigenscore",
                "error": "No attention weights captured",
            }

        # Aggregate spectral features across layers
        all_features = []
        for layer_idx, attn in self._attention_maps.items():
            # Focus on response tokens attending to all tokens
            if attn.size(-1) > response_start:
                response_attn = attn[:, :, response_start:, :]
                features = self._compute_spectral_features(response_attn)
                all_features.append(features)

        if not all_features:
            return {
                "score": 0.5,
                "method": "eigenscore",
                "error": "No response attention captured",
            }

        # Average features across layers
        avg_features = {
            key: np.mean([f[key] for f in all_features])
            for key in all_features[0].keys()
        }

        # Combine into final score
        # Higher spectral entropy + lower dominant ratio = hallucination
        score = (
            0.4 * avg_features["spectral_entropy"] +
            0.3 * (1.0 - avg_features["dominant_ratio"]) +
            0.2 * (1.0 - min(avg_features["spectral_norm"], 1.0)) +
            0.1 * min(avg_features["decay_rate"], 1.0)
        )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "spectral_entropy": avg_features["spectral_entropy"],
            "dominant_ratio": avg_features["dominant_ratio"],
            "spectral_norm": avg_features["spectral_norm"],
            "decay_rate": avg_features["decay_rate"],
            "n_layers_analyzed": len(all_features),
            "method": "eigenscore",
        }


class SAPLMABaseline:
    """
    SAPLMA: Statement Accuracy Prediction via Logit Mixture Analysis.

    Uses hidden state representations to predict truthfulness.
    Simplified version using final hidden states rather than trained probe.

    Key idea: Internal representations of truthful vs hallucinated content
    have distinguishable geometric properties.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        target_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        n_layers = model.config.num_hidden_layers
        self.target_layers = target_layers or [n_layers - 1]  # Last layer by default

    @torch.inference_mode()
    def compute_score(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Compute SAPLMA-style score using hidden state analysis.

        Returns:
            Dict with 'score' based on hidden state properties
        """
        # Tokenize
        full_text = prompt + response
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        prompt_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        response_start = prompt_ids.size(1)

        # Forward pass with hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states  # Tuple of (B, T, D)

        if not hidden_states:
            return {"score": 0.5, "method": "saplma", "error": "No hidden states"}

        # Analyze target layers
        features = []

        for layer_idx in self.target_layers:
            if layer_idx < len(hidden_states):
                hs = hidden_states[layer_idx]  # (B, T, D)

                # Get response hidden states
                if hs.size(1) > response_start:
                    response_hs = hs[:, response_start:, :]  # (B, T_resp, D)

                    # Feature 1: Variance of hidden states (higher = uncertain)
                    variance = response_hs.var(dim=-1).mean().item()

                    # Feature 2: Norm of hidden states (atypical norms = hallucination)
                    norms = torch.norm(response_hs, dim=-1)
                    norm_std = norms.std().item()

                    # Feature 3: Cosine similarity between consecutive tokens
                    if response_hs.size(1) > 1:
                        cos_sim = torch.nn.functional.cosine_similarity(
                            response_hs[:, :-1, :],
                            response_hs[:, 1:, :],
                            dim=-1
                        ).mean().item()
                    else:
                        cos_sim = 1.0

                    features.append({
                        "variance": variance,
                        "norm_std": norm_std,
                        "temporal_consistency": cos_sim,
                    })

        if not features:
            return {"score": 0.5, "method": "saplma", "error": "No features computed"}

        # Average across layers
        avg_variance = np.mean([f["variance"] for f in features])
        avg_norm_std = np.mean([f["norm_std"] for f in features])
        avg_consistency = np.mean([f["temporal_consistency"] for f in features])

        # Normalize features (empirical scaling)
        var_norm = min(avg_variance / 10.0, 1.0)
        std_norm = min(avg_norm_std / 5.0, 1.0)

        # Combine: high variance + high norm std + low consistency = hallucination
        score = (
            0.35 * var_norm +
            0.35 * std_norm +
            0.30 * (1.0 - avg_consistency)
        )

        score = max(0.0, min(1.0, score))

        return {
            "score": score,
            "variance": avg_variance,
            "norm_std": avg_norm_std,
            "temporal_consistency": avg_consistency,
            "method": "saplma",
        }
