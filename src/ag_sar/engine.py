"""
AGSAR Engine - Main Orchestrator.

This is the primary user-facing interface for AG-SAR uncertainty quantification.
It coordinates the modeling, measures, and ops layers to provide a clean API.

Example:
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> config = AGSARConfig(semantic_layers=4, uncertainty_metric='gse')
    >>> agsar = AGSAR(model, tokenizer, config)
    >>> score = agsar.compute_uncertainty(prompt, response)
    >>> is_hallucination, confidence, details = agsar.detect_hallucination(prompt, response)
"""

from typing import Optional, Tuple, Dict, Any, List
import json
import torch
import torch.nn as nn

from .config import AGSARConfig
from .modeling import ModelAdapter
from .measures import (
    compute_authority_score,
    compute_register_mask,
    compute_mlp_divergence,
    compute_sink_aware_centrality,
    aggregate_value_norms,
    compute_hebbian_weights,
    compute_token_entropy,
    compute_graph_shifted_entropy,
    detect_hallucination,
    compute_bounded_surprisal,
    compute_manifold_consistent_spectral_surprisal,
)
from .ops import EMAState
from .utils import enable_tf32, get_model_dtype, get_model_device


class AGSAR:
    """
    AG-SAR Uncertainty Quantification Engine.

    Provides zero-latency hallucination detection by analyzing internal
    attention graph structure without external semantic models.

    Supports multiple uncertainty metrics:
        - GSE: Graph-Shifted Entropy (default)
        - GSS: Graph-Shifted Surprisal (for forced responses)
        - MC-SS: Manifold-Consistent Spectral Surprisal
        - v31/authority: Authority Flow with optional roughness penalty

    Example:
        >>> agsar = AGSAR(model, tokenizer)
        >>> score = agsar.compute_uncertainty("What is the capital of France?", "Paris")
        >>> print(f"Uncertainty: {score:.3f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[AGSARConfig] = None,
    ):
        """
        Initialize AG-SAR engine.

        Args:
            model: Language model (GPT-2, Llama, etc.)
            tokenizer: Tokenizer for the model
            config: Configuration (default: AGSARConfig())
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AGSARConfig()

        self.dtype = get_model_dtype(model)
        self.device = get_model_device(model)

        # Determine semantic layers to analyze
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            num_layers = model.config.num_hidden_layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            num_layers = len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        else:
            num_layers = 12  # Default

        start_layer = max(0, num_layers - self.config.semantic_layers)
        self._semantic_layer_indices = list(range(start_layer, num_layers))

        # Initialize model adapter for attention extraction
        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
        )
        self._adapter.register()

        # Load SGSS head scores if configured
        self._head_scores = None
        if self.config.use_spectral_steering and self.config.head_scores_path:
            self._load_head_scores()

        # v3.1 streaming state
        self._ema_state: Optional[EMAState] = None
        self._authority_history: Optional[torch.Tensor] = None
        self._prompt_length: int = 0
        self._value_history: Optional[torch.Tensor] = None
        self._h_attn_history: Optional[torch.Tensor] = None

    def _load_head_scores(self):
        """Load pre-calibrated head scores for SGSS."""
        try:
            with open(self.config.head_scores_path) as f:
                data = json.load(f)
            scores = torch.tensor(data['head_z_scores'], dtype=self.dtype, device=self.device)
            self._head_scores = scores
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load head scores: {e}")
            self._head_scores = None

    def _tokenize(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize prompt and response, returning combined sequence."""
        prompt_enc = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
        response_enc = self.tokenizer(response, return_tensors='pt', add_special_tokens=False)

        prompt_ids = prompt_enc['input_ids']
        response_ids = response_enc['input_ids']

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.ones_like(input_ids)

        response_start = prompt_ids.size(1)

        return input_ids, attention_mask, response_start

    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        return_details: bool = False
    ) -> Any:
        """
        Compute uncertainty score for a prompt-response pair.

        Args:
            prompt: Input prompt
            response: Generated response
            return_details: Return dict with intermediate values

        Returns:
            If return_details=False: float uncertainty score
            If return_details=True: dict with score, entropy, relevance, etc.
        """
        # Route to appropriate metric
        metric = self.config.uncertainty_metric

        if metric in ('v31', 'authority'):
            return self.compute_uncertainty_v31(prompt, response, return_details)

        # Tokenize
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Extract attention data
        Q_stack, K_stack, value_norms_dict, model_output = self._adapter.extract(
            input_ids, attention_mask
        )

        # Aggregate value norms
        value_norms = aggregate_value_norms(
            value_norms_dict, self.config.semantic_layers
        )

        logits = model_output.logits

        # Compute surprisal for SGSS if enabled
        surprisal = None
        if self.config.use_spectral_steering:
            from .measures.spectral import compute_token_surprisal
            surprisal = compute_token_surprisal(logits, input_ids, attention_mask)

        # Compute centrality
        hebbian_weights = None
        if metric == 'mcss':
            hebbian_weights = compute_hebbian_weights(
                K_stack, response_start, self.config.mcss_hebbian_tau, attention_mask
            )

        relevance, centrality, _ = compute_sink_aware_centrality(
            value_norms=value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
            attention_mask=attention_mask,
            num_iterations=self.config.power_iteration_steps,
            tol=self.config.power_iteration_tol,
            residual_weight=self.config.residual_weight,
            sink_token_count=self.config.sink_token_count,
            hebbian_weights=hebbian_weights,
            use_hebbian=(metric == 'mcss'),
            surprisal=surprisal,
            head_scores=self._head_scores,
            steering_alpha=self.config.steering_alpha,
            steering_beta=self.config.steering_beta,
            response_start=response_start,
        )

        # Compute metric-specific score
        if metric == 'gse':
            entropy = compute_token_entropy(logits, attention_mask)
            score = compute_graph_shifted_entropy(entropy, relevance, attention_mask)
            uncertainty = score.item()
        elif metric == 'gss':
            from .measures.spectral import compute_token_surprisal, compute_graph_shifted_surprisal
            token_surprisal = compute_token_surprisal(logits, input_ids, attention_mask)
            score = compute_graph_shifted_surprisal(token_surprisal, relevance, attention_mask)
            uncertainty = score.item()
        elif metric == 'mcss':
            bounded_surprisal = compute_bounded_surprisal(
                logits, input_ids, self.config.mcss_beta, attention_mask
            )
            score = compute_manifold_consistent_spectral_surprisal(
                bounded_surprisal, centrality, attention_mask, self.config.mcss_penalty_weight
            )
            uncertainty = score.item()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if return_details:
            return {
                'score': uncertainty,
                'metric': metric,
                'response_start': response_start,
                'sequence_length': input_ids.size(1),
            }

        return uncertainty

    def compute_uncertainty_v31(
        self,
        prompt: str,
        response: str,
        return_details: bool = False
    ) -> Any:
        """
        Compute v3.1 Authority Flow uncertainty.

        Uses Register Filter + Recursive Authority Flow + Optional Roughness.
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        Q_stack, K_stack, value_norms_dict, model_output = self._adapter.extract(
            input_ids, attention_mask
        )

        # Get value states and attention outputs for roughness
        value_states = self._adapter.capture.value_states
        attn_outputs = self._adapter.capture.attn_outputs
        block_outputs = self._adapter.capture.block_outputs
        attention_weights = self._adapter.capture.attention_weights

        # Compute register mask
        last_layer = self._semantic_layer_indices[-1]
        v_states = value_states.get(last_layer)

        register_mask = None
        if self.config.enable_register_filter and v_states is not None:
            register_mask, self._ema_state = compute_register_mask(
                v_states,
                self._ema_state,
                self.config.kurtosis_threshold,
                self.config.sink_token_count,
                self.config.ema_decay,
            )

        # Compute authority flow
        attn = attention_weights.get(last_layer)
        if attn is None:
            # Fall back to GSE if no attention weights
            return self.compute_uncertainty(prompt, response, return_details)

        authority = compute_authority_score(
            attn, response_start, register_mask,
            roughness=None,  # Applied separately
            lambda_roughness=0,
            attention_mask=attention_mask,
            use_vectorized=True,
        )

        # Compute roughness penalty
        roughness = None
        if self.config.enable_spectral_roughness:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                roughness = compute_mlp_divergence(h_attn, h_block, attention_mask)

        # Apply roughness penalty
        if roughness is not None and self.config.lambda_roughness > 0:
            authority = authority / (1.0 + self.config.lambda_roughness * roughness)
            authority = authority.clamp(0.0, 1.0)

        # Hallucination score = 1 - authority (over response tokens)
        response_authority = authority[:, response_start:].mean().item()
        uncertainty = 1.0 - response_authority

        if return_details:
            return {
                'score': uncertainty,
                'authority': response_authority,
                'metric': 'v31',
                'response_start': response_start,
            }

        return uncertainty

    def detect_hallucination(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect hallucination with binary classification.

        Args:
            prompt: Input prompt
            response: Generated response
            threshold: Override default threshold

        Returns:
            is_hallucination: True if likely hallucinating
            confidence: How confident the detection is
            details: Dict with intermediate values
        """
        threshold = threshold or self.config.hallucination_threshold
        details = self.compute_uncertainty(prompt, response, return_details=True)

        score = details['score']
        is_hall, conf = detect_hallucination(torch.tensor([score]), threshold)

        return is_hall.item(), conf.item(), details

    def reset(self):
        """Reset streaming state (call before new prompt-response pair)."""
        self._ema_state = None
        self._authority_history = None
        self._prompt_length = 0
        self._value_history = None
        self._h_attn_history = None

    def cleanup(self):
        """Release resources."""
        self._adapter.cleanup()
        self.reset()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
