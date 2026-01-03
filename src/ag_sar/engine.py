"""
AGSAR Engine - Main Orchestrator.

This is the primary user-facing interface for AG-SAR uncertainty quantification.
It coordinates the modeling, measures, and ops layers to provide a clean API.

Example:
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> config = AGSARConfig(uncertainty_metric='v31')
    >>> agsar = AGSAR(model, tokenizer, config)
    >>> score = agsar.compute_uncertainty(prompt, response)
    >>> is_hallucination, confidence, details = agsar.detect_hallucination(prompt, response)
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn

from .config import AGSARConfig
from .modeling import ModelAdapter
from .measures import (
    compute_authority_score,
    compute_register_mask,
    compute_mlp_divergence,
)
from .ops import EMAState
from .utils import get_model_dtype, get_model_device


class AGSAR:
    """
    AG-SAR Uncertainty Quantification Engine.

    Provides zero-latency hallucination detection by analyzing internal
    attention graph structure without external semantic models.

    Uses v3.1/v3.2 Authority Flow + MLP Divergence architecture:
        1. Register Filter: Kurtosis-based token classification
        2. Authority Flow: Recursive prompt recharge
        3. MLP Divergence: Detects when MLP overrides attention

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

        # For multi-GPU models (device_map="balanced"), determine compute device
        # Use the device where the output layer resides for tensor aggregation
        self._compute_device = self._get_compute_device(model)

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

        # v3.1 streaming state
        self._ema_state: Optional[EMAState] = None

    def _get_compute_device(self, model: nn.Module) -> torch.device:
        """
        Determine the compute device for tensor aggregation.

        For multi-GPU models using Accelerate's device_map, finds the device
        where the LM head (output layer) resides. This is where logits are
        computed and is the natural aggregation point.

        Returns:
            torch.device: The compute device for tensor operations
        """
        # Check for Accelerate's device map (set by device_map="balanced" etc.)
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            # Find the output layer device (lm_head for causal LM)
            for key in ['lm_head', 'embed_out', 'output']:
                if key in device_map:
                    device_id = device_map[key]
                    if isinstance(device_id, int):
                        return torch.device(f"cuda:{device_id}")
                    elif isinstance(device_id, str) and device_id != 'cpu':
                        return torch.device(device_id)
            # Fallback: use the last layer's device
            layer_keys = [k for k in device_map.keys() if 'layer' in k.lower()]
            if layer_keys:
                last_layer = sorted(layer_keys)[-1]
                device_id = device_map[last_layer]
                if isinstance(device_id, int):
                    return torch.device(f"cuda:{device_id}")

        # Single GPU or CPU: use model's device
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return self.device

    def _tokenize(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize prompt and response, returning combined sequence."""
        # Input validation
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not response or not response.strip():
            raise ValueError("response must be a non-empty string")

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
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute uncertainty score for a prompt-response pair.

        Uses v3.1 Authority Flow with optional MLP Divergence penalty.

        Args:
            prompt: Input prompt
            response: Generated response
            return_details: Return dict with intermediate values

        Returns:
            If return_details=False: float uncertainty score (0=confident, 1=uncertain)
            If return_details=True: dict with score, authority, metric, etc.
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)

        # Use pre-computed device for multi-GPU tensor aggregation
        compute_device = self._compute_device

        input_ids = input_ids.to(compute_device)
        attention_mask = attention_mask.to(compute_device)

        # Extract attention data
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

        # Move captured tensors to compute device for multi-GPU compatibility
        if v_states is not None:
            v_states = v_states.to(compute_device)

        register_mask = None
        if self.config.enable_register_filter and v_states is not None:
            register_mask, self._ema_state = compute_register_mask(
                v_states,
                self._ema_state,
                self.config.kurtosis_threshold,
                self.config.sink_token_count,
                self.config.ema_decay,
            )
            # Move register_mask to compute device for multi-GPU compatibility
            if register_mask is not None:
                register_mask = register_mask.to(compute_device)

        # Compute authority flow
        attn = attention_weights.get(last_layer)
        if attn is None:
            raise RuntimeError(
                "Attention weights not captured. This may indicate hook registration "
                "failed or architecture mismatch. Check that the model architecture "
                "is supported (GPT-2, Llama, Mistral, Qwen)."
            )
        # Move attention to compute device for multi-GPU compatibility
        attn = attn.to(compute_device)

        authority = compute_authority_score(
            attn, response_start, register_mask,
            roughness=None,  # Applied separately
            lambda_roughness=0,
            attention_mask=attention_mask,
            use_vectorized=True,
        )

        # Compute MLP divergence penalty
        roughness = None
        if self.config.enable_spectral_roughness:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                # Move to compute device for multi-GPU compatibility
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)
                roughness = compute_mlp_divergence(h_attn, h_block, attention_mask)

        # Apply roughness penalty
        if roughness is not None and self.config.lambda_roughness > 0:
            authority = authority / (1.0 + self.config.lambda_roughness * roughness)
            authority = authority.clamp(0.0, 1.0)

        # Hallucination score = 1 - authority (over response tokens)
        response_authority = authority[:, response_start:].mean().item()
        uncertainty = 1.0 - response_authority

        if return_details:
            # Compute model confidence (mean probability of response tokens)
            # Move logits to compute device for multi-GPU compatibility
            logits = model_output.logits.to(compute_device)
            log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
            response_tokens = input_ids[:, response_start:]
            token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
            model_confidence = token_log_probs.exp().mean().item()

            return {
                'score': uncertainty,
                'authority': response_authority,
                'model_confidence': model_confidence,
                'metric': 'v31',
                'response_start': response_start,
                'sequence_length': input_ids.size(1),
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
            confidence: How confident the detection is (distance from threshold)
            details: Dict with intermediate values
        """
        threshold = threshold or self.config.hallucination_threshold

        # Validate threshold bounds
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        details = self.compute_uncertainty(prompt, response, return_details=True)

        score = details['score']
        is_hallucination = score > threshold
        confidence = abs(score - threshold)  # Distance from decision boundary

        return is_hallucination, confidence, details

    def reset(self):
        """Reset streaming state (call before new prompt-response pair)."""
        self._ema_state = None

    def cleanup(self):
        """Release resources."""
        self._adapter.cleanup()
        self.reset()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
