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
    compute_lid_penalty,
    compute_lid_with_calibration,
    compute_lid_fast,
    compute_manifold_signature,
    compute_adaptive_lid_penalty,
    ManifoldSignature,
    # v6.0 Spectral-Structural
    compute_laplacian_entropy_per_token,
    compute_layer_divergence,
    # v7.0 Context-Dependent Gating
    compute_v7_gated_authority,
    # v8.0 Semantic Dispersion
    compute_v8_semantic_authority,
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

        # v6.0 Spectral: Compute early and late layer indices for DoLa
        if self.config.enable_spectral:
            self._early_layer = int(num_layers * self.config.early_layer_ratio)
            self._late_layer = int(num_layers * self.config.late_layer_ratio)
            # Ensure early layer is captured
            if self._early_layer not in self._semantic_layer_indices:
                self._semantic_layer_indices = [self._early_layer] + self._semantic_layer_indices
        else:
            self._early_layer = None
            self._late_layer = self._semantic_layer_indices[-1] if self._semantic_layer_indices else None

        # Initialize model adapter for attention extraction
        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
        )
        self._adapter.register()

        # v8.0 Semantic Dispersion: Capture output embedding matrix
        # This is the "unembedding" matrix that maps hidden states to vocab
        self._embed_matrix = None
        if self.config.enable_semantic_dispersion:
            try:
                output_embeddings = model.get_output_embeddings()
                if output_embeddings is not None:
                    self._embed_matrix = output_embeddings.weight.detach()
            except Exception:
                # Fallback: some models don't have get_output_embeddings
                if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                    self._embed_matrix = model.lm_head.weight.detach()

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

        # v7.0/v8.0 Context-Dependent Gating: Unified RAG + Free Gen
        # v7.0: Uses stability gate + raw confidence
        # v8.0: Uses stability gate + semantic dispersion (consistency over confidence)
        if self.config.enable_v7_gating:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)
                logits = model_output.logits.to(compute_device)

                # v8.0: Semantic Dispersion (Consistency over Confidence)
                if self.config.enable_semantic_dispersion and self._embed_matrix is not None:
                    embed_matrix = self._embed_matrix.to(compute_device)
                    authority = compute_v8_semantic_authority(
                        attention_weights=attn,
                        prompt_length=response_start,
                        h_attn=h_attn,
                        h_block=h_block,
                        logits=logits,
                        embed_matrix=embed_matrix,
                        register_mask=register_mask,
                        attention_mask=attention_mask,
                        stability_sensitivity=self.config.stability_sensitivity,
                        parametric_weight=self.config.parametric_weight,
                        dispersion_k=self.config.dispersion_k,
                        dispersion_sensitivity=self.config.dispersion_sensitivity,
                    )
                else:
                    # v7.0: Raw confidence
                    authority = compute_v7_gated_authority(
                        attention_weights=attn,
                        prompt_length=response_start,
                        h_attn=h_attn,
                        h_block=h_block,
                        logits=logits,
                        register_mask=register_mask,
                        attention_mask=attention_mask,
                        stability_sensitivity=self.config.stability_sensitivity,
                        parametric_weight=self.config.parametric_weight,
                    )
            else:
                # Fallback to regular authority if hidden states not captured
                authority = compute_authority_score(
                    attn, response_start, register_mask,
                    roughness=None,
                    lambda_roughness=0,
                    attention_mask=attention_mask,
                    use_vectorized=True,
                )
        else:
            # Standard v3.1 Authority Flow
            authority = compute_authority_score(
                attn, response_start, register_mask,
                roughness=None,  # Applied separately
                lambda_roughness=0,
                attention_mask=attention_mask,
                use_vectorized=True,
                subject_boost=self.config.subject_boost,
                subject_token_count=self.config.subject_token_count,
            )

        # Compute MLP divergence penalty (v3.2) - only if v7 not enabled
        # v7.0 incorporates divergence into the gating mechanism
        roughness = None
        if self.config.enable_spectral_roughness and not self.config.enable_v7_gating:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                # Move to compute device for multi-GPU compatibility
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)
                roughness = compute_mlp_divergence(h_attn, h_block, attention_mask)

                # v4.0 Enhancement: Entropy-Weighted Divergence
                # Amplifies divergence when model is uncertain (high entropy)
                # Targets WikiBio-style free generation where guesses have high entropy
                if self.config.entropy_beta > 0:
                    logits = model_output.logits.to(compute_device)
                    # Compute per-token entropy: H(p) = -sum(p * log(p))
                    probs = torch.softmax(logits, dim=-1)
                    log_probs = torch.log(probs + 1e-10)
                    token_entropy = -torch.sum(probs * log_probs, dim=-1)  # [B, S]
                    # Normalize entropy by log(vocab_size) to get [0, 1] range
                    vocab_size = logits.size(-1)
                    normalized_entropy = token_entropy / torch.log(torch.tensor(vocab_size, device=compute_device, dtype=logits.dtype))
                    # Weight roughness: high entropy = amplify divergence penalty
                    entropy_weight = 1.0 + self.config.entropy_beta * normalized_entropy
                    roughness = roughness * entropy_weight

        # Apply roughness penalty
        if roughness is not None and self.config.lambda_roughness > 0:
            authority = authority / (1.0 + self.config.lambda_roughness * roughness)
            authority = authority.clamp(0.0, 1.0)

        # v5.0/v5.1 Enhancement: Local Intrinsic Dimension (LID) penalty
        # Detects confabulation by measuring manifold complexity
        # High LID = disordered manifold = hallucination
        if self.config.enable_lid:
            h_block = block_outputs.get(last_layer)
            if h_block is not None:
                h_block = h_block.to(compute_device)

                if self.config.lid_calibration == "prompt":
                    # v5.1: Prompt-anchored calibration (zero-shot)
                    # Use the prompt's manifold as the baseline
                    lid_penalty, _ = compute_lid_with_calibration(
                        h_block,
                        prompt_length=response_start,
                        k=self.config.lid_k,
                        sensitivity=self.config.lid_weight,
                    )
                else:
                    # Legacy: Fixed calibration (deprecated)
                    lid_penalty = compute_lid_penalty(
                        h_block,
                        k=self.config.lid_k,
                        lid_mean=self.config.lid_mean,
                        lid_std=self.config.lid_std,
                    )

                # Apply LID penalty: high LID crushes authority
                # Use smoother application: authority *= (1 - penalty)
                authority = authority * (1.0 - lid_penalty)
                authority = authority.clamp(0.0, 1.0)

        # v6.0 Enhancement: Spectral-Structural penalty
        # Combines Laplacian Spectral Entropy (attention graph structure) with
        # Layer-Contrastive Divergence (DoLa-style depth dynamics)
        if self.config.enable_spectral:
            # 1. Laplacian Spectral Entropy: measures attention graph structure
            # High entropy = diffuse/random graph = hallucination
            # Low entropy = star/clique graph = structured reasoning (fact)
            last_layer = self._semantic_layer_indices[-1]
            attn_weights = self._adapter.capture.attention_weights.get(last_layer)
            if attn_weights is not None:
                attn_weights = attn_weights.to(compute_device)
                laplacian_entropy = compute_laplacian_entropy_per_token(
                    attn_weights,
                    window_size=self.config.spectral_window,
                    normalize=True,
                )

                # 2. Layer-Contrastive Divergence (DoLa-style)
                # High divergence = late layer corrected early layer = fact retrieval
                # Low divergence = late layer followed early layer = autocomplete/hallucination
                layer_divergence = None
                if self._early_layer is not None and self._late_layer is not None:
                    h_early = block_outputs.get(self._early_layer)
                    h_late = block_outputs.get(self._late_layer)

                    if h_early is not None and h_late is not None:
                        h_early = h_early.to(compute_device)
                        h_late = h_late.to(compute_device)

                        # Project hidden states to logits via LM head
                        lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None
                        if lm_head is not None:
                            with torch.no_grad():
                                logits_early = lm_head(h_early)
                                logits_late = lm_head(h_late)
                            layer_divergence = compute_layer_divergence(logits_early, logits_late)

                # Combine: score = alpha * entropy - beta * divergence
                # High entropy (bad) and low divergence (bad) = hallucination
                spectral_score = self.config.spectral_alpha * laplacian_entropy
                if layer_divergence is not None:
                    spectral_score = spectral_score - self.config.spectral_beta * layer_divergence

                # Convert to penalty (sigmoid to [0, 1])
                # Positive spectral_score = likely hallucination = reduce authority
                spectral_penalty = torch.sigmoid(spectral_score)

                # Apply penalty: authority *= (1 - penalty)
                authority = authority * (1.0 - spectral_penalty)
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
