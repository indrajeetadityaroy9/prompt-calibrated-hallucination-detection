"""
AG-SAR Engine Module.

This module implements the AGSAR class, the primary user-facing interface for
uncertainty quantification. The engine orchestrates attention extraction,
authority flow computation, unified gating, and semantic dispersion into a
single forward pass.

Mechanism:
    Given a (prompt, response) pair, the engine:
    1. Tokenizes and concatenates prompt + response
    2. Runs a single forward pass with attention hooks registered
    3. Extracts attention weights from configured semantic layers
    4. Computes authority flow from prompt tokens to response tokens
    5. Applies unified gating to balance context vs parametric signals
    6. (Optionally) computes semantic dispersion over top-k predictions
    7. Aggregates per-token scores into sequence-level uncertainty

Pipeline Position:
    AGSAR is the top-level orchestrator. It consumes AGSARConfig and a
    pre-loaded model/tokenizer, then provides compute_uncertainty() for
    evaluation. Internal state is managed via ModelAdapter hooks.

Assumptions:
    - Model is frozen (eval mode, no gradient computation)
    - Tokenizer is compatible with model vocabulary
    - Prompt contains source context for faithfulness verification
    - Response has already been generated (not streaming)

Thread Safety:
    AGSAR instances are NOT thread-safe. Each thread should create its own
    instance. The ModelAdapter registers global hooks that modify model state.

Resource Management:
    Call cleanup() when done to remove hooks and free cached tensors.
    The destructor attempts cleanup but explicit calls are recommended.
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn

from .config import AGSARConfig
from .modeling import ModelAdapter
from .measures import (
    compute_authority_score,
    compute_gated_authority,
    compute_semantic_authority,
    compute_semantic_dispersion,
)
from .ops import compute_stability_gate
from .utils import get_model_dtype, get_model_device


class AGSAR:
    """
    AG-SAR Uncertainty Quantification Engine.

    This class provides the primary interface for detecting extrinsic
    hallucinations (context violations) in LLM-generated text. It combines
    Authority Flow, Unified Gating, and Semantic Dispersion into a single
    forward pass with O(N) memory complexity.

    Detection Scope:
        AG-SAR detects when generated text is UNFAITHFUL to provided context,
        NOT when it contradicts world knowledge. Optimal for:
        - RAG faithfulness monitoring
        - Summarization accuracy
        - QA with context

    Mechanism:
        For each response token t, computes uncertainty U(t) = 1 - A(t) where:

            A(t) = G(t) × Flow(t) + (1 - G(t)) × Trust(t) × w_param

        - Flow(t): Authority from attention to prompt (information provenance)
        - G(t): Stability gate ∈ [0,1] (context reliance indicator)
        - Trust(t): 1 - dispersion (semantic consistency of top-k)
        - w_param: Parametric weight (trust in model confidence)

    Attributes:
        model: The frozen language model (GPT-2, Llama, Mistral, Qwen).
        tokenizer: Tokenizer compatible with model vocabulary.
        config: AGSARConfig with mechanism parameters.
        dtype: Model's compute dtype (bfloat16, float16, float32).
        device: Primary device for model parameters.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> agsar = AGSAR(model, tokenizer)
        >>> uncertainty = agsar.compute_uncertainty(
        ...     prompt="The capital of France is",
        ...     response=" Paris."
        ... )
        >>> print(f"Uncertainty: {uncertainty:.3f}")
        Uncertainty: 0.142
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[AGSARConfig] = None,
    ):
        """
        Initialize the AG-SAR engine.

        Registers attention hooks on the model to capture intermediate states
        during forward passes. The model should be in eval mode with gradients
        disabled for inference.

        Args:
            model: Pre-loaded language model. Must have standard attention
                interface (attention weights accessible via hooks).
            tokenizer: Tokenizer for the model. Must have encode/decode methods
                and pad_token defined (uses eos_token if None).
            config: Configuration parameters. If None, uses AGSARConfig()
                defaults (unified gating + semantic dispersion enabled).

        Raises:
            RuntimeError: If embedding matrix extraction fails and semantic
                dispersion is enabled (auto-disables with warning instead).

        Side Effects:
            - Registers forward hooks on model attention layers
            - Caches embedding matrix reference for semantic dispersion
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AGSARConfig()

        self.dtype = get_model_dtype(model)
        self.device = get_model_device(model)

        # For multi-GPU models, determine aggregation device (where lm_head resides)
        self._compute_device = self._get_compute_device(model)

        # Detect model layer count for semantic layer selection
        num_layers = self._detect_num_layers(model)
        self._num_layers = num_layers

        # Select final N layers for analysis (late-layer focus is empirically optimal)
        start_layer = max(0, num_layers - self.config.semantic_layers)
        self._semantic_layer_indices = list(range(start_layer, num_layers))

        # Initialize model adapter with attention hooks
        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
        )
        self._adapter.register()

        # Cache embedding matrix for semantic dispersion computation
        self._embed_matrix = None
        if self.config.enable_semantic_dispersion:
            self._embed_matrix = self._extract_embedding_matrix(model)

    def _detect_num_layers(self, model: nn.Module) -> int:
        """
        Detect the number of transformer layers in the model.

        Handles different architecture conventions (HuggingFace config,
        GPT-2 style transformer.h, Llama style model.layers).

        Args:
            model: Language model to inspect.

        Returns:
            int: Number of transformer layers. Defaults to 12 if undetectable.
        """
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            return model.config.num_hidden_layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)
        return 12  # Conservative default

    def _extract_embedding_matrix(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Extract the output embedding (unembedding) matrix for semantic dispersion.

        The embedding matrix maps hidden states to vocabulary logits. Required
        for computing semantic similarity between top-k predictions.

        Args:
            model: Language model with lm_head or get_output_embeddings().

        Returns:
            Tensor of shape (vocab_size, hidden_dim), or None if extraction fails.

        Side Effects:
            - Emits RuntimeWarning if fallback to lm_head is used
            - Disables semantic_dispersion in config if extraction fails
        """
        import warnings

        try:
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings.weight.detach()
        except Exception as e:
            # Fallback for models without get_output_embeddings
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                warnings.warn(
                    f"Using lm_head fallback for embedding matrix (get_output_embeddings failed: {e})",
                    RuntimeWarning,
                )
                return model.lm_head.weight.detach()

        # Extraction failed - disable semantic dispersion
        warnings.warn(
            "Could not extract embedding matrix for semantic dispersion. "
            "Disabling semantic_dispersion feature.",
            RuntimeWarning,
        )
        self.config.enable_semantic_dispersion = False
        return None

    def _get_compute_device(self, model: nn.Module) -> torch.device:
        """
        Determine the device for tensor aggregation in multi-GPU setups.

        For models distributed with Accelerate's device_map, finds the device
        where the output layer (lm_head) resides. This is the natural
        aggregation point for computing final scores.

        Args:
            model: Language model, potentially distributed across GPUs.

        Returns:
            torch.device: Device for compute operations. Defaults to cuda:0
                or CPU if CUDA unavailable.
        """
        # Check for Accelerate's device map
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map

            # Find output layer device
            for key in ['lm_head', 'embed_out', 'output']:
                if key in device_map:
                    device_id = device_map[key]
                    if isinstance(device_id, int):
                        return torch.device(f"cuda:{device_id}")
                    elif isinstance(device_id, str) and device_id != 'cpu':
                        return torch.device(device_id)

            # Fallback to last layer's device
            layer_keys = [k for k in device_map.keys() if 'layer' in k.lower()]
            if layer_keys:
                last_layer = sorted(layer_keys)[-1]
                device_id = device_map[last_layer]
                if isinstance(device_id, int):
                    return torch.device(f"cuda:{device_id}")

        # Single GPU or CPU fallback
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return self.device

    def _tokenize(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenize prompt and response into concatenated sequence.

        Encodes prompt with special tokens, response without, then concatenates.
        Returns the boundary index for separating prompt from response tokens.

        Args:
            prompt: Source context/question text. Must be non-empty.
            response: Generated text to evaluate. Must be non-empty.

        Returns:
            Tuple containing:
                - input_ids: Shape (1, seq_len) token IDs
                - attention_mask: Shape (1, seq_len) attention mask (all 1s)
                - response_start: Index where response tokens begin

        Raises:
            ValueError: If prompt or response is empty or whitespace-only.
        """
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

        Executes the full AG-SAR pipeline: tokenization, forward pass with
        attention extraction, authority flow, unified gating, and semantic
        dispersion. Returns a scalar uncertainty or detailed breakdown.

        Args:
            prompt: Source context/question. Should contain the information
                against which the response will be verified.
            response: Generated text to evaluate for faithfulness.
            return_details: If True, return dict with component scores.

        Returns:
            If return_details=False:
                float: Uncertainty score in [0, 1]. Higher = more likely
                    to be a context violation.

            If return_details=True:
                dict with keys:
                    - score: float, uncertainty score
                    - authority: float, aggregated authority
                    - model_confidence: float, mean token probability
                    - gate: float, mean stability gate (if gating enabled)
                    - dispersion: float, mean dispersion (if dispersion enabled)
                    - response_start: int, token index where response begins
                    - sequence_length: int, total sequence length

        Raises:
            ValueError: If prompt or response is empty.
            RuntimeError: If attention weights not captured (architecture mismatch).

        Example:
            >>> score = agsar.compute_uncertainty(
            ...     "The Eiffel Tower is located in Paris.",
            ...     "The Eiffel Tower is in London."
            ... )
            >>> print(f"Uncertainty: {score:.3f}")  # High due to contradiction
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)

        compute_device = self._compute_device
        input_ids = input_ids.to(compute_device)
        attention_mask = attention_mask.to(compute_device)

        # Forward pass with attention extraction
        Q_stack, K_stack, value_norms_dict, model_output = self._adapter.extract(
            input_ids, attention_mask
        )

        # Retrieve captured intermediate states
        attn_outputs = self._adapter.capture.attn_outputs
        block_outputs = self._adapter.capture.block_outputs
        attention_weights = self._adapter.capture.attention_weights

        # Get attention from final semantic layer
        last_layer = self._semantic_layer_indices[-1]
        attn = attention_weights.get(last_layer)
        if attn is None:
            raise RuntimeError(
                "Attention weights not captured. This may indicate hook registration "
                "failed or architecture mismatch. Check that the model architecture "
                "is supported (GPT-2, Llama, Mistral, Qwen)."
            )
        attn = attn.to(compute_device)

        # Compute authority based on configuration
        if self.config.enable_unified_gating:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)
                logits = model_output.logits.to(compute_device)

                if self.config.enable_semantic_dispersion and self._embed_matrix is not None:
                    # Full mechanism: Authority + Gating + Dispersion
                    embed_matrix = self._embed_matrix.to(compute_device)
                    authority = compute_semantic_authority(
                        attention_weights=attn,
                        prompt_length=response_start,
                        h_attn=h_attn,
                        h_block=h_block,
                        logits=logits,
                        embed_matrix=embed_matrix,
                        attention_mask=attention_mask,
                        stability_sensitivity=self.config.stability_sensitivity,
                        parametric_weight=self.config.parametric_weight,
                        dispersion_k=self.config.dispersion_k,
                        dispersion_sensitivity=self.config.dispersion_sensitivity,
                        dispersion_method=self.config.dispersion_method,
                        nucleus_top_p=self.config.nucleus_top_p,
                    )
                else:
                    # Gating without dispersion (raw confidence)
                    authority = compute_gated_authority(
                        attention_weights=attn,
                        prompt_length=response_start,
                        h_attn=h_attn,
                        h_block=h_block,
                        logits=logits,
                        attention_mask=attention_mask,
                        stability_sensitivity=self.config.stability_sensitivity,
                        parametric_weight=self.config.parametric_weight,
                    )
            else:
                # Fallback if hidden states unavailable
                authority = compute_authority_score(
                    attn, response_start,
                    attention_mask=attention_mask,
                    use_vectorized=True,
                )
        else:
            # Ablation: Pure authority flow without gating
            authority = compute_authority_score(
                attn, response_start,
                attention_mask=attention_mask,
                use_vectorized=True,
            )

        # Aggregate per-token authority into sequence score
        response_authority_tokens = authority[:, response_start:]
        response_authority = self._aggregate_authority(
            response_authority_tokens, input_ids, response_start, model_output
        )

        uncertainty = 1.0 - response_authority

        if return_details:
            return self._build_details_dict(
                uncertainty, response_authority, input_ids, response_start,
                model_output, attn_outputs, block_outputs, last_layer, compute_device
            )

        return uncertainty

    def _aggregate_authority(
        self,
        authority_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        response_start: int,
        model_output
    ) -> float:
        """
        Aggregate per-token authority scores into sequence-level score.

        Applies the configured aggregation method (mean, min, percentile, or
        importance-weighted) to reduce (B, S_response) tensor to scalar.

        Args:
            authority_tokens: Shape (B, S_response) per-token authority scores.
            input_ids: Shape (B, S) full input token IDs (for importance weighting).
            response_start: Index where response begins (for importance weighting).
            model_output: Model forward pass output with logits (for importance weighting).

        Returns:
            float: Aggregated authority score in [0, 1].
        """
        method = self.config.aggregation_method

        if method == "mean":
            return authority_tokens.mean().item()
        elif method == "min":
            return authority_tokens.min().item()
        elif method == "percentile_10":
            return torch.quantile(authority_tokens.flatten(), 0.10).item()
        elif method == "percentile_25":
            return torch.quantile(authority_tokens.flatten(), 0.25).item()
        elif method == "importance_weighted":
            # Weight by self-information: rare tokens count more
            # Rationale: Errors on proper nouns/dates are more severe
            compute_device = authority_tokens.device
            logits = model_output.logits.to(compute_device)
            probs = torch.softmax(logits[:, response_start-1:-1, :], dim=-1)
            response_ids = input_ids[:, response_start:]
            token_probs = probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)
            self_info = -torch.log(token_probs + 1e-10)
            importance_weights = self_info / (self_info.sum(dim=-1, keepdim=True) + 1e-10)
            return (authority_tokens * importance_weights).sum().item()
        else:
            return authority_tokens.mean().item()

    def _build_details_dict(
        self,
        uncertainty: float,
        authority: float,
        input_ids: torch.Tensor,
        response_start: int,
        model_output,
        attn_outputs: dict,
        block_outputs: dict,
        last_layer: int,
        compute_device: torch.device
    ) -> Dict[str, Any]:
        """
        Build detailed results dictionary with component scores.

        Args:
            uncertainty: Computed uncertainty score.
            authority: Aggregated authority score.
            input_ids: Token IDs for sequence.
            response_start: Response boundary index.
            model_output: Model forward output.
            attn_outputs: Captured attention outputs by layer.
            block_outputs: Captured block outputs by layer.
            last_layer: Index of final semantic layer.
            compute_device: Device for tensor operations.

        Returns:
            dict: Detailed breakdown of uncertainty computation.
        """
        logits = model_output.logits.to(compute_device)
        log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
        response_tokens = input_ids[:, response_start:]
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
        model_confidence = token_log_probs.exp().mean().item()

        details = {
            'score': uncertainty,
            'authority': authority,
            'model_confidence': model_confidence,
            'metric': 'authority_flow_v8',
            'response_start': response_start,
            'sequence_length': input_ids.size(1),
        }

        # Add component scores if gating enabled
        if self.config.enable_unified_gating:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)
            if h_attn is not None and h_block is not None:
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)

                gate = compute_stability_gate(
                    h_attn, h_block, self.config.stability_sensitivity
                )
                details['gate'] = gate[:, response_start:].mean().item()

                if self.config.enable_semantic_dispersion and self._embed_matrix is not None:
                    embed_matrix = self._embed_matrix.to(compute_device)
                    dispersion = compute_semantic_dispersion(
                        logits=logits,
                        embed_matrix=embed_matrix,
                        k=self.config.dispersion_k,
                        method=self.config.dispersion_method,
                    )
                    details['dispersion'] = dispersion[:, response_start:].mean().item()

        return details

    def compute_uncertainty_raw(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Compute uncertainty with all internal signals exposed.

        Returns raw per-token tensors for advanced analysis, custom calibration,
        or visualization. Includes attention weights, hidden states, and
        per-token authority scores.

        Args:
            prompt: Source context/question text.
            response: Generated text to evaluate.

        Returns:
            dict with keys:
                - score: float, baseline uncertainty (1 - mean authority)
                - authority: float, mean authority score
                - authority_per_token: Tensor (B, S), per-position authority
                - attention_weights: Tensor (B, H, S, S), attention from last layer
                - h_attn: Tensor (B, S, D), pre-MLP hidden states
                - h_block: Tensor (B, S, D), post-MLP hidden states
                - logits: Tensor (B, S, V), output logits
                - confidence_per_token: Tensor (B, S), max softmax probability
                - response_start: int, response boundary index
                - input_ids: Tensor (B, S), token IDs
                - attention_mask: Tensor (B, S), attention mask
                - embed_matrix: Tensor (V, D), output embeddings (if available)
                - model_confidence: float, mean token probability

        Raises:
            RuntimeError: If attention weights not captured.
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)
        compute_device = self._compute_device

        input_ids = input_ids.to(compute_device)
        attention_mask = attention_mask.to(compute_device)

        Q_stack, K_stack, value_norms_dict, model_output = self._adapter.extract(
            input_ids, attention_mask
        )

        attn_outputs = self._adapter.capture.attn_outputs
        block_outputs = self._adapter.capture.block_outputs
        attention_weights = self._adapter.capture.attention_weights

        last_layer = self._semantic_layer_indices[-1]
        attn = attention_weights.get(last_layer)
        if attn is None:
            raise RuntimeError("Attention weights not captured.")
        attn = attn.to(compute_device)

        h_attn = attn_outputs.get(last_layer)
        h_block = block_outputs.get(last_layer)
        if h_attn is not None:
            h_attn = h_attn.to(compute_device)
        if h_block is not None:
            h_block = h_block.to(compute_device)

        logits = model_output.logits.to(compute_device)

        # Compute raw authority (per-token, baseline method)
        from .ops import compute_authority_flow_vectorized
        authority_per_token = compute_authority_flow_vectorized(
            attn, response_start, None, attention_mask
        )

        # Compute per-token confidence
        probs = torch.softmax(logits, dim=-1)
        confidence_per_token = probs.max(dim=-1).values

        # Compute model confidence for response tokens
        log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
        response_tokens = input_ids[:, response_start:]
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
        model_confidence = token_log_probs.exp().mean().item()

        response_authority = authority_per_token[:, response_start:].mean().item()

        return {
            "score": 1.0 - response_authority,
            "authority": response_authority,
            "authority_per_token": authority_per_token,
            "attention_weights": attn,
            "h_attn": h_attn,
            "h_block": h_block,
            "logits": logits,
            "confidence_per_token": confidence_per_token,
            "response_start": response_start,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "embed_matrix": self._embed_matrix.to(compute_device) if self._embed_matrix is not None else None,
            "model_confidence": model_confidence,
        }

    def detect_context_violation(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect context violations with binary classification.

        Computes uncertainty and compares against threshold to produce a
        binary decision. Returns confidence as distance from decision boundary.

        Detection Scope:
            This detects EXTRINSIC hallucinations (unfaithful to provided context),
            NOT INTRINSIC hallucinations (factually incorrect world knowledge).

        Suitable Use Cases:
            - RAG faithfulness: Is response supported by retrieved documents?
            - Summarization: Does summary misrepresent source?
            - QA with context: Does answer contradict provided passage?

        Unsuitable Use Cases:
            - Open-domain QA without context (no reference to verify against)
            - Detecting outdated knowledge (model may be confidently wrong)

        Args:
            prompt: Source context/question. Should contain reference information.
            response: Generated text to evaluate.
            threshold: Decision boundary in [0, 1]. If None, uses
                config.hallucination_threshold (default 0.7).

        Returns:
            Tuple containing:
                - is_violation: bool, True if uncertainty > threshold
                - confidence: float, |uncertainty - threshold| (certainty of decision)
                - details: dict, component scores from compute_uncertainty

        Raises:
            ValueError: If threshold not in [0, 1].

        Example:
            >>> context = "The capital of France is Paris."
            >>> response = "The capital of France is London."
            >>> is_bad, conf, details = agsar.detect_context_violation(
            ...     context, response
            ... )
            >>> print(f"Violation: {is_bad}, Confidence: {conf:.2f}")
        """
        threshold = threshold or self.config.hallucination_threshold

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        details = self.compute_uncertainty(prompt, response, return_details=True)

        score = details['score']
        is_violation = score > threshold
        confidence = abs(score - threshold)

        return is_violation, confidence, details

    def detect_hallucination(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect hallucination with binary classification.

        .. deprecated:: 0.4.0
            Use :meth:`detect_context_violation` instead. This method only
            detects extrinsic hallucinations; the new name reflects this.

        Args:
            prompt: Input prompt.
            response: Generated response.
            threshold: Decision threshold override.

        Returns:
            Tuple of (is_hallucination, confidence, details).
        """
        import warnings
        warnings.warn(
            "detect_hallucination() is deprecated. Use detect_context_violation() instead. "
            "Note: AG-SAR only detects EXTRINSIC hallucinations (context violations), "
            "not intrinsic hallucinations (factual errors without context).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.detect_context_violation(prompt, response, threshold)

    def reset(self):
        """
        Reset internal state between evaluations.

        Currently a no-op as hooks automatically clear per forward pass.
        Provided for API compatibility and future extensibility.
        """
        pass

    def cleanup(self):
        """
        Release resources and remove hooks.

        Should be called when the engine is no longer needed. Removes
        registered forward hooks from the model and clears cached tensors.

        Side Effects:
            - Removes hooks from model
            - Clears embedding matrix cache
        """
        self._adapter.cleanup()

    def __del__(self):
        """Destructor attempts cleanup but explicit calls are recommended."""
        try:
            self.cleanup()
        except Exception:
            pass
