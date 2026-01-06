"""
AGSAR Engine - Main Orchestrator.

This is the primary user-facing interface for AG-SAR uncertainty quantification.
It coordinates the modeling, measures, and ops layers to provide a clean API.

Scope: AG-SAR detects EXTRINSIC hallucinations (unfaithful to source context),
not INTRINSIC hallucinations (unfaithful to reality). For RAG faithfulness
monitoring, the ground truth is provided in the context.

Example:
    >>> from ag_sar import AGSAR, AGSARConfig
    >>> config = AGSARConfig()  # Uses v8.0 defaults
    >>> agsar = AGSAR(model, tokenizer, config)
    >>> score = agsar.compute_uncertainty(prompt, response)
    >>> is_hallucination, confidence, details = agsar.detect_hallucination(prompt, response)
"""

from typing import Optional, Tuple, Dict, Any, Union
import os
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
    AG-SAR Uncertainty Quantification Engine (v8.0 Gold Master).

    Provides zero-latency detection of EXTRINSIC hallucinations by analyzing
    internal attention graph structure without external semantic models.

    Core architecture:
        1. Authority Flow: Tracks signal provenance from prompt to response
        2. Unified Gating: Balances context vs parametric trust dynamically
        3. Semantic Dispersion: Measures consistency over raw confidence

    Scope: Detects violations of SOURCE CONTEXT, not violations of reality.
    Optimized for RAG faithfulness monitoring.

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

        self._num_layers = num_layers

        # Use last N layers for semantic analysis (v8.0: late-layer focus is optimal)
        start_layer = max(0, num_layers - self.config.semantic_layers)
        self._semantic_layer_indices = list(range(start_layer, num_layers))

        # Initialize model adapter for attention extraction
        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
        )
        self._adapter.register()

        # Capture output embedding matrix (needed for semantic dispersion)
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

        Uses Authority Flow with Unified Gating and Semantic Dispersion.

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

        # Get captured tensors for authority flow computation
        attn_outputs = self._adapter.capture.attn_outputs
        block_outputs = self._adapter.capture.block_outputs
        attention_weights = self._adapter.capture.attention_weights

        # Get attention weights from last semantic layer
        last_layer = self._semantic_layer_indices[-1]
        attn = attention_weights.get(last_layer)
        if attn is None:
            raise RuntimeError(
                "Attention weights not captured. This may indicate hook registration "
                "failed or architecture mismatch. Check that the model architecture "
                "is supported (GPT-2, Llama, Mistral, Qwen)."
            )
        # Move attention to compute device for multi-GPU compatibility
        attn = attn.to(compute_device)

        # Context-Dependent Gating: Unified RAG + Free Gen
        # Uses stability gate + semantic dispersion (consistency over confidence)
        if self.config.enable_unified_gating:
            h_attn = attn_outputs.get(last_layer)
            h_block = block_outputs.get(last_layer)

            if h_attn is not None and h_block is not None:
                h_attn = h_attn.to(compute_device)
                h_block = h_block.to(compute_device)
                logits = model_output.logits.to(compute_device)

                # Semantic Dispersion (Consistency over Confidence) - default mode
                if self.config.enable_semantic_dispersion and self._embed_matrix is not None:
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
                    # Fallback: Raw confidence (when semantic dispersion disabled)
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
                # Fallback to regular authority if hidden states not captured
                authority = compute_authority_score(
                    attn, response_start,
                    attention_mask=attention_mask,
                    use_vectorized=True,
                )
        else:
            # v3.1 Legacy Path (for paper comparison: pure authority flow)
            authority = compute_authority_score(
                attn, response_start,
                attention_mask=attention_mask,
                use_vectorized=True,
            )

        # Aggregate authority over response tokens using configured method
        # Conservative methods (min, percentile) improve TPR @ low FPR
        response_authority_tokens = authority[:, response_start:]

        if self.config.aggregation_method == "mean":
            response_authority = response_authority_tokens.mean().item()
        elif self.config.aggregation_method == "min":
            response_authority = response_authority_tokens.min().item()
        elif self.config.aggregation_method == "percentile_10":
            response_authority = torch.quantile(response_authority_tokens.flatten(), 0.10).item()
        elif self.config.aggregation_method == "percentile_25":
            response_authority = torch.quantile(response_authority_tokens.flatten(), 0.25).item()
        elif self.config.aggregation_method == "importance_weighted":
            # SOTA: Weight by self-information (-log p) to emphasize rare tokens
            # Errors on rare tokens (names, dates) count 5x more than common tokens
            # This aligns uncertainty with human judgment of severity
            logits = model_output.logits.to(compute_device)
            probs = torch.softmax(logits[:, response_start-1:-1, :], dim=-1)
            response_ids = input_ids[:, response_start:]
            # Get probability of actual token: P(token_t | context)
            token_probs = probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)  # (B, S_resp)
            # Self-information: -log(p) - higher for rare tokens
            self_info = -torch.log(token_probs + 1e-10)  # (B, S_resp)
            # Normalize to weights (sum to 1)
            importance_weights = self_info / (self_info.sum(dim=-1, keepdim=True) + 1e-10)
            # Weighted average of authority scores
            response_authority = (response_authority_tokens * importance_weights).sum().item()
        else:
            # Fallback to mean
            response_authority = response_authority_tokens.mean().item()

        # Uncertainty = 1 - aggregated authority
        uncertainty = 1.0 - response_authority

        if return_details:
            # Compute model confidence (mean probability of response tokens)
            # Move logits to compute device for multi-GPU compatibility
            logits = model_output.logits.to(compute_device)
            log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
            response_tokens = input_ids[:, response_start:]
            token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
            model_confidence = token_log_probs.exp().mean().item()

            details = {
                'score': uncertainty,
                'authority': response_authority,
                'model_confidence': model_confidence,
                'metric': 'authority_flow_v8',
                'response_start': response_start,
                'sequence_length': input_ids.size(1),
            }

            # Compute component scores for mechanism analysis (Knowledge Conflict Diagnosis)
            if self.config.enable_unified_gating:
                h_attn = attn_outputs.get(last_layer)
                h_block = block_outputs.get(last_layer)
                if h_attn is not None and h_block is not None:
                    h_attn = h_attn.to(compute_device)
                    h_block = h_block.to(compute_device)

                    # Gate: Context reliance (1.0 = trust context, 0.0 = trust memory)
                    gate = compute_stability_gate(
                        h_attn, h_block, self.config.stability_sensitivity
                    )
                    response_gate = gate[:, response_start:].mean().item()
                    details['gate'] = response_gate

                    # Dispersion: Semantic consistency (lower = more consistent)
                    if self.config.enable_semantic_dispersion and self._embed_matrix is not None:
                        embed_matrix = self._embed_matrix.to(compute_device)
                        dispersion = compute_semantic_dispersion(
                            logits=logits,
                            embed_matrix=embed_matrix,
                            k=self.config.dispersion_k,
                            method=self.config.dispersion_method,
                        )
                        response_dispersion = dispersion[:, response_start:].mean().item()
                        details['dispersion'] = response_dispersion

            return details

        return uncertainty

    def compute_uncertainty_raw(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Compute uncertainty with ALL internal signals exposed.

        This method exposes raw internal signals for advanced analysis
        or custom calibration approaches.

        Returns:
            Dict containing:
                - score: Final uncertainty score
                - authority_per_token: (B, S) raw authority scores
                - attention_weights: (B, H, S, S) attention from last layer
                - h_attn: (B, S, D) pre-MLP hidden states
                - h_block: (B, S, D) post-MLP hidden states
                - logits: (B, S, V) output logits
                - confidence_per_token: (B, S) softmax confidence
                - response_start: int
                - embed_matrix: (V, D) output embeddings (if available)
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)
        compute_device = self._compute_device

        input_ids = input_ids.to(compute_device)
        attention_mask = attention_mask.to(compute_device)

        # Extract attention data
        Q_stack, K_stack, value_norms_dict, model_output = self._adapter.extract(
            input_ids, attention_mask
        )

        # Get captured tensors
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

        # Compute raw authority (per-token, not aggregated)
        from .ops import compute_authority_flow_vectorized

        authority_per_token = compute_authority_flow_vectorized(
            attn, response_start, None, attention_mask
        )

        # Compute confidence per token
        probs = torch.softmax(logits, dim=-1)
        confidence_per_token = probs.max(dim=-1).values

        # Compute model confidence (mean prob of response tokens)
        # This avoids a redundant forward pass by computing inline
        log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
        response_tokens = input_ids[:, response_start:]
        token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)
        model_confidence = token_log_probs.exp().mean().item()

        # Compute aggregated authority for baseline comparison (without another forward pass)
        response_authority = authority_per_token[:, response_start:].mean().item()

        return {
            "score": 1.0 - response_authority,  # Baseline uncertainty
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
        Detect context violations (extrinsic hallucinations) with binary classification.

        This method detects when the model's response is UNFAITHFUL to the provided context,
        not when the response is factually incorrect in general.

        IMPORTANT: AG-SAR only detects EXTRINSIC hallucinations (context violations).
        It does NOT detect INTRINSIC hallucinations (factually incorrect claims that
        are consistent with the model's parametric knowledge).

        Use cases where this works well:
        - RAG faithfulness: Is the response supported by retrieved documents?
        - Summarization accuracy: Does the summary misrepresent the source?
        - QA with context: Does the answer contradict the provided passage?

        Use cases where this does NOT work:
        - Open-domain QA without context (use external fact-checking)
        - Detecting outdated knowledge (the model confidently uses old facts)

        Args:
            prompt: Input prompt (should include context for best results)
            response: Generated response to evaluate
            threshold: Override default threshold (default: 0.7)

        Returns:
            Tuple of:
                - is_violation: True if response likely violates context
                - confidence: Distance from decision boundary (higher = more certain)
                - details: Dict with intermediate values (score, authority, etc.)

        Example:
            >>> context = "The capital of France is Paris."
            >>> question = "What is the capital of France?"
            >>> response = "The capital of France is London."
            >>> is_bad, conf, details = agsar.detect_context_violation(
            ...     f"{context}\\n\\n{question}", response
            ... )
            >>> print(f"Violation: {is_bad}, Confidence: {conf:.2f}")
            Violation: True, Confidence: 0.23
        """
        threshold = threshold or self.config.hallucination_threshold

        # Validate threshold bounds
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")

        details = self.compute_uncertainty(prompt, response, return_details=True)

        score = details['score']
        is_violation = score > threshold
        confidence = abs(score - threshold)  # Distance from decision boundary

        return is_violation, confidence, details

    def detect_hallucination(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect hallucination with binary classification.

        .. deprecated::
            Use :meth:`detect_context_violation` instead. This method only detects
            EXTRINSIC hallucinations (context violations), not intrinsic ones.
            The new name better reflects the actual capability.

        Args:
            prompt: Input prompt
            response: Generated response
            threshold: Override default threshold

        Returns:
            is_hallucination: True if likely hallucinating
            confidence: How confident the detection is (distance from threshold)
            details: Dict with intermediate values
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
        """Reset state between prompt-response pairs (no-op in v8.0)."""
        pass

    def cleanup(self):
        """Release resources."""
        self._adapter.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
