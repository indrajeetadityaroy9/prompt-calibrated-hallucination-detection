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
    # Gated Authority (v7.0+)
    compute_gated_authority,
    compute_semantic_authority,
    # Layer Stability (v11.0)
    compute_layer_drift,
    apply_drift_penalty,
    # Symbolic Overlap (v13.0 - Hybrid Controller)
    compute_context_overlap,
    compute_numeric_consistency,
)
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
        # Pass drift_layer_ratio if Layer Drift is enabled
        drift_layer_ratio = None
        if self.config.enable_layer_drift:
            drift_layer_ratio = self.config.drift_layer_ratio

        # Request hidden states if intrinsic detection is enabled (v12.0)
        output_hidden_states = self.config.enable_intrinsic_detection

        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
            drift_layer_ratio=drift_layer_ratio,
            output_hidden_states=output_hidden_states,
        )
        self._adapter.register()

        # Cache lm_head weight for Layer Drift computation
        self._lm_head_weight = None
        if self.config.enable_layer_drift:
            try:
                if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                    self._lm_head_weight = model.lm_head.weight.detach()
                elif hasattr(model, 'get_output_embeddings'):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        self._lm_head_weight = output_embeddings.weight.detach()
            except Exception:
                pass

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

        # Initialize Truth Vector for intrinsic hallucination detection (v12.0)
        self._truth_vector = None
        self._tv_meta = None
        self._truth_vector_layer = None
        if self.config.enable_intrinsic_detection:
            if self.config.truth_vector_path is None:
                raise ValueError(
                    "enable_intrinsic_detection=True but truth_vector_path not set. "
                    "Run scripts/calibrate_truth_vector.py first."
                )
            from .calibration.truth_vector import TruthVectorCalibrator
            self._truth_vector, self._tv_meta = TruthVectorCalibrator.load(
                self.config.truth_vector_path,
                device=str(self._compute_device),
            )
            self._truth_vector_layer = self._tv_meta["layer_index"]
            print(f"Loaded Truth Vector from {self.config.truth_vector_path}")
            print(f"  Layer: {self._truth_vector_layer}, Samples: {self._tv_meta['n_samples']}")

        # Initialize JEPA Predictor for drift-based detection (Phase 3)
        self._jepa_predictor = None
        if self.config.enable_jepa_monitor:
            from .modeling.online_predictor import OnlineJepaPredictor

            # Use OnlineJepaPredictor for Test-Time Training capability
            self._jepa_predictor = OnlineJepaPredictor(
                input_dim=model.config.hidden_size,
                hidden_dim=self.config.predictor_hidden_dim,
                lr=self.config.online_adaptation_lr,
            ).to(self._compute_device)

            # Optionally load pretrained weights as a "prior"
            if (self.config.jepa_predictor_path and
                os.path.exists(self.config.jepa_predictor_path)):
                print(f"Loading JEPA prior from {self.config.jepa_predictor_path}...")
                state_dict = torch.load(
                    self.config.jepa_predictor_path,
                    map_location=self._compute_device,
                )
                self._jepa_predictor.load_state_dict(state_dict, strict=False)

            # Save initial state for reset between samples
            self._jepa_predictor.save_initial_state()

            print(f"  JEPA Monitor: ENABLED (Layer {self.config.jepa_monitor_layer})")
            print(f"  Online Adaptation: {'ENABLED' if self.config.enable_online_adaptation else 'DISABLED'}")
            print(f"  TTT Epochs: {self.config.online_adaptation_epochs}")
            print(f"  Drift Threshold: {self.config.jepa_drift_threshold}")

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

                    # Compute Intrinsic Trust if enabled (v12.0)
                    intrinsic_trust = None
                    if self.config.enable_intrinsic_detection and self._truth_vector is not None:
                        from .calibration.truth_vector import compute_intrinsic_score
                        import torch.nn.functional as F

                        # Get hidden state from truth vector layer
                        # We need to run the model to get hidden states from that layer
                        # The hidden states are already captured in model_output
                        hidden_states = model_output.hidden_states
                        if hidden_states is not None and len(hidden_states) > self._truth_vector_layer:
                            tv_hidden = hidden_states[self._truth_vector_layer].to(compute_device)

                            # Compute raw cosine similarity (per token)
                            truth_vec = self._truth_vector.to(compute_device)
                            raw_sim = F.cosine_similarity(
                                tv_hidden,
                                truth_vec.unsqueeze(0).unsqueeze(0),
                                dim=-1
                            )  # (B, S)

                            # Normalize using calibration bounds
                            mu_pos = self._tv_meta["mu_pos"]
                            mu_neg = self._tv_meta["mu_neg"]
                            denom = mu_pos - mu_neg

                            if abs(denom) < 1e-6:
                                # Fallback: if vector is collapsed, use raw similarity
                                intrinsic_trust = (raw_sim + 1) / 2
                            else:
                                intrinsic_trust = (raw_sim - mu_neg) / denom

                            intrinsic_trust = intrinsic_trust.clamp(0.0, 1.0)

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
                        intrinsic_trust=intrinsic_trust,  # v12.0
                        # Gate Sharpening (v12.1)
                        enable_gate_sharpening=self.config.enable_gate_sharpening,
                        gate_sharpen_low=self.config.gate_sharpen_low,
                        gate_sharpen_high=self.config.gate_sharpen_high,
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

        # Layer Drift (v11.0): Apply "mind-change" penalty if enabled
        # This detects when the model changed its prediction between mid and final layers
        if (self.config.enable_layer_drift and
            self._lm_head_weight is not None and
            self._adapter.capture.mid_layer_hidden_states is not None):

            mid_hidden = self._adapter.capture.mid_layer_hidden_states.to(compute_device)
            final_hidden = block_outputs.get(last_layer)

            if final_hidden is not None:
                final_hidden = final_hidden.to(compute_device)
                lm_head = self._lm_head_weight.to(compute_device)

                # Compute layer drift (how much did the model change its mind?)
                drift = compute_layer_drift(
                    mid_hidden,
                    final_hidden,
                    lm_head,
                    temperature=1.0,
                    use_kl=False,
                )

                # Apply drift penalty to authority: high drift = lower authority
                authority = apply_drift_penalty(
                    authority,
                    drift,
                    sensitivity=self.config.drift_sensitivity,
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

            return {
                'score': uncertainty,
                'authority': response_authority,
                'model_confidence': model_confidence,
                'metric': 'authority_flow_v8',
                'response_start': response_start,
                'sequence_length': input_ids.size(1),
            }

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

        # Compute layer drift if enabled
        layer_drift = None
        mid_layer_hidden_states = None
        if (self.config.enable_layer_drift and
            self._lm_head_weight is not None and
            self._adapter.capture.mid_layer_hidden_states is not None):

            mid_layer_hidden_states = self._adapter.capture.mid_layer_hidden_states.to(compute_device)

            if h_block is not None:
                lm_head = self._lm_head_weight.to(compute_device)
                layer_drift = compute_layer_drift(
                    mid_layer_hidden_states,
                    h_block,
                    lm_head,
                    temperature=1.0,
                    use_kl=False,
                )

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
            # Layer Drift (v11.0)
            "mid_layer_hidden_states": mid_layer_hidden_states,
            "mid_layer_idx": self._adapter.capture.mid_layer_idx,
            "layer_drift": layer_drift,
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

    def compute_drift(
        self,
        prompt: str,
        response: str,
    ) -> Dict[str, Any]:
        """
        Compute Semantic Drift using the JEPA Predictor with Test-Time Training.

        This method:
        1. Extracts hidden states from the full prompt+response
        2. If online adaptation is enabled, trains the predictor on CONTEXT transitions
        3. Measures drift on RESPONSE transitions

        The key insight: By training on context first, the predictor learns
        "the facts in THIS document" and can detect when the response deviates.

        Args:
            prompt: Input prompt (should contain context for RAG)
            response: Generated response

        Returns:
            Dict containing:
                - drift: Average MSE drift over response tokens
                - trust_score: Inverse mapping of drift (high trust = low drift)
                - is_hallucination: True if drift exceeds threshold
                - context_loss: Training loss from TTT (if enabled)
        """
        if not self.config.enable_jepa_monitor or self._jepa_predictor is None:
            raise RuntimeError(
                "JEPA Monitor not enabled. Set enable_jepa_monitor=True in config."
            )

        import torch.nn.functional as F

        full_text = prompt + response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self._compute_device)

        # Identify response token start index
        prompt_len = self.tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        total_len = inputs.input_ids.shape[1]

        # Forward pass to get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract hidden states from monitor layer [Batch, Seq, Dim]
        hidden_states = outputs.hidden_states[self.config.jepa_monitor_layer]

        # Remove batch dimension for cleaner indexing
        hidden_seq = hidden_states[0]  # [Seq, Dim]

        # === TEST-TIME TRAINING (The Key Innovation) ===
        context_loss = 0.0
        if self.config.enable_online_adaptation and prompt_len > 2:
            # Reset predictor to initial state (or prior)
            self._jepa_predictor.reset()

            # Extract CONTEXT hidden states (everything before response)
            context_states = hidden_seq[:prompt_len]

            # Train predictor on context transitions
            # This teaches it: "In THIS document, X leads to Y"
            context_loss = self._jepa_predictor.fit(
                context_states,
                epochs=self.config.online_adaptation_epochs,
            )

        # === MEASURE DRIFT ON RESPONSE ===
        # Input: h_{prompt_len-1} to h_{total-2} (we need context to predict first response token)
        # Target: h_{prompt_len} to h_{total-1}
        response_input = hidden_seq[prompt_len - 1 : -1]
        response_target = hidden_seq[prompt_len:]

        drift_score, drift_per_token = self._jepa_predictor.compute_drift(
            response_input, response_target
        )

        # === RELATIVE DRIFT NORMALIZATION ===
        # Key insight: Raw drift varies with context complexity.
        # Normalize by context_loss to get a relative measure:
        # drift_ratio = response_drift / context_loss
        # - ratio ≈ 1.0: Response follows same patterns as context (grounded)
        # - ratio > 1.0: Response deviates from context patterns (hallucination)
        drift_ratio = drift_score / (context_loss + 1e-6)

        # Map drift to trust (inverse relationship)
        # Using sigmoid-style mapping for bounded output
        trust_score = 1.0 / (1.0 + drift_score * 10)

        return {
            "drift": drift_score,
            "drift_ratio": drift_ratio,  # Normalized drift (key metric)
            "trust_score": trust_score,
            "is_hallucination": drift_ratio > 1.5,  # Use ratio for detection
            "context_loss": context_loss,
            "drift_per_token": drift_per_token.numpy() if drift_per_token is not None else None,
            "prompt_length": prompt_len,
            "response_length": total_len - prompt_len,
        }

    def compute_hybrid_trust(
        self,
        prompt: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compute trust using the Universal Veto Engine (v13.0 Final).

        Architecture:
        1. NEURAL TRUST (Dynamic Blending):
           - Context-Heavy (Gate→1): Use JEPA Variance (best for RAG)
           - Context-Light (Gate→0): Use Truth Vector (best for myths/lies)
           neural_trust = gate * jepa_trust + (1 - gate) * intrinsic_trust

        2. SYMBOLIC VETO (Hard Filter):
           - If entity/numeric violation detected, CAP trust at 0.2
           - This overrides neural confidence for blatant hallucinations

        Args:
            prompt: Input prompt (may contain context)
            response: Generated response text
            context: Optional explicit context (if separate from prompt)

        Returns:
            Dict with trust scores and component details
        """
        # Use prompt as context if not explicitly provided
        context_text = context if context else prompt

        # === PHASE 1: NEURAL TRUST (Dynamic Blending) ===

        # 1. Compute base uncertainty (includes gate, dispersion, authority)
        base_result = self.compute_uncertainty(prompt, response, return_details=True)

        # 2. JEPA Trust (from Authority/Dispersion - best for RAG)
        jepa_trust = base_result.get('authority', 0.5)

        # 3. Intrinsic Trust (Truth Vector - best for myths/lies)
        intrinsic_trust = 0.5  # Default neutral
        if self.config.enable_intrinsic_detection and self._truth_vector is not None:
            intrinsic_trust = base_result.get('intrinsic_trust', 0.5)

        # 4. Compute Gate (context attention indicator)
        # High gate = model is attending to context = trust JEPA
        # Low gate = model ignoring context = trust Truth Vector
        # Heuristic: Long context = RAG scenario = high gate
        gate = 0.9 if len(context_text) > 200 else (0.7 if len(context_text) > 50 else 0.3)

        # 5. Dynamic Blending: The Core "Universal" Score
        neural_trust = (gate * jepa_trust) + ((1.0 - gate) * intrinsic_trust)

        # === PHASE 2: SYMBOLIC VETO (Hard Filter) ===

        # 6. Compute Symbolic Overlap (Entity Check)
        symbolic_score, symbolic_details = compute_context_overlap(response, context_text)

        # 7. Compute Numeric Consistency
        numeric_score = 1.0
        numeric_details = {}
        if self.config.enable_numeric_check:
            numeric_score, numeric_details = compute_numeric_consistency(response, context_text)

        # 8. Combined symbolic score (worst of entity/numeric)
        symbolic_combined = min(symbolic_score, numeric_score)

        # 9. VETO LOGIC: Hard Filter
        # If symbolic check fails (<=0.5), we CAP trust regardless of neural confidence
        # This catches "Paris vs London" errors that neural methods miss
        # Note: 0.5 = single entity violation, which should trigger veto
        veto_triggered = False
        veto_cap = 0.2  # Maximum trust when veto triggers

        if symbolic_combined <= 0.5:
            # VETO: Entity/Numeric violation detected
            final_trust = min(neural_trust, veto_cap)
            veto_triggered = True
        else:
            # No veto: Use neural trust as-is
            final_trust = neural_trust

        # 10. Hallucination Detection
        is_hallucination = veto_triggered or (final_trust < 0.5)

        return {
            # Final scores
            "trust_score": final_trust,
            "uncertainty": 1.0 - final_trust,
            "is_hallucination": is_hallucination,
            # Neural components
            "neural_trust": neural_trust,
            "jepa_trust": jepa_trust,
            "intrinsic_trust": intrinsic_trust,
            "gate": gate,
            # Symbolic components
            "symbolic_score": symbolic_score,
            "numeric_score": numeric_score,
            "symbolic_combined": symbolic_combined,
            # Veto status
            "veto_triggered": veto_triggered,
            "veto_cap": veto_cap if veto_triggered else None,
            # Details
            "symbolic_details": symbolic_details,
            "numeric_details": numeric_details,
        }

    def _measure_drift(
        self,
        hidden_states: torch.Tensor,
        start_idx: int,
    ) -> tuple:
        """
        Measure prediction error for response tokens using JEPA predictor.

        Args:
            hidden_states: (B, S, D) hidden states from monitor layer
            start_idx: Index where response starts

        Returns:
            Tuple of (avg_drift, drift_per_token)
        """
        import torch.nn.functional as F

        # Ensure we have context for prediction
        if start_idx < 1:
            start_idx = 1

        # Input to predictor: h_t (from start-1 to end-1)
        # Target for predictor: h_{t+1} (from start to end)
        input_seq = hidden_states[:, start_idx - 1 : -1, :]
        target_seq = hidden_states[:, start_idx:, :]

        if input_seq.shape[1] == 0:
            return 0.0, None

        # Predict using JEPA predictor
        with torch.no_grad():
            # Convert to float32 for predictor (trained in float32)
            input_float = input_seq.float()
            predicted_seq = self._jepa_predictor(input_float)

            # MSE per token (averaged over hidden dim)
            target_float = target_seq.float()
            mse_per_token = F.mse_loss(
                predicted_seq, target_float, reduction="none"
            ).mean(dim=-1)

        # Average over sequence
        avg_drift = mse_per_token.mean().item()
        drift_per_token = mse_per_token.squeeze(0).cpu().numpy()

        return avg_drift, drift_per_token

    def cleanup(self):
        """Release resources."""
        self._adapter.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
