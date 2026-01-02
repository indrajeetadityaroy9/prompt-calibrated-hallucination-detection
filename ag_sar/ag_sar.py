"""
AG-SAR: Attention-Graph Shifting Attention to Relevance

Main pipeline class that orchestrates all modules for zero-latency
uncertainty quantification in LLMs.

Example:
    >>> from transformers import GPT2LMHeadModel, GPT2Tokenizer
    >>> from ag_sar import AGSAR
    >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
    >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    >>> ag_sar = AGSAR(model, tokenizer)
    >>> gse = ag_sar.compute_uncertainty("The capital of France is", "Paris")
    >>> is_hall, conf, details = ag_sar.detect_hallucination(
    ...     "The capital of France is", "London"
    ... )
"""

from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import AGSARConfig
from .utils import enable_tf32, get_model_dtype, get_model_device
from .attention_extractor import AttentionExtractor
from .centrality import (
    compute_sink_aware_centrality,
    aggregate_value_norms,
    compute_hebbian_weights,
)
from .uncertainty import (
    compute_token_entropy,
    compute_graph_shifted_entropy,
    compute_graph_shifted_surprisal,
    compute_token_surprisal,
    detect_hallucination as gse_detect_hallucination,
    compute_per_token_uncertainty,
    compute_token_entropy_compiled,
    compute_gse_compiled,
    compute_bounded_surprisal,
    compute_manifold_consistent_spectral_surprisal,
)


class AGSAR:
    """
    AG-SAR: Attention-Graph Shifting Attention to Relevance.

    Zero-latency uncertainty quantification by analyzing internal
    attention graph structure. No external semantic models required.

    Optimized for NVIDIA H100 with:
    - BFloat16 precision (prevents GPT-2 NaN overflow)
    - torch.compile for hot paths
    - Flash Attention 2 support
    - Pure PyTorch graph operations (no NetworkX)

    Attributes:
        model: The language model (GPT-2 or similar)
        tokenizer: Corresponding tokenizer
        config: AG-SAR configuration
        dtype: Tensor dtype (bfloat16 recommended)
        device: Model device

    Example:
        >>> ag_sar = AGSAR(model, tokenizer)
        >>> gse = ag_sar.compute_uncertainty("What is 2+2?", "4")
        >>> print(f"Uncertainty: {gse:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[AGSARConfig] = None
    ):
        """
        Initialize AG-SAR pipeline.

        Args:
            model: HuggingFace model (GPT2LMHeadModel or similar)
            tokenizer: Corresponding tokenizer
            config: AG-SAR configuration (uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AGSARConfig()

        # Match model precision - use bfloat16 for H100
        self.dtype = self.config.preferred_dtype
        self.device = get_model_device(model)

        # Enable TF32 for H100 Tensor Core acceleration
        enable_tf32()

        # Determine which layers to use
        self._detect_model_config()

        # Initialize attention extractor with semantic layers only
        self.extractor = AttentionExtractor(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype
        )

        # Register hooks
        self.extractor.register_hooks()

        # Load head scores for SGSS (Surprisal-Gated Spectral Steering)
        # These are Z-scored calibration scores: positive=Truth Head, negative=Induction Head
        self._head_scores = None
        if self.config.use_spectral_steering and self.config.head_scores_path:
            self._head_scores = self._load_head_scores(self.config.head_scores_path)

    def _detect_model_config(self) -> None:
        """Detect model architecture and set layer configuration."""
        # Get transformer block reference
        if hasattr(self.model, 'transformer'):
            # GPT-2 style: model.transformer
            transformer = self.model.transformer
        elif hasattr(self.model, 'model'):
            # Llama/Qwen/Mistral style: model.model
            transformer = self.model.model
        else:
            transformer = self.model

        # Get number of layers
        if hasattr(transformer, 'h'):
            # GPT-2 style: transformer.h[i]
            self.num_layers = len(transformer.h)
        elif hasattr(transformer, 'layers'):
            # Llama/Qwen/Mistral style: transformer.layers[i]
            self.num_layers = len(transformer.layers)
        elif hasattr(transformer, 'config') and hasattr(transformer.config, 'num_hidden_layers'):
            # Fallback: Llama config
            self.num_layers = transformer.config.num_hidden_layers
        elif hasattr(transformer, 'config') and hasattr(transformer.config, 'n_layer'):
            # Fallback: GPT-2 config
            self.num_layers = transformer.config.n_layer
        else:
            self.num_layers = 12  # GPT-2 default

        # Calculate semantic layer indices (last N layers)
        semantic_count = min(self.config.semantic_layers, self.num_layers)
        start_layer = max(0, self.num_layers - semantic_count)
        self._semantic_layer_indices = list(range(start_layer, self.num_layers))

    def _load_head_scores(self, path: str) -> torch.Tensor:
        """
        Load head scores for SGSS with auto-detection of format.

        SGSS uses Z-scored head calibration scores:
        - Positive scores = Truth Heads (upweight when confident)
        - Negative scores = Induction Heads (downweight when confident)

        Auto-detects format:
        - Native SGSS: {"head_z_scores": [...]}
        - Legacy sigmoid: {"head_weights": [...]} in [0,1] range

        For legacy format, converts to Z-scores via inverse sigmoid scaling:
        z = (w - 0.5) * 4.0  # Maps 0.5→0, 0→-2, 1→+2

        Args:
            path: Path to head scores JSON file

        Returns:
            head_scores: (L*H,) tensor of Z-scored head calibration scores

        Raises:
            FileNotFoundError: If scores file doesn't exist
            ValueError: If file contains neither valid format
        """
        import json
        with open(path) as f:
            data = json.load(f)

        if 'head_z_scores' in data:
            # Native SGSS format (Z-scores)
            scores = torch.tensor(
                data['head_z_scores'],
                dtype=self.dtype,
                device=self.device
            )
        elif 'head_weights' in data:
            # Legacy format: convert [0,1] sigmoid weights to centered Z-scores
            # Maps 0.5 → 0 (neutral), 0 → -2 (strong Induction), 1 → +2 (strong Truth)
            weights = torch.tensor(
                data['head_weights'],
                dtype=self.dtype,
                device=self.device
            )
            scores = (weights - 0.5) * 4.0
        else:
            raise ValueError(
                f"Scores file {path} must contain 'head_z_scores' (native SGSS) "
                f"or 'head_weights' (legacy sigmoid) key"
            )

        return scores

    def _tokenize(
        self,
        prompt: str,
        response: str
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Tokenize prompt and response.

        Args:
            prompt: Input prompt text
            response: Generated response text

        Returns:
            input_ids: (1, seq_len) token IDs
            attention_mask: (1, seq_len) attention mask
            response_start: Index where response begins
        """
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Tokenize prompt + response together to preserve proper tokenization
        # (GPT-style tokenizers include leading spaces as part of tokens)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_text = prompt + response
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        response_start = len(prompt_tokens)

        input_ids = torch.tensor([full_tokens], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, response_start

    @torch.inference_mode()
    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        return_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute uncertainty for a prompt-response pair.

        Supports multiple uncertainty metrics via config.uncertainty_metric:
        - "gse" (default): Graph-Shifted Entropy
        - "mcss": Manifold-Consistent Spectral Surprisal (catches Confident Lies)

        Full pipeline:
        1. Extract Q/K/V vectors via hooks (no O(N²) attention materialization)
        2. Compute Hebbian weights if MC-SS (prompt-anchored manifold prior)
        3. Compute sink-aware centrality via Triton kernel (O(N) matrix-free)
        4. Calculate final uncertainty metric (GSE or MC-SS)

        MC-SS uses ADDITIVE formulation: S_bounded + λ(1 - centrality)
        This catches "Confident Lies" where S ≈ 0 but token is ungrounded.

        Args:
            prompt: Input prompt text
            response: Generated response text
            return_details: If True, return dict with intermediate computations

        Returns:
            If return_details=False: Uncertainty score (float)
            If return_details=True: Dict with uncertainty and all intermediate values

        Raises:
            ValueError: If response is empty (no tokens to analyze)
        """
        # Handle empty response
        if not response or not response.strip():
            if return_details:
                return {
                    'gse': 0.0,
                    'token_entropy': torch.tensor([[]], device=self.device),
                    'relevance': torch.tensor([[]], device=self.device),
                    'centrality': torch.tensor([[]], device=self.device),
                    'value_norms': torch.tensor([[]], device=self.device),
                    'response_start': 0,
                    'input_ids': torch.tensor([[]], device=self.device),
                    'attention_mask': torch.tensor([[]], device=self.device)
                }
            return 0.0

        # Tokenize
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)

        # Validate response bounds
        seq_len = input_ids.size(1)
        if response_start >= seq_len:
            # Response tokens were somehow not added (shouldn't happen, but defensive)
            if return_details:
                return {
                    'gse': 0.0,
                    'token_entropy': torch.zeros(1, seq_len, device=self.device, dtype=self.dtype),
                    'relevance': torch.zeros(1, seq_len, device=self.device, dtype=self.dtype),
                    'centrality': torch.zeros(1, seq_len, device=self.device, dtype=self.dtype),
                    'value_norms': torch.zeros(1, seq_len, device=self.device, dtype=self.dtype),
                    'response_start': response_start,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask
                }
            return 0.0

        # Matrix-free extraction: get Q/K stacks without O(N^2) attention matrices
        Q_stack, K_stack, value_norms, model_output = self.extractor.extract_semantic_qk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_flash_attn=True  # Always use Flash Attention for matrix-free pipeline
        )

        # Get logits for entropy computation, cast to target dtype
        logits = model_output.logits.to(self.dtype)

        # Aggregate value norms across layers and heads
        aggregated_value_norms = aggregate_value_norms(
            value_norms,
            semantic_layers=len(self._semantic_layer_indices),
            aggregation='mean'
        ).to(self.dtype)

        # Compute raw surprisal EARLY for SGSS gating (before centrality)
        # SGSS needs token-level surprisal to gate head contributions
        raw_surprisal = None
        if self.config.use_spectral_steering and self._head_scores is not None:
            raw_surprisal = compute_token_surprisal(
                logits,
                input_ids,
                attention_mask=None  # No masking for raw surprisal - SGSS needs all tokens
            )

        # Determine if MC-SS metric is requested
        use_mcss = self.config.uncertainty_metric == "mcss"

        # Compute Hebbian weights if MC-SS is enabled
        # Uses prompt-only centroid as anchor (prevents hallucination from polluting manifold)
        hebbian_weights = None
        if use_mcss:
            hebbian_weights = compute_hebbian_weights(
                K_stack=K_stack,
                prompt_end_idx=response_start,
                tau=self.config.mcss_hebbian_tau,
                attention_mask=attention_mask,
            )

        # Compute sink-aware centrality via matrix-free Triton kernel
        # When return_details=True, also get per-head contributions for head specialization
        # When use_hebbian=True, applies Hebbian prior to filter centrality by semantic grounding
        # When SGSS is enabled, applies surprisal-gated dynamic head weighting
        relevance, centrality, per_head_contrib = compute_sink_aware_centrality(
            value_norms=aggregated_value_norms,
            Q_stack=Q_stack,
            K_stack=K_stack,
            attention_mask=attention_mask,
            num_iterations=self.config.power_iteration_steps,
            tol=self.config.power_iteration_tol,
            residual_weight=self.config.residual_weight,
            return_raw=return_details,  # Only compute per-head contrib when needed
            sink_token_count=self.config.sink_token_count,  # Mask BOS/sink tokens
            hebbian_weights=hebbian_weights,
            use_hebbian=use_mcss,
            # SGSS: Surprisal-Gated Spectral Steering
            surprisal=raw_surprisal,
            head_scores=self._head_scores,
            steering_alpha=self.config.steering_alpha,
            steering_beta=self.config.steering_beta,
            response_start=response_start,
        )

        # Create response mask (only compute entropy on response tokens)
        response_mask = torch.zeros_like(attention_mask)
        response_mask[:, response_start:] = 1

        # Compute token entropy (only for response)
        # Use compiled version if enabled for reduced Python overhead
        if self.config.use_torch_compile:
            token_entropy = compute_token_entropy_compiled(
                logits,
                attention_mask=response_mask
            )
        else:
            token_entropy = compute_token_entropy(
                logits,
                attention_mask=response_mask
            )

        # Compute final uncertainty metric based on config
        if self.config.uncertainty_metric == "mcss":
            # MC-SS: Manifold-Consistent Spectral Surprisal
            # Uses ADDITIVE formulation: S_bounded + λ(1 - MaxNorm(centrality))
            # This catches "Confident Lies" that multiplicative would miss
            bounded_surprisal = compute_bounded_surprisal(
                logits=logits,
                input_ids=input_ids,
                beta=self.config.mcss_beta,
                attention_mask=response_mask,
            )
            uncertainty = compute_manifold_consistent_spectral_surprisal(
                bounded_surprisal=bounded_surprisal,
                centrality=relevance,  # Use relevance (value-norm weighted centrality)
                attention_mask=response_mask,
                penalty_weight=self.config.mcss_penalty_weight,
            )
        elif self.config.uncertainty_metric == "gss":
            # GSS: Graph-Shifted Surprisal
            # Uses surprisal (NLL) instead of entropy, weighted by relevance
            # Better for forced response evaluation where we have ground truth tokens
            token_surprisal = compute_token_surprisal(
                logits,
                input_ids,
                attention_mask=response_mask
            )
            uncertainty = compute_graph_shifted_surprisal(
                token_surprisal,
                relevance,
                attention_mask=response_mask
            )
        else:
            # GSE: Graph-Shifted Entropy (default)
            # Use compiled version if enabled for reduced Python overhead
            if self.config.use_torch_compile:
                uncertainty = compute_gse_compiled(
                    token_entropy,
                    relevance,
                    attention_mask=response_mask
                )
            else:
                uncertainty = compute_graph_shifted_entropy(
                    token_entropy,
                    relevance,
                    attention_mask=response_mask
                )

        if return_details:
            result = {
                'uncertainty': uncertainty.item(),
                'metric': self.config.uncertainty_metric,
                # Legacy key for backwards compatibility
                'gse': uncertainty.item(),
                'token_entropy': token_entropy,
                'relevance': relevance,
                'centrality': centrality,
                'value_norms': aggregated_value_norms,
                'response_start': response_start,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'logits': logits,  # Include for downstream surprisal computation
            }
            # Include MC-SS specific fields when using that metric
            if self.config.uncertainty_metric == "mcss":
                result['bounded_surprisal'] = bounded_surprisal
                result['hebbian_weights'] = hebbian_weights
            # Include SGSS specific fields when using spectral steering
            if self.config.use_spectral_steering and raw_surprisal is not None:
                result['raw_surprisal'] = raw_surprisal
                result['sgss_enabled'] = True
            # Include per-head contributions for head specialization analysis
            if per_head_contrib is not None:
                # Compute head importance as mean contribution across sequence
                # per_head_contrib shape: (B, L*H, S) -> head_importance: (L*H,)
                head_importance = per_head_contrib.abs().mean(dim=(0, 2))  # Average over batch and sequence
                result['head_importance'] = head_importance
                result['per_head_contrib'] = per_head_contrib
            return result

        return uncertainty.item()

    @torch.inference_mode()
    def detect_hallucination(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if response contains hallucination.

        Args:
            prompt: Input prompt
            response: Generated response
            threshold: Custom threshold (uses config default if None)

        Returns:
            is_hallucination: Boolean indicating likely hallucination
            confidence: Confidence score (0-1, higher = more certain)
            details: Dictionary with GSE and other metrics
        """
        threshold = threshold or self.config.hallucination_threshold

        # Get full uncertainty details
        details = self.compute_uncertainty(prompt, response, return_details=True)
        gse = torch.tensor([details['gse']], device=self.device)

        # Apply threshold
        is_hall, confidence = gse_detect_hallucination(gse, threshold)

        return (
            is_hall.item(),
            confidence.item(),
            details
        )

    @torch.inference_mode()
    def batch_compute_uncertainty(
        self,
        prompts: List[str],
        responses: List[str]
    ) -> List[float]:
        """
        Compute uncertainty for multiple prompt-response pairs.

        Note: Currently processes sequentially. For true batching,
        inputs would need padding alignment.

        Args:
            prompts: List of prompt texts
            responses: List of response texts

        Returns:
            List of GSE scores
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")

        return [
            self.compute_uncertainty(p, r)
            for p, r in zip(prompts, responses)
        ]

    def get_token_contributions(
        self,
        prompt: str,
        response: str
    ) -> Dict[str, Any]:
        """
        Get per-token uncertainty contributions.

        Useful for understanding which tokens drive the GSE score.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Dictionary with per-token analysis
        """
        details = self.compute_uncertainty(prompt, response, return_details=True)

        contributions = compute_per_token_uncertainty(
            details['token_entropy'],
            details['relevance'],
            details['attention_mask']
        )

        # Decode tokens
        input_ids = details['input_ids'][0]
        response_start = details['response_start']

        token_info = []
        for i in range(response_start, len(input_ids)):
            token_id = input_ids[i].item()
            token_info.append({
                'position': i,
                'token': self.tokenizer.decode([token_id]),
                'entropy': details['token_entropy'][0, i].item(),
                'relevance': details['relevance'][0, i].item(),
                'centrality': details['centrality'][0, i].item(),
                'contribution': contributions[0, i].item()
            })

        return {
            'gse': details['gse'],
            'tokens': token_info,
            'response_start': response_start
        }

    def cleanup(self) -> None:
        """Remove hooks and free resources."""
        self.extractor.remove_hooks()
        self.extractor.clear_cache()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'extractor'):
            self.cleanup()

    def __repr__(self) -> str:
        return (
            f"AGSAR(model={type(self.model).__name__}, "
            f"layers={len(self._semantic_layer_indices)}, "
            f"dtype={self.dtype}, "
            f"threshold={self.config.hallucination_threshold})"
        )
