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
from .centrality import compute_sink_aware_centrality, aggregate_value_norms
from .uncertainty import (
    compute_token_entropy,
    compute_graph_shifted_entropy,
    detect_hallucination as gse_detect_hallucination,
    compute_per_token_uncertainty,
    compute_token_entropy_compiled,
    compute_gse_compiled,
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

        # Tokenize separately to track boundary
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        full_tokens = prompt_tokens + response_tokens
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
        Compute uncertainty (GSE) for a prompt-response pair.

        Full pipeline:
        1. Extract attention maps + value vectors (via hooks)
        2. Filter semantic heads (entropy threshold)
        3. Build global attention graph (with padding mask)
        4. Compute sink-aware centrality (power iteration)
        5. Calculate Graph-Shifted Entropy

        Args:
            prompt: Input prompt text
            response: Generated response text
            return_details: If True, return dict with intermediate computations

        Returns:
            If return_details=False: GSE score (float)
            If return_details=True: Dict with GSE and all intermediate values

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
            use_flash_attn=self.config.use_flash_attn
        )

        # Get logits for entropy computation, cast to target dtype
        logits = model_output.logits.to(self.dtype)

        # Aggregate value norms across layers and heads
        aggregated_value_norms = aggregate_value_norms(
            value_norms,
            semantic_layers=len(self._semantic_layer_indices),
            aggregation='mean'
        ).to(self.dtype)

        # Compute sink-aware centrality via matrix-free Triton kernel
        # When return_details=True, also get per-head contributions for head specialization
        relevance, centrality, per_head_contrib = compute_sink_aware_centrality(
            value_norms=aggregated_value_norms,
            attention_mask=attention_mask,
            num_iterations=self.config.power_iteration_steps,
            tol=self.config.power_iteration_tol,
            Q_stack=Q_stack,
            K_stack=K_stack,
            residual_weight=self.config.residual_weight,
            return_raw=return_details,  # Only compute per-head contrib when needed
            sink_token_count=self.config.sink_token_count,  # Mask BOS/sink tokens
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

        # Compute Graph-Shifted Entropy
        # Use compiled version if enabled for reduced Python overhead
        if self.config.use_torch_compile:
            gse = compute_gse_compiled(
                token_entropy,
                relevance,
                attention_mask=response_mask
            )
        else:
            gse = compute_graph_shifted_entropy(
                token_entropy,
                relevance,
                attention_mask=response_mask
            )

        if return_details:
            result = {
                'gse': gse.item(),
                'token_entropy': token_entropy,
                'relevance': relevance,
                'centrality': centrality,
                'value_norms': aggregated_value_norms,
                'response_start': response_start,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            # Include per-head contributions for head specialization analysis
            if per_head_contrib is not None:
                # Compute head importance as mean contribution across sequence
                # per_head_contrib shape: (B, L*H, S) -> head_importance: (L*H,)
                head_importance = per_head_contrib.abs().mean(dim=(0, 2))  # Average over batch and sequence
                result['head_importance'] = head_importance
                result['per_head_contrib'] = per_head_contrib
            return result

        return gse.item()

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
