"""
Attention extraction for GPT-2 models.

CRITICAL: GPT-2 uses a fused QKV layer (c_attn) unlike Llama which has
separate q_proj, k_proj, v_proj. We must hook c_attn and split the output.

SDPA/Flash Attention Compatibility:
    Flash Attention doesn't materialize the NxN attention matrix, so we can't
    get attention weights from output_attentions=True when using SDPA.

    Solution: Hook Q and K vectors, then reconstruct attention post-hoc:
    A = softmax(Q @ K.T / sqrt(d_head))

    This gives us zero-latency inference (Flash Attention speed) with
    post-processing cost only for the layers we analyze.
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionExtractor:
    """
    Hook-based attention weight and value norm extraction for GPT-2.

    GPT-2 Architecture Note:
        GPT-2 uses a single c_attn (Conv1D) layer that outputs 3 * hidden_dim.
        This is split into Q, K, V internally. We hook c_attn to extract V norms.

    Example:
        >>> from transformers import GPT2LMHeadModel
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> extractor = AttentionExtractor(model)
        >>> extractor.register_hooks()
        >>> attn_weights, value_norms, output = extractor.extract(input_ids)
        >>> extractor.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize attention extractor for GPT-2.

        Args:
            model: GPT-2 model (GPT2Model or GPT2LMHeadModel)
            layers: Layer indices to extract from (default: all layers)
            dtype: Tensor dtype (default: model's dtype)
        """
        self.model = model
        self.dtype = dtype or next(model.parameters()).dtype
        self.device = next(model.parameters()).device

        # Detect model structure (GPT2LMHeadModel vs GPT2Model)
        if hasattr(model, 'transformer'):
            self.transformer = model.transformer
        else:
            self.transformer = model

        self.num_layers = len(self.transformer.h)
        self.layers = layers if layers is not None else list(range(self.num_layers))

        # Get model config
        self.num_heads = self.transformer.config.n_head
        self.hidden_size = self.transformer.config.n_embd
        self.head_dim = self.hidden_size // self.num_heads

        # Storage for extracted data (cleared after each forward)
        self._attention_weights: Dict[int, torch.Tensor] = {}
        self._value_norms: Dict[int, torch.Tensor] = {}
        # Q/K storage for post-hoc attention reconstruction (SDPA compatibility)
        self._query_states: Dict[int, torch.Tensor] = {}
        self._key_states: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _make_c_attn_hook(self, layer_idx: int):
        """
        Create hook for GPT-2's fused c_attn layer.

        CRITICAL: GPT-2 c_attn outputs (batch, seq, 3 * hidden_dim).
        We split into Q, K, V and:
        1. Compute V norms immediately (for sink-aware centrality)
        2. Store Q, K for post-hoc attention reconstruction (SDPA compatibility)

        This enables zero-latency inference with Flash Attention while still
        allowing attention graph construction.
        """
        def hook_fn(module, input_args, output):
            # output shape: (batch, seq, 3 * hidden_dim)
            batch_size, seq_len, _ = output.shape

            # Split into Q, K, V (each is hidden_dim sized)
            q, k, v = output.split(self.hidden_size, dim=-1)

            # Reshape to (batch, num_heads, seq, head_dim) for attention computation
            q_heads = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_heads = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_heads = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Store Q, K for post-hoc attention reconstruction
            # Detach to prevent gradient tracking, keep on device for speed
            self._query_states[layer_idx] = q_heads.detach().to(self.dtype)
            self._key_states[layer_idx] = k_heads.detach().to(self.dtype)

            # Compute L2 norm per head per token: (batch, num_heads, seq)
            # CRITICAL: Compute norm inside hook and detach immediately
            v_norms = torch.norm(v_heads, dim=-1, p=2)

            # Detach and store with proper dtype
            self._value_norms[layer_idx] = v_norms.detach().to(self.dtype)

        return hook_fn

    def _make_attn_hook(self, layer_idx: int):
        """
        Create hook for attention weights.

        GPT2Attention.forward returns (attn_output, present, attn_weights)
        when output_attentions=True.
        """
        def hook_fn(module, input_args, output):
            # GPT2Attention output is tuple: (attn_output, present, attn_weights) or (attn_output, attn_weights)
            if isinstance(output, tuple):
                # Find attention weights in output
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        # Shape: (batch, num_heads, seq, seq)
                        if item.size(1) == self.num_heads and item.size(2) == item.size(3):
                            self._attention_weights[layer_idx] = item.detach().to(self.dtype)
                            break

        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on attention layers."""
        self.remove_hooks()

        for layer_idx in self.layers:
            block = self.transformer.h[layer_idx]

            # Hook c_attn for value norms (fused QKV projection)
            c_attn_handle = block.attn.c_attn.register_forward_hook(
                self._make_c_attn_hook(layer_idx)
            )
            self._hooks.append(c_attn_handle)

            # Hook attention module for attention weights
            attn_handle = block.attn.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._hooks.append(attn_handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def clear_cache(self) -> None:
        """Clear stored attention weights, value norms, and Q/K states."""
        self._attention_weights.clear()
        self._value_norms.clear()
        self._query_states.clear()
        self._key_states.clear()

    def reconstruct_attention(
        self,
        layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Reconstruct attention weights from stored Q and K vectors.

        This is the key to SDPA/Flash Attention compatibility:
        A = softmax(Q @ K.T / sqrt(d_head))

        Args:
            layer_idx: Layer index to reconstruct
            attention_mask: Optional causal/padding mask (batch, seq)

        Returns:
            attention_weights: (batch, heads, seq, seq)
        """
        if layer_idx not in self._query_states:
            raise ValueError(f"No Q/K stored for layer {layer_idx}")

        Q = self._query_states[layer_idx]  # (batch, heads, seq, head_dim)
        K = self._key_states[layer_idx]    # (batch, heads, seq, head_dim)

        # Compute attention scores: Q @ K.T / sqrt(d)
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        # = (batch, heads, seq, seq)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) * scale

        # Apply causal mask (GPT-2 is causal/autoregressive)
        seq_len = Q.size(2)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq) -> (batch, 1, 1, seq)
            padding_mask = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores - padding_mask * 1e9

        # Softmax to get attention probabilities
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

        return attn_weights.to(self.dtype)

    @torch.inference_mode()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_flash_attn: bool = False
    ) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Any]:
        """
        Run forward pass and extract attention data.

        SDPA/Flash Attention Compatible:
            When using SDPA (default in modern transformers), attention weights
            are NOT returned by the model. Instead, we:
            1. Hook Q, K vectors during forward pass
            2. Reconstruct attention post-hoc: A = softmax(QK^T / sqrt(d))

            This gives zero-latency inference with Flash Attention speed,
            paying reconstruction cost only for the semantic layers we analyze.

        Args:
            input_ids: (batch, seq) input token IDs
            attention_mask: (batch, seq) attention mask
            use_flash_attn: Enable Flash Attention context (H100 optimization)

        Returns:
            attention_weights: Dict[layer_idx, Tensor(batch, heads, seq, seq)]
            value_norms: Dict[layer_idx, Tensor(batch, heads, seq)]
            model_output: Original model output with logits
        """
        self.clear_cache()

        # Move inputs to model device with async transfer
        input_ids = input_ids.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        # Run forward pass
        # Don't require output_attentions - we reconstruct from Q/K hooks
        if use_flash_attn:
            # Use new API (torch.nn.attention.sdpa_kernel) if available,
            # fall back to deprecated API for older PyTorch versions
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=False,  # Not needed - we reconstruct
                        return_dict=True
                    )
            except ImportError:
                # Fall back to deprecated API for PyTorch < 2.5
                if hasattr(torch.backends.cuda, 'sdp_kernel'):
                    with torch.backends.cuda.sdp_kernel(
                        enable_flash=True,
                        enable_math=False,
                        enable_mem_efficient=False
                    ):
                        output = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_attentions=False,
                            return_dict=True
                        )
                else:
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=False,
                        return_dict=True
                    )
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,  # Not needed - we reconstruct
                return_dict=True
            )

        # Reconstruct attention weights from Q/K (SDPA-compatible)
        # This is the key innovation: post-hoc reconstruction instead of
        # asking SDPA to materialize the NxN matrix during forward pass
        for layer_idx in self.layers:
            if layer_idx in self._query_states:
                self._attention_weights[layer_idx] = self.reconstruct_attention(
                    layer_idx, attention_mask
                )

        return (
            dict(self._attention_weights),
            dict(self._value_norms),
            output
        )

    @property
    def extracted_layers(self) -> List[int]:
        """Return list of layer indices that have been extracted."""
        return sorted(set(self._attention_weights.keys()) | set(self._value_norms.keys()))
