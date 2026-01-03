"""
Attention extraction for GPT-2 and Llama-3 models.

Supports:
1. GPT-2: Fused QKV layer (c_attn), no RoPE
2. Llama-3: Separate q_proj, k_proj, v_proj with RoPE and GQA

CRITICAL (Llama-3 RoPE):
    In Llama-3, q_proj and k_proj output UN-ROTATED vectors. RoPE is applied
    AFTER projection but BEFORE attention. We MUST capture post-RoPE vectors
    by monkey-patching LlamaAttention.forward, NOT by hooking q_proj/k_proj.

CRITICAL (Llama-3 GQA):
    Llama-3-8B uses Grouped Query Attention: 32 Q-heads but only 8 KV-heads.
    We expand K from (B, 8, S, D) -> (B, 32, S, D) via repeat_interleave.

SDPA/Flash Attention Compatibility:
    Flash Attention doesn't materialize the NxN attention matrix, so we can't
    get attention weights from output_attentions=True when using SDPA.

    Solution: Hook Q and K vectors, then reconstruct attention post-hoc:
    A = softmax(Q @ K.T / sqrt(d_head))

    This gives us zero-latency inference (Flash Attention speed) with
    post-processing cost only for the layers we analyze.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import types
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version guard for monkey-patched attention methods
# These patches may break with major transformers updates
try:
    import transformers
    _TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    if _TRANSFORMERS_VERSION >= (5, 0):
        warnings.warn(
            f"transformers {transformers.__version__} detected. "
            "Attention monkey-patching in AG-SAR was tested with transformers 4.x. "
            "If you encounter issues with Llama/Qwen/Mistral models, please downgrade "
            "to transformers<5.0.0 or report the issue.",
            UserWarning
        )
except (ImportError, ValueError):
    # transformers not installed or version parsing failed
    _TRANSFORMERS_VERSION = (0, 0)


class AttentionExtractor:
    """
    Hook-based attention weight and value norm extraction for GPT-2 and Llama-3.

    Supports:
        - GPT-2: Hooks c_attn (fused QKV), no RoPE needed
        - Llama-3: Monkey-patches LlamaAttention.forward to capture POST-RoPE Q/K

    GPT-2 Architecture Note:
        GPT-2 uses a single c_attn (Conv1D) layer that outputs 3 * hidden_dim.
        This is split into Q, K, V internally. We hook c_attn to extract V norms.

    Llama-3 Architecture Note:
        Llama-3 uses separate q_proj, k_proj, v_proj followed by RoPE.
        We MUST capture Q/K AFTER RoPE application (inside patched forward).
        Llama-3-8B uses GQA: 32 Q-heads, 8 KV-heads (4 Q per KV group).

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained('gpt2')
        >>> extractor = AttentionExtractor(model)
        >>> extractor.register_hooks()
        >>> Q, K, value_norms, output = extractor.extract_semantic_qk(input_ids)
        >>> extractor.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = None,
        architecture: str = "auto"
    ):
        """
        Initialize attention extractor.

        Args:
            model: Language model (GPT-2, Llama-3, etc.)
            layers: Layer indices to extract from (default: all layers)
            dtype: Tensor dtype (default: model's dtype)
            architecture: "auto", "gpt2", or "llama"
        """
        self.model = model
        self.dtype = dtype or next(model.parameters()).dtype
        self.device = next(model.parameters()).device

        # Detect architecture
        self.architecture = self._detect_architecture(model, architecture)

        # Get model backbone and config based on architecture
        if self.architecture == "gpt2":
            if hasattr(model, 'transformer'):
                self.transformer = model.transformer
            else:
                self.transformer = model
            self.num_layers = len(self.transformer.h)
            self.num_heads = self.transformer.config.n_head
            self.hidden_size = self.transformer.config.n_embd
            self.head_dim = self.hidden_size // self.num_heads
            # GPT-2 has no GQA
            self.num_kv_heads = self.num_heads
            self.heads_per_group = 1
        elif self.architecture == "llama":
            if hasattr(model, 'model'):
                self.transformer = model.model
            else:
                self.transformer = model
            self.num_layers = len(self.transformer.layers)
            self.num_heads = self.transformer.config.num_attention_heads
            self.hidden_size = self.transformer.config.hidden_size
            self.head_dim = self.hidden_size // self.num_heads
            # Llama-3 GQA: fewer KV heads than Q heads
            self.num_kv_heads = getattr(
                self.transformer.config, 'num_key_value_heads', self.num_heads
            )
            self.heads_per_group = self.num_heads // self.num_kv_heads
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self.layers = layers if layers is not None else list(range(self.num_layers))

        # Storage for extracted data (cleared after each forward)
        self._attention_weights: Dict[int, torch.Tensor] = {}
        self._value_norms: Dict[int, torch.Tensor] = {}
        # v3.1 additions: full value states and attention outputs for spectral roughness
        self._value_states: Dict[int, torch.Tensor] = {}   # Full V vectors (B, S, D)
        self._attn_outputs: Dict[int, torch.Tensor] = {}   # h_attn before MLP (B, S, D)
        # v3.2 addition: block outputs for MLP Divergence metric (Llama-3)
        self._block_outputs: Dict[int, torch.Tensor] = {}  # h_final after MLP (B, S, D)
        # Q/K storage for post-hoc attention reconstruction (SDPA compatibility)
        self._query_states: Dict[int, torch.Tensor] = {}
        self._key_states: Dict[int, torch.Tensor] = {}
        # Pre-RoPE Q/K for Llama/Qwen/Mistral (RoPE applied post-hoc)
        self._query_states_pre_rope: Dict[int, torch.Tensor] = {}
        self._key_states_pre_rope: Dict[int, torch.Tensor] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._block_hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Llama-specific: store original forward methods for cleanup
        self._original_forwards: Dict[int, Callable] = {}
        self._llama_patched: bool = False

    def _detect_architecture(self, model: nn.Module, hint: str = "auto") -> str:
        """
        Detect model architecture from structure.

        Supports:
            - GPT-2: transformer.h[i].attn.c_attn (fused QKV, no RoPE)
            - Llama/Qwen/Mistral: model.layers[i].self_attn.q_proj (RoPE + GQA)

        Args:
            model: The language model
            hint: "auto", "gpt2", or "llama"

        Returns:
            "gpt2" or "llama" (llama also covers Qwen, Mistral, and similar)
        """
        if hint != "auto":
            return hint

        # Check for GPT-2 structure
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # Verify it's GPT-2 style (has c_attn)
            if hasattr(model.transformer.h[0].attn, 'c_attn'):
                return "gpt2"

        # Check for Llama/Qwen/Mistral structure (all use model.layers + q_proj)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Verify it's Llama-style (has q_proj in self_attn)
            if hasattr(model.model.layers[0].self_attn, 'q_proj'):
                return "llama"

        # Fallback: check model_type from config
        model_type = getattr(model.config, 'model_type', '').lower()
        if 'gpt2' in model_type:
            return "gpt2"
        # Llama, Qwen, Qwen2, Mistral all use "llama" architecture
        if any(arch in model_type for arch in ['llama', 'qwen', 'mistral']):
            return "llama"

        raise ValueError(
            f"Could not detect model architecture for model_type='{model_type}'. "
            f"Set architecture='gpt2' or architecture='llama' explicitly."
        )

    def _make_c_attn_hook(self, layer_idx: int):
        """
        Create hook for GPT-2's fused c_attn layer.

        CRITICAL: GPT-2 c_attn outputs (batch, seq, 3 * hidden_dim).
        We split into Q, K, V and:
        1. Store full V vectors for spectral roughness (v3.1)
        2. Compute V norms immediately (for sink-aware centrality)
        3. Store Q, K for post-hoc attention reconstruction (SDPA compatibility)

        This enables zero-latency inference with Flash Attention while still
        allowing attention graph construction.
        """
        def hook_fn(module, input_args, output):
            # output shape: (batch, seq, 3 * hidden_dim)
            batch_size, seq_len, _ = output.shape

            # Split into Q, K, V (each is hidden_dim sized)
            q, k, v = output.split(self.hidden_size, dim=-1)

            # v3.1: Store full value vectors (B, S, D) for spectral roughness
            self._value_states[layer_idx] = v.detach().to(self.dtype)

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
        Create hook for attention weights and attention output (h_attn).

        GPT2Attention.forward returns (attn_output, present, attn_weights)
        when output_attentions=True.

        v3.1: Also captures attn_output (h_attn) for spectral roughness.
        """
        def hook_fn(module, input_args, output):
            # GPT2Attention output is tuple: (attn_output, present, attn_weights) or (attn_output, attn_weights)
            if isinstance(output, tuple):
                # v3.1: Capture attention output (h_attn) - first element is always attn_output
                # This is after c_proj but before residual/MLP
                attn_output = output[0]
                if isinstance(attn_output, torch.Tensor) and attn_output.dim() == 3:
                    # Shape: (batch, seq, hidden_size)
                    self._attn_outputs[layer_idx] = attn_output.detach().to(self.dtype)

                # Find attention weights in output
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        # Shape: (batch, num_heads, seq, seq)
                        if item.size(1) == self.num_heads and item.size(2) == item.size(3):
                            self._attention_weights[layer_idx] = item.detach().to(self.dtype)
                            break

        return hook_fn

    def _patch_qwen2_attention(self, layer_idx: int) -> None:
        """
        Copy-paste patch for Qwen2Attention.forward with POST-RoPE capture.

        This copies the exact model code and inserts capture lines after RoPE.
        No math re-computation - we just grab the already-computed variables.
        """
        from transformers.models.qwen2.modeling_qwen2 import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        layer = self.transformer.layers[layer_idx]
        attn = layer.self_attn
        original_forward = attn.forward
        self._original_forwards[layer_idx] = original_forward
        extractor = self

        # Copy-paste of Qwen2Attention.forward with capture inserted
        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Any] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if attn.layer_idx is None:
                    raise ValueError("Cache requires layer index")
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)

            # RoPE application - using Qwen2's exact signature
            cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
            )

            # ★★★ CAPTURE POST-ROPE Q/K HERE ★★★
            extractor._query_states[layer_idx] = query_states.detach().to(extractor.dtype)
            extractor._key_states[layer_idx] = key_states.detach().to(extractor.dtype)
            # Compute V norms (before repeat_kv)
            v_norms = torch.norm(value_states, dim=-1, p=2)
            extractor._value_norms[layer_idx] = v_norms.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )

            # Repeat K/V for GQA
            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            # v3.1: Store full value vectors AFTER repeat_kv (B, S, hidden_size)
            # This ensures shape matches h_attn for roughness computation
            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            extractor._value_states[layer_idx] = v_flat.detach().to(extractor.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ★★★ v3.1: CAPTURE attention weights BEFORE dropout ★★★
            extractor._attention_weights[layer_idx] = attn_weights.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

            # ★★★ v3.1: CAPTURE h_attn BEFORE o_proj ★★★
            extractor._attn_outputs[layer_idx] = attn_output.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_output = attn.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward
        self._llama_patched = True

    def _patch_mistral_attention(self, layer_idx: int) -> None:
        """
        Copy-paste patch for MistralAttention.forward with POST-RoPE capture.

        Mistral architecture is nearly identical to Llama regarding RoPE.
        The main difference is the sliding window attention, but we don't
        need to modify that - we just capture Q/K after RoPE.
        """
        from transformers.models.mistral.modeling_mistral import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        layer = self.transformer.layers[layer_idx]
        attn = layer.self_attn
        original_forward = attn.forward
        self._original_forwards[layer_idx] = original_forward
        extractor = self

        # Full copy-paste of MistralAttention.forward with capture inserted
        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Any] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            # Mistral RoPE - uses seq_len keyword
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if attn.layer_idx is None:
                    raise ValueError("Cache requires layer index")
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)
            cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # ★★★ CAPTURE POST-ROPE Q/K HERE ★★★
            extractor._query_states[layer_idx] = query_states.detach().to(extractor.dtype)
            extractor._key_states[layer_idx] = key_states.detach().to(extractor.dtype)
            # Compute V norms (before repeat_kv)
            v_norms = torch.norm(value_states, dim=-1, p=2)
            extractor._value_norms[layer_idx] = v_norms.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )

            # Repeat K/V for GQA
            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            # v3.1: Store full value vectors AFTER repeat_kv (B, S, hidden_size)
            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            extractor._value_states[layer_idx] = v_flat.detach().to(extractor.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ★★★ v3.1: CAPTURE attention weights BEFORE dropout ★★★
            extractor._attention_weights[layer_idx] = attn_weights.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

            # ★★★ v3.1: CAPTURE h_attn BEFORE o_proj ★★★
            extractor._attn_outputs[layer_idx] = attn_output.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_output = attn.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward
        self._llama_patched = True

    def _patch_llama_attention(self, layer_idx: int) -> None:
        """
        Copy-paste patch for LlamaAttention.forward with POST-RoPE capture.

        v3.1: Full copy-paste to capture h_attn before o_proj.
        """
        from transformers.models.llama.modeling_llama import (
            apply_rotary_pos_emb,
            repeat_kv,
        )

        layer = self.transformer.layers[layer_idx]
        attn = layer.self_attn
        original_forward = attn.forward
        self._original_forwards[layer_idx] = original_forward
        extractor = self

        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Optional[Any] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            cos, sin = attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # ★★★ CAPTURE POST-ROPE Q/K HERE ★★★
            extractor._query_states[layer_idx] = query_states.detach().to(extractor.dtype)
            extractor._key_states[layer_idx] = key_states.detach().to(extractor.dtype)
            # Compute V norms (before repeat_kv)
            v_norms = torch.norm(value_states, dim=-1, p=2)
            extractor._value_norms[layer_idx] = v_norms.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            if past_key_value is not None:
                # Handle cache for generation
                cache_kwargs = {"sin": sin, "cos": cos}
                if hasattr(past_key_value, 'update'):
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )

            # Repeat K/V for GQA
            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            # v3.1: Store full value vectors AFTER repeat_kv (B, S, hidden_size)
            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            extractor._value_states[layer_idx] = v_flat.detach().to(extractor.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # ★★★ v3.1: CAPTURE attention weights ★★★
            extractor._attention_weights[layer_idx] = attn_weights.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

            # ★★★ v3.1: CAPTURE h_attn BEFORE o_proj ★★★
            extractor._attn_outputs[layer_idx] = attn_output.detach().to(extractor.dtype)
            # ★★★ END CAPTURE ★★★

            attn_output = attn.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward
        self._llama_patched = True

    def _restore_llama_attention(self) -> None:
        """
        Restore original LlamaAttention.forward methods.

        Call this in cleanup() to avoid side effects after extraction.
        """
        for layer_idx, original_forward in self._original_forwards.items():
            layer = self.transformer.layers[layer_idx]
            # Restore original bound method
            layer.self_attn.forward = original_forward

        self._original_forwards.clear()
        self._llama_patched = False

    def _make_block_hook(self, layer_idx: int):
        """
        Create hook for decoder block output (post-MLP).

        v3.2: Captures h_final for MLP Divergence metric on Llama-3.
        The MLP Divergence measures: 1 - CosineSim(h_attn, h_final)

        Args:
            layer_idx: Index of the decoder layer

        Returns:
            Hook function that captures block output
        """
        def hook_fn(module, input_args, output):
            # Decoder layer output is tuple: (hidden_states, ...)
            # hidden_states is after Attention + MLP + Residuals
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                # Shape: (batch, seq, hidden_size)
                self._block_outputs[layer_idx] = hidden_states.detach().to(self.dtype)

        return hook_fn

    def register_hooks(self) -> None:
        """Register forward hooks on attention layers."""
        self.remove_hooks()

        if self.architecture == "gpt2":
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

        elif self.architecture == "llama":
            # Detect specific model type for correct patching
            model_type = getattr(self.model.config, 'model_type', '').lower()

            for layer_idx in self.layers:
                if 'qwen' in model_type:
                    self._patch_qwen2_attention(layer_idx)
                elif 'mistral' in model_type:
                    self._patch_mistral_attention(layer_idx)
                else:
                    self._patch_llama_attention(layer_idx)

                # v3.2: Register block hooks for MLP Divergence metric
                # Block output = hidden state after Attention + MLP + Residuals
                layer = self.transformer.layers[layer_idx]
                block_handle = layer.register_forward_hook(
                    self._make_block_hook(layer_idx)
                )
                self._block_hooks.append(block_handle)

    def remove_hooks(self) -> None:
        """Remove all registered hooks and restore patched methods."""
        # Remove GPT-2 style hooks
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Remove block hooks (v3.2)
        for handle in self._block_hooks:
            handle.remove()
        self._block_hooks.clear()

        # Restore Llama patched forward methods
        if self._llama_patched:
            self._restore_llama_attention()

    def clear_cache(self) -> None:
        """Clear stored attention weights, value norms, and Q/K states."""
        self._attention_weights.clear()
        self._value_norms.clear()
        self._value_states.clear()
        self._attn_outputs.clear()
        self._block_outputs.clear()
        self._query_states.clear()
        self._key_states.clear()
        self._query_states_pre_rope.clear()
        self._key_states_pre_rope.clear()

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

    @torch.inference_mode()
    def extract_semantic_qk(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_flash_attn: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Any]:
        """
        Extract stacked Q/K vectors for matrix-free centrality computation.

        Unlike extract(), this does NOT call reconstruct_attention() and
        does NOT create O(S^2) attention matrices. This is the key to
        the matrix-free centrality approach.

        The method:
        1. Runs forward pass with hooks registered (captures Q, K, V norms)
        2. Stacks Q/K from all registered layers
        3. Returns value_norms for sink-aware weighting

        Args:
            input_ids: (B, S) input tokens
            attention_mask: (B, S) padding mask
            use_flash_attn: Enable Flash Attention context

        Returns:
            Q_stack: (B, L*H, S, D) stacked queries from semantic layers
            K_stack: (B, L*H, S, D) stacked keys from semantic layers
            value_norms: Dict[layer_idx, (B, H, S)] value norms per layer
            model_output: Model forward output with logits
        """
        self.clear_cache()

        # Move inputs to model device with async transfer
        input_ids = input_ids.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        # Run forward pass (hooks capture Q, K, V norms)
        if use_flash_attn:
            try:
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=False,
                        return_dict=True
                    )
            except ImportError:
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
                output_attentions=False,
                return_dict=True
            )

        # Post-RoPE Q/K are already captured by patched forward methods
        # No post-hoc processing needed

        # Stack Q/K from all registered layers (NO attention reconstruction!)
        Q_list = []
        K_list = []
        V_norms_expanded = {}

        for layer_idx in sorted(self.layers):
            if layer_idx in self._query_states:
                Q = self._query_states[layer_idx]  # (B, num_q_heads, S, D)
                K = self._key_states[layer_idx]    # (B, num_kv_heads, S, D)

                # Handle GQA expansion for Llama-3
                # If num_kv_heads < num_heads, expand K to match Q
                if self.heads_per_group > 1:
                    # Expand K from (B, 8, S, D) -> (B, 32, S, D)
                    # repeat_interleave(4, dim=1) duplicates each KV head 4 times
                    K = K.repeat_interleave(self.heads_per_group, dim=1)

                    # Also expand V norms for consistency
                    if layer_idx in self._value_norms:
                        v_norms = self._value_norms[layer_idx]  # (B, num_kv_heads, S)
                        v_norms_expanded = v_norms.repeat_interleave(
                            self.heads_per_group, dim=1
                        )  # (B, num_q_heads, S)
                        V_norms_expanded[layer_idx] = v_norms_expanded

                Q_list.append(Q)
                K_list.append(K)

        if not Q_list:
            raise RuntimeError(
                "No Q/K states captured. Ensure hooks are registered "
                "and layers are correctly specified."
            )

        # Stack along head dimension: (B, L*H, S, D)
        Q_stack = torch.cat(Q_list, dim=1)
        K_stack = torch.cat(K_list, dim=1)

        # Use expanded V norms if GQA, otherwise original
        value_norms_out = V_norms_expanded if V_norms_expanded else dict(self._value_norms)

        return Q_stack, K_stack, value_norms_out, output

    @property
    def extracted_layers(self) -> List[int]:
        """Return list of layer indices that have been extracted."""
        return sorted(set(self._attention_weights.keys()) | set(self._value_norms.keys()))
