"""
Unified Model Hooking for Attention Extraction.

ModelAdapter provides a clean interface for extracting Q/K/V states from:
- GPT-2: Fused c_attn hook (no RoPE)
- Llama/Qwen/Mistral: Monkey-patched forward for post-RoPE capture + GQA

H100 Optimizations:
- Flash Attention 2 (~2x speedup)
- BFloat16 precision (stable numerics, high performance)
- Multi-GPU support with device_map="balanced"
- Native device tensor handling (no unnecessary transfers)

CRITICAL: Llama-style models apply RoPE AFTER projection. We must capture
Q/K after RoPE application, not from raw projection outputs.

CRITICAL: transformers >= 4.45 has breaking changes for attention hooks.
Use transformers >= 4.40.0, < 4.45.0.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import types
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Version guard
try:
    import transformers
    _TRANSFORMERS_VERSION = tuple(int(x) for x in transformers.__version__.split('.')[:2])
    if _TRANSFORMERS_VERSION >= (5, 0):
        warnings.warn(
            f"transformers {transformers.__version__} detected. "
            "AG-SAR was tested with transformers 4.x. "
            "If you encounter issues, downgrade to transformers<5.0.0."
        )
except (ImportError, ValueError):
    _TRANSFORMERS_VERSION = (0, 0)


def load_model_h100(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "flash_attention_2",
    device_map: str = "balanced",
    low_cpu_mem_usage: bool = True,
) -> Tuple[nn.Module, Any]:
    """
    Load model with H100-optimized settings.

    Optimal for 2x H100 80GB setup with Llama-3-70B.

    Args:
        model_id: HuggingFace model ID or local path
        dtype: Model precision (bfloat16 recommended for H100)
        attn_implementation: Attention backend ("flash_attention_2" for H100)
        device_map: Multi-GPU strategy ("balanced" distributes evenly)
        low_cpu_mem_usage: Use system RAM during loading

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_model_h100("meta-llama/Llama-3-70B-Instruct")
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.eval()

    return model, tokenizer


@dataclass
class AttentionCapture:
    """
    Container for captured attention data from one forward pass.

    Multi-GPU Note:
        Tensors are stored on their NATIVE device (GPU where layer resides).
        This avoids costly NVLink transfers during capture.
        Aggregation to a single device happens only when computing final scores.
    """
    query_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    key_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    value_norms: Dict[int, torch.Tensor] = field(default_factory=dict)
    value_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    block_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        """Clear all captured data."""
        self.query_states.clear()
        self.key_states.clear()
        self.value_norms.clear()
        self.value_states.clear()
        self.attn_outputs.clear()
        self.block_outputs.clear()
        self.attention_weights.clear()

    def get_device(self, layer_idx: int) -> Optional[torch.device]:
        """Get the device where a layer's tensors are stored."""
        if layer_idx in self.query_states:
            return self.query_states[layer_idx].device
        return None


class ModelAdapter:
    """
    Unified adapter for extracting attention data from various LLM architectures.

    Supported architectures:
        - GPT-2: Uses c_attn hook for fused QKV extraction
        - Llama/Qwen/Mistral: Monkey-patches attention.forward for post-RoPE Q/K

    Example:
        >>> adapter = ModelAdapter(model)
        >>> adapter.register()
        >>> Q, K, v_norms, output = adapter.extract(input_ids)
        >>> adapter.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize model adapter.

        Args:
            model: Language model (GPT-2, Llama, etc.)
            layers: Layer indices to extract (default: all layers)
            dtype: Tensor dtype (default: model's dtype)
        """
        self.model = model
        self.dtype = dtype or next(model.parameters()).dtype
        self.device = next(model.parameters()).device

        # Detect architecture
        self.architecture = self._detect_architecture()

        # Get backbone and config
        self._setup_architecture()

        # Use specified layers or all
        self.layers = layers if layers is not None else list(range(self.num_layers))

        # Storage and hooks
        self.capture = AttentionCapture()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._original_forwards: Dict[int, Callable] = {}
        self._is_registered = False

    def _detect_architecture(self) -> str:
        """Detect model architecture from structure."""
        # GPT-2 detection
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            if hasattr(self.model.transformer.h[0].attn, 'c_attn'):
                return "gpt2"

        # Llama/Qwen/Mistral detection
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            if hasattr(self.model.model.layers[0].self_attn, 'q_proj'):
                return "llama"

        # Check config model_type
        model_type = getattr(self.model.config, 'model_type', '').lower()
        if 'gpt2' in model_type:
            return "gpt2"
        # MoE architectures (Mixtral, Qwen-MoE) use standard attention
        # MoE only affects FFN layers (sparse experts), not attention hooks
        # Qwen1.5-MoE uses 'qwen2_moe', Mixtral uses 'mixtral'
        if any(arch in model_type for arch in ['llama', 'qwen', 'mistral', 'mixtral', 'moe']):
            return "llama"

        raise ValueError(
            f"Unknown architecture for model_type='{model_type}'. "
            "Supported: gpt2, llama, qwen, mistral, mixtral, qwen-moe."
        )

    def _setup_architecture(self):
        """Setup architecture-specific attributes."""
        if self.architecture == "gpt2":
            self.backbone = (
                self.model.transformer if hasattr(self.model, 'transformer')
                else self.model
            )
            self.num_layers = len(self.backbone.h)
            self.num_heads = self.backbone.config.n_head
            self.hidden_size = self.backbone.config.n_embd
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = self.num_heads
            self.heads_per_group = 1
        else:  # llama
            self.backbone = (
                self.model.model if hasattr(self.model, 'model')
                else self.model
            )
            self.num_layers = len(self.backbone.layers)
            self.num_heads = self.backbone.config.num_attention_heads
            self.hidden_size = self.backbone.config.hidden_size
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = getattr(
                self.backbone.config, 'num_key_value_heads', self.num_heads
            )
            self.heads_per_group = self.num_heads // self.num_kv_heads

    def register(self) -> None:
        """Register hooks for attention extraction."""
        if self._is_registered:
            return

        self.cleanup()

        if self.architecture == "gpt2":
            self._register_gpt2_hooks()
        else:
            self._register_llama_hooks()

        self._is_registered = True

    def cleanup(self) -> None:
        """Remove all hooks and restore original methods."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        # Architecture-specific cleanup: restore original forward methods
        if self.architecture in ("llama", "mistral", "qwen"):
            for layer_idx, original in self._original_forwards.items():
                self.backbone.layers[layer_idx].self_attn.forward = original
        # GPT-2 uses hooks only, no monkey-patching to undo
        self._original_forwards.clear()

        self._is_registered = False

    def _register_gpt2_hooks(self):
        """Register GPT-2 style hooks."""
        for layer_idx in self.layers:
            block = self.backbone.h[layer_idx]

            # Hook c_attn for QKV and attention weights
            handle = block.attn.c_attn.register_forward_hook(
                self._make_gpt2_qkv_hook(layer_idx)
            )
            self._hooks.append(handle)

            # Hook attention for h_attn (attention output)
            handle = block.attn.register_forward_hook(
                self._make_gpt2_attn_hook(layer_idx)
            )
            self._hooks.append(handle)

            # Block hook for MLP divergence (GPT-2 block output)
            handle = block.register_forward_hook(
                self._make_block_hook(layer_idx)
            )
            self._hooks.append(handle)

    def _register_llama_hooks(self):
        """Register Llama/Qwen/Mistral hooks via monkey-patching."""
        model_type = getattr(self.model.config, 'model_type', '').lower()

        for layer_idx in self.layers:
            if 'qwen' in model_type:
                self._patch_qwen_attention(layer_idx)
            elif 'mistral' in model_type:
                self._patch_mistral_attention(layer_idx)
            else:
                self._patch_llama_attention(layer_idx)

            # Block hook for MLP divergence
            layer = self.backbone.layers[layer_idx]
            handle = layer.register_forward_hook(self._make_block_hook(layer_idx))
            self._hooks.append(handle)

    def _make_gpt2_qkv_hook(self, layer_idx: int):
        """Hook for GPT-2 fused c_attn."""
        def hook_fn(module, input_args, output):
            batch_size, seq_len, _ = output.shape
            q, k, v = output.split(self.hidden_size, dim=-1)

            tensor_device = v.device
            self.capture.value_states[layer_idx] = v.detach().to(device=tensor_device, dtype=self.dtype)

            q_heads = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_heads = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_heads = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            self.capture.query_states[layer_idx] = q_heads.detach().to(device=tensor_device, dtype=self.dtype)
            self.capture.key_states[layer_idx] = k_heads.detach().to(device=tensor_device, dtype=self.dtype)
            self.capture.value_norms[layer_idx] = torch.norm(v_heads, dim=-1, p=2).detach().to(device=tensor_device, dtype=self.dtype)

            # Compute attention weights for GPT-2 (no RoPE, so Q/K are ready)
            # GPT-2 uses causal mask internally, but we compute full attention here
            attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Apply causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tensor_device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.dtype)
            self.capture.attention_weights[layer_idx] = attn_weights.detach()

        return hook_fn

    def _make_gpt2_attn_hook(self, layer_idx: int):
        """Hook for GPT-2 attention output."""
        def hook_fn(module, input_args, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                if isinstance(attn_output, torch.Tensor) and attn_output.dim() == 3:
                    self.capture.attn_outputs[layer_idx] = attn_output.detach().to(
                        device=attn_output.device, dtype=self.dtype
                    )

        return hook_fn

    def _make_block_hook(self, layer_idx: int):
        """Hook for decoder block output (MLP divergence)."""
        def hook_fn(module, input_args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                self.capture.block_outputs[layer_idx] = hidden_states.detach().to(
                    device=hidden_states.device, dtype=self.dtype
                )

        return hook_fn

    def _patch_llama_attention(self, layer_idx: int):
        """Patch LlamaAttention.forward for post-RoPE capture."""
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

        layer = self.backbone.layers[layer_idx]
        attn = layer.self_attn
        self._original_forwards[layer_idx] = attn.forward
        adapter = self

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

            # CAPTURE POST-ROPE (preserve native device for multi-GPU)
            tensor_device = query_states.device
            adapter.capture.query_states[layer_idx] = query_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.key_states[layer_idx] = key_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2).detach().to(device=tensor_device, dtype=adapter.dtype)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                if hasattr(past_key_value, 'update'):
                    key_states, value_states = past_key_value.update(
                        key_states, value_states, layer_idx, cache_kwargs
                    )

            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            adapter.capture.value_states[layer_idx] = v_flat.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            adapter.capture.attention_weights[layer_idx] = attn_weights.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

            attn_output = attn.o_proj(attn_output)

            # CAPTURE AFTER o_proj to match block_outputs hidden_size
            adapter.capture.attn_outputs[layer_idx] = attn_output.detach().to(device=tensor_device, dtype=adapter.dtype)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward

    def _patch_qwen_attention(self, layer_idx: int):
        """Patch Qwen2Attention.forward for post-RoPE capture."""
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

        layer = self.backbone.layers[layer_idx]
        attn = layer.self_attn
        self._original_forwards[layer_idx] = attn.forward
        adapter = self

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

            cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

            # CAPTURE (preserve native device for multi-GPU)
            tensor_device = query_states.device
            adapter.capture.query_states[layer_idx] = query_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.key_states[layer_idx] = key_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2).detach().to(device=tensor_device, dtype=adapter.dtype)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            adapter.capture.value_states[layer_idx] = v_flat.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            adapter.capture.attention_weights[layer_idx] = attn_weights.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, attn.hidden_size)

            attn_output = attn.o_proj(attn_output)

            # CAPTURE AFTER o_proj to match block_outputs hidden_size
            adapter.capture.attn_outputs[layer_idx] = attn_output.detach().to(device=tensor_device, dtype=adapter.dtype)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward

    def _patch_mistral_attention(self, layer_idx: int):
        """Patch MistralAttention.forward for post-RoPE capture."""
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

        layer = self.backbone.layers[layer_idx]
        attn = layer.self_attn
        self._original_forwards[layer_idx] = attn.forward
        adapter = self

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

            # Apply RoPE - use position_ids like Llama (modern transformers API)
            cos, sin = attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # CAPTURE (preserve native device for multi-GPU)
            tensor_device = query_states.device
            adapter.capture.query_states[layer_idx] = query_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.key_states[layer_idx] = key_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2).detach().to(device=tensor_device, dtype=adapter.dtype)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn.layer_idx, cache_kwargs
                )

            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            v_flat = value_states.transpose(1, 2).reshape(bsz, q_len, -1)
            adapter.capture.value_states[layer_idx] = v_flat.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            adapter.capture.attention_weights[layer_idx] = attn_weights.detach().to(device=tensor_device, dtype=adapter.dtype)

            attn_weights = F.dropout(attn_weights, p=attn.attention_dropout, training=attn.training)
            attn_output = torch.matmul(attn_weights, value_states)

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, -1)

            attn_output = attn.o_proj(attn_output)

            # CAPTURE AFTER o_proj to match block_outputs hidden_size
            adapter.capture.attn_outputs[layer_idx] = attn_output.detach().to(device=tensor_device, dtype=adapter.dtype)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value

        attn.forward = patched_forward

    @torch.inference_mode()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_flash_attn: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Any]:
        """
        Run forward pass and extract Q/K stacks for matrix-free centrality.

        Args:
            input_ids: (B, S) input token IDs
            attention_mask: (B, S) attention mask
            use_flash_attn: Enable Flash Attention context

        Returns:
            Q_stack: (B, L*H, S, D) stacked queries
            K_stack: (B, L*H, S, D) stacked keys
            value_norms: Dict[layer_idx, (B, H, S)]
            model_output: Model forward output
        """
        if not self._is_registered:
            self.register()

        self.capture.clear()

        input_ids = input_ids.to(self.device, non_blocking=True)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device, non_blocking=True)

        # Forward pass
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

        # Stack Q/K with GQA expansion
        Q_list = []
        K_list = []
        V_norms_expanded = {}

        for layer_idx in sorted(self.layers):
            if layer_idx in self.capture.query_states:
                Q = self.capture.query_states[layer_idx]
                K = self.capture.key_states[layer_idx]

                # GQA expansion
                if self.heads_per_group > 1:
                    K = K.repeat_interleave(self.heads_per_group, dim=1)

                    if layer_idx in self.capture.value_norms:
                        v_norms = self.capture.value_norms[layer_idx]
                        V_norms_expanded[layer_idx] = v_norms.repeat_interleave(
                            self.heads_per_group, dim=1
                        )

                Q_list.append(Q)
                K_list.append(K)

        if not Q_list:
            raise RuntimeError("No Q/K states captured. Ensure hooks are registered.")

        Q_stack = torch.cat(Q_list, dim=1)
        K_stack = torch.cat(K_list, dim=1)
        value_norms_out = V_norms_expanded if V_norms_expanded else dict(self.capture.value_norms)

        return Q_stack, K_stack, value_norms_out, output

    def __del__(self):
        self.cleanup()
