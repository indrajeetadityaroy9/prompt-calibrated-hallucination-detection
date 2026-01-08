"""
Model Hooking for Attention Extraction.

Captures post-RoPE Q/K states from transformer models via monkey-patching.
Supports GPT-2, Llama, Qwen, and Mistral architectures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import get_transformers_version, get_adapter, UnsupportedVersionError

_TRANSFORMERS_VERSION = get_transformers_version()


@dataclass
class AttentionCapture:
    """Container for captured attention data."""
    query_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    key_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    value_norms: Dict[int, torch.Tensor] = field(default_factory=dict)
    value_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    attn_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    block_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    attention_weights: Dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        for d in [self.query_states, self.key_states, self.value_norms,
                  self.value_states, self.attn_outputs, self.block_outputs,
                  self.attention_weights]:
            d.clear()


class ModelAdapter:
    """Unified adapter for extracting attention data from LLM architectures."""

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.dtype = dtype or next(model.parameters()).dtype
        self.device = next(model.parameters()).device

        self.architecture = self._detect_architecture()
        self._setup_architecture()

        self.layers = layers if layers is not None else list(range(self.num_layers))
        self.capture = AttentionCapture()
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._original_forwards: Dict[int, Callable] = {}
        self._is_registered = False

    def _detect_architecture(self) -> str:
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            if hasattr(self.model.transformer.h[0].attn, 'c_attn'):
                return "gpt2"

        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            if hasattr(self.model.model.layers[0].self_attn, 'q_proj'):
                return "llama"

        model_type = getattr(self.model.config, 'model_type', '').lower()
        if 'gpt2' in model_type:
            return "gpt2"
        if any(arch in model_type for arch in ['llama', 'qwen', 'mistral', 'mixtral', 'moe']):
            return "llama"

        raise ValueError(f"Unknown architecture: {model_type}")

    def _setup_architecture(self):
        if self.architecture == "gpt2":
            self.backbone = self.model.transformer if hasattr(self.model, 'transformer') else self.model
            self.num_layers = len(self.backbone.h)
            self.num_heads = self.backbone.config.n_head
            self.hidden_size = self.backbone.config.n_embd
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = self.num_heads
        else:
            self.backbone = self.model.model if hasattr(self.model, 'model') else self.model
            self.num_layers = len(self.backbone.layers)
            self.num_heads = self.backbone.config.num_attention_heads
            self.hidden_size = self.backbone.config.hidden_size
            self.head_dim = self.hidden_size // self.num_heads
            self.num_kv_heads = getattr(self.backbone.config, 'num_key_value_heads', self.num_heads)

        self.heads_per_group = self.num_heads // self.num_kv_heads

    def register(self) -> None:
        if self._is_registered:
            return
        self.cleanup()

        if self.architecture == "gpt2":
            self._register_gpt2_hooks()
        else:
            self._register_llama_hooks()

        self._is_registered = True

    def cleanup(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

        if self.architecture != "gpt2":
            for layer_idx, original in self._original_forwards.items():
                self.backbone.layers[layer_idx].self_attn.forward = original
        self._original_forwards.clear()
        self._is_registered = False

    def _register_gpt2_hooks(self):
        for layer_idx in self.layers:
            block = self.backbone.h[layer_idx]
            handle = block.attn.c_attn.register_forward_hook(self._make_gpt2_qkv_hook(layer_idx))
            self._hooks.append(handle)
            handle = block.attn.register_forward_hook(self._make_attn_output_hook(layer_idx))
            self._hooks.append(handle)
            handle = block.register_forward_hook(self._make_block_hook(layer_idx))
            self._hooks.append(handle)

    def _register_llama_hooks(self):
        model_type = getattr(self.model.config, 'model_type', '').lower()
        arch_name = "qwen" if 'qwen' in model_type else "mistral" if 'mistral' in model_type or 'mixtral' in model_type else "llama"

        try:
            self._arch_adapter = get_adapter(arch_name, _TRANSFORMERS_VERSION)
        except UnsupportedVersionError as e:
            raise UnsupportedVersionError(e.version, e.architecture, str(e))

        apply_rope, repeat_kv = self._arch_adapter.get_rope_functions()

        for layer_idx in self.layers:
            self._patch_attention(layer_idx, apply_rope, repeat_kv, arch_name)
            handle = self.backbone.layers[layer_idx].register_forward_hook(self._make_block_hook(layer_idx))
            self._hooks.append(handle)

    def _patch_attention(self, layer_idx: int, apply_rope, repeat_kv, arch_name: str):
        """Unified attention patching for Llama/Qwen/Mistral."""
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
            position_embeddings: Optional[tuple] = None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()

            query_states = attn.q_proj(hidden_states)
            key_states = attn.k_proj(hidden_states)
            value_states = attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)

            # Apply RoPE
            if position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                if position_ids is None:
                    past_length = past_key_value.get_seq_length() if past_key_value and hasattr(past_key_value, 'get_seq_length') else 0
                    position_ids = torch.arange(past_length, past_length + q_len, device=hidden_states.device).unsqueeze(0)

                if arch_name == "qwen":
                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None and hasattr(attn, 'layer_idx'):
                        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, attn.layer_idx)
                    cos, sin = attn.rotary_emb(value_states, seq_len=kv_seq_len)
                else:
                    cos, sin = attn.rotary_emb(value_states, position_ids)

            if arch_name == "qwen":
                query_states, key_states = apply_rope(query_states, key_states, cos, sin, position_ids)
            else:
                query_states, key_states = apply_rope(query_states, key_states, cos, sin)

            # Capture post-RoPE states
            tensor_device = query_states.device
            adapter.capture.query_states[layer_idx] = query_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.key_states[layer_idx] = key_states.detach().to(device=tensor_device, dtype=adapter.dtype)
            adapter.capture.value_norms[layer_idx] = torch.norm(value_states, dim=-1, p=2).detach()

            # Update cache
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                cache_layer_idx = getattr(attn, 'layer_idx', layer_idx)
                key_states, value_states = past_key_value.update(key_states, value_states, cache_layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, attn.num_key_value_groups)
            value_states = repeat_kv(value_states, attn.num_key_value_groups)

            adapter.capture.value_states[layer_idx] = value_states.transpose(1, 2).reshape(bsz, q_len, -1).detach()

            # Compute attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn.head_dim)

            kv_seq_len = key_states.size(2)
            if attention_mask is not None and attention_mask.shape[-1] == kv_seq_len and attention_mask.shape[-2] == q_len:
                attn_weights = attn_weights + attention_mask
            else:
                causal_mask = torch.triu(torch.full((q_len, kv_seq_len), float('-inf'), device=tensor_device, dtype=attn_weights.dtype), diagonal=kv_seq_len - q_len + 1)
                attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            adapter.capture.attention_weights[layer_idx] = attn_weights.detach()

            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = attn.o_proj(attn_output)

            adapter.capture.attn_outputs[layer_idx] = attn_output.detach()

            return attn_output, None if not output_attentions else attn_weights, past_key_value

        attn.forward = patched_forward

    def _make_gpt2_qkv_hook(self, layer_idx: int):
        def hook_fn(module, input_args, output):
            batch_size, seq_len, _ = output.shape
            q, k, v = output.split(self.hidden_size, dim=-1)

            self.capture.value_states[layer_idx] = v.detach()

            q_heads = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_heads = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_heads = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            self.capture.query_states[layer_idx] = q_heads.detach()
            self.capture.key_states[layer_idx] = k_heads.detach()
            self.capture.value_norms[layer_idx] = torch.norm(v_heads, dim=-1, p=2).detach()

            attn_weights = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=output.device, dtype=torch.bool), diagonal=1)
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(self.dtype)
            self.capture.attention_weights[layer_idx] = attn_weights.detach()

        return hook_fn

    def _make_attn_output_hook(self, layer_idx: int):
        def hook_fn(module, input_args, output):
            if isinstance(output, tuple):
                attn_output = output[0]
                if isinstance(attn_output, torch.Tensor) and attn_output.dim() == 3:
                    self.capture.attn_outputs[layer_idx] = attn_output.detach()
        return hook_fn

    def _make_block_hook(self, layer_idx: int):
        def hook_fn(module, input_args, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                self.capture.block_outputs[layer_idx] = hidden_states.detach()
        return hook_fn

    @torch.inference_mode()
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Any]:
        """Run forward pass and extract Q/K stacks."""
        if not self._is_registered:
            self.register()

        self.capture.clear()

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, return_dict=True)

        Q_list, K_list = [], []
        for layer_idx in sorted(self.layers):
            if layer_idx in self.capture.query_states:
                Q = self.capture.query_states[layer_idx]
                K = self.capture.key_states[layer_idx]
                if self.heads_per_group > 1:
                    K = K.repeat_interleave(self.heads_per_group, dim=1)
                Q_list.append(Q)
                K_list.append(K)

        if not Q_list:
            raise RuntimeError("No Q/K states captured")

        return torch.cat(Q_list, dim=1), torch.cat(K_list, dim=1), dict(self.capture.value_norms), output

    def __del__(self):
        self.cleanup()
