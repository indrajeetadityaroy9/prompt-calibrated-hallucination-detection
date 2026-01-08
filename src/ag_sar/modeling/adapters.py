"""
Attention Patching Adapters for AG-SAR.

Provides architecture-specific adapters for capturing post-RoPE Q/K tensors.
Supports transformers 4.40.x - 4.44.x (Llama, Qwen, Mistral).
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Any, Dict
import torch
import torch.nn as nn


class UnsupportedVersionError(Exception):
    """Raised when transformers version is not supported."""
    pass


def get_transformers_version() -> Tuple[int, int, int]:
    """Get transformers version as (major, minor, patch)."""
    try:
        import transformers
        parts = transformers.__version__.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch_str = parts[2].split('+')[0].split('.')[0] if len(parts) > 2 else "0"
        patch = int(''.join(c for c in patch_str if c.isdigit()) or "0")
        return (major, minor, patch)
    except (ImportError, ValueError, IndexError):
        return (4, 40, 0)


def check_version_supported(version: Tuple[int, int, int]) -> None:
    """Check if transformers version is supported."""
    major, minor, _ = version
    if major >= 5 or (major == 4 and minor >= 45):
        raise UnsupportedVersionError(
            f"transformers {major}.{minor}.x not supported. Need >= 4.40.0, < 4.45.0"
        )
    if major == 4 and minor < 40:
        raise UnsupportedVersionError(
            f"transformers {major}.{minor}.x too old. Need >= 4.40.0"
        )


class AttentionPatchAdapter(ABC):
    """Base class for architecture-specific attention patching."""

    def __init__(self, architecture: str, version: Tuple[int, int, int]):
        self.architecture = architecture
        self.version = version

    @abstractmethod
    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        """Get (apply_rotary_pos_emb, repeat_kv) functions."""
        pass

    @abstractmethod
    def compute_rope(self, attn: nn.Module, value_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE (cos, sin) embeddings."""
        pass

    @abstractmethod
    def apply_rope(self, query_states: torch.Tensor, key_states: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor,
                   position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to Q/K."""
        pass

    def update_cache(self, past_key_value: Any, key_states: torch.Tensor,
                     value_states: torch.Tensor, layer_idx: int,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache if present."""
        if past_key_value is None:
            return key_states, value_states
        cache_kwargs = {"sin": sin, "cos": cos}
        if hasattr(past_key_value, 'update'):
            return past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states


class LlamaAdapter(AttentionPatchAdapter):
    """Llama attention adapter for transformers 4.40.x - 4.44.x."""

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("llama", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(self, attn: nn.Module, value_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return attn.rotary_emb(value_states, position_ids)

    def apply_rope(self, query_states: torch.Tensor, key_states: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor,
                   position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


class QwenAdapter(AttentionPatchAdapter):
    """Qwen2 attention adapter for transformers 4.40.x - 4.44.x."""

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("qwen", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(self, attn: nn.Module, value_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len is None:
            seq_len = value_states.shape[-2]
        return attn.rotary_emb(value_states, seq_len=seq_len)

    def apply_rope(self, query_states: torch.Tensor, key_states: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor,
                   position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


class MistralAdapter(AttentionPatchAdapter):
    """Mistral attention adapter for transformers 4.40.x - 4.44.x."""

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("mistral", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(self, attn: nn.Module, value_states: torch.Tensor,
                     position_ids: Optional[torch.Tensor] = None,
                     seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return attn.rotary_emb(value_states, position_ids)

    def apply_rope(self, query_states: torch.Tensor, key_states: torch.Tensor,
                   cos: torch.Tensor, sin: torch.Tensor,
                   position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


_ADAPTERS = {
    "llama": LlamaAdapter,
    "qwen": QwenAdapter,
    "mistral": MistralAdapter,
}


def get_adapter(architecture: str, version: Optional[Tuple[int, int, int]] = None) -> AttentionPatchAdapter:
    """Get adapter for architecture and version."""
    if version is None:
        version = get_transformers_version()

    arch_lower = architecture.lower()
    if arch_lower in ("qwen2", "qwen2_moe"):
        arch_lower = "qwen"
    elif arch_lower == "mixtral":
        arch_lower = "mistral"

    check_version_supported(version)

    if arch_lower not in _ADAPTERS:
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {list(_ADAPTERS.keys())}")

    return _ADAPTERS[arch_lower](version)


def is_version_supported(architecture: str, version: Optional[Tuple[int, int, int]] = None) -> bool:
    """Check if version is supported without raising."""
    try:
        get_adapter(architecture, version)
        return True
    except (UnsupportedVersionError, ValueError):
        return False
