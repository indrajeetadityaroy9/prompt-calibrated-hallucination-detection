"""
Version-Adaptive Attention Patching for AG-SAR.

This module provides architecture and version-specific adapters for attention
monkey-patching. Standard PyTorch hooks cannot access post-RoPE Q/K tensors,
so we must patch the attention.forward method directly.

Supported transformers versions:
- 4.40.x - 4.44.x: Full support (Llama, Qwen, Mistral)
- 4.45.x+: UNSUPPORTED (breaking API changes)
- 5.x: UNSUPPORTED (major version, untested)

Usage:
    version = get_transformers_version()
    try:
        adapter = get_adapter("llama", version)
        patched_forward = adapter.create_patched_forward(...)
    except UnsupportedVersionError as e:
        # Handle gracefully
"""

from abc import ABC, abstractmethod
from typing import Tuple, Callable, Optional, Any, Dict
import warnings

import torch
import torch.nn as nn


class UnsupportedVersionError(Exception):
    """Raised when transformers version is not supported."""

    def __init__(self, version: Tuple[int, int, int], architecture: str, message: str = ""):
        self.version = version
        self.architecture = architecture
        super().__init__(
            message or f"transformers {'.'.join(map(str, version))} is not supported for {architecture}. "
            f"AG-SAR requires transformers >= 4.40.0, < 4.45.0."
        )


def get_transformers_version() -> Tuple[int, int, int]:
    """
    Get transformers library version as tuple.

    Returns:
        (major, minor, patch) version tuple

    Example:
        >>> get_transformers_version()
        (4, 42, 0)
    """
    try:
        import transformers
        parts = transformers.__version__.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        # Handle patch versions like "0.dev0" or "1+local"
        patch_str = parts[2].split('+')[0].split('.')[0] if len(parts) > 2 else "0"
        patch = int(''.join(c for c in patch_str if c.isdigit()) or "0")
        return (major, minor, patch)
    except (ImportError, ValueError, IndexError):
        return (0, 0, 0)


def check_version_supported(version: Tuple[int, int, int], architecture: str) -> None:
    """
    Check if transformers version is supported for the given architecture.

    Args:
        version: (major, minor, patch) version tuple
        architecture: Model architecture ("llama", "qwen", "mistral")

    Raises:
        UnsupportedVersionError: If version is not supported
    """
    major, minor, _ = version

    if major >= 5:
        raise UnsupportedVersionError(
            version, architecture,
            f"transformers {major}.x is not supported. AG-SAR requires transformers >= 4.40.0, < 4.45.0. "
            f"The attention hook mechanism has breaking changes in newer versions."
        )

    if major == 4 and minor >= 45:
        raise UnsupportedVersionError(
            version, architecture,
            f"transformers 4.{minor}.x has breaking changes for attention hooks. "
            f"Please downgrade to transformers >= 4.40.0, < 4.45.0."
        )

    if major == 4 and minor < 40:
        raise UnsupportedVersionError(
            version, architecture,
            f"transformers 4.{minor}.x is too old. AG-SAR requires transformers >= 4.40.0."
        )


class AttentionPatchAdapter(ABC):
    """
    Abstract base class for architecture-specific attention patching.

    Each adapter provides:
    1. RoPE function imports for the architecture
    2. Methods to apply RoPE correctly
    3. Cache update handling
    4. Patched forward method creation

    Why monkey-patching?
        Standard PyTorch hooks (register_forward_hook, register_forward_pre_hook)
        only have access to module inputs and outputs, not intermediate tensors.
        For AG-SAR, we need post-RoPE Q/K states, which are computed INSIDE
        the attention forward method. Monkey-patching is the only way to
        capture these intermediate values.
    """

    def __init__(self, architecture: str, version: Tuple[int, int, int]):
        self.architecture = architecture
        self.version = version

    @abstractmethod
    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        """
        Get RoPE application and KV repeat functions for this architecture.

        Returns:
            (apply_rotary_pos_emb, repeat_kv) callables
        """
        pass

    @abstractmethod
    def compute_rope(
        self,
        attn: nn.Module,
        value_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cos/sin embeddings.

        Different architectures and versions use different APIs:
        - Llama 4.40+: rotary_emb(value_states, position_ids)
        - Qwen 4.40-4.44: rotary_emb(value_states, seq_len=kv_seq_len)

        Args:
            attn: Attention module with rotary_emb
            value_states: Value tensor (used for device/dtype)
            position_ids: Position IDs (Llama-style)
            seq_len: Sequence length (Qwen-style)

        Returns:
            (cos, sin) tensors for RoPE
        """
        pass

    @abstractmethod
    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to Q/K.

        Args:
            query_states: Query tensor
            key_states: Key tensor
            cos: Cosine embeddings
            sin: Sine embeddings
            position_ids: Position IDs (some versions require this)

        Returns:
            (rotated_query, rotated_key) tensors
        """
        pass

    @abstractmethod
    def update_cache(
        self,
        past_key_value: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV cache if present.

        Args:
            past_key_value: Cache object (may be None)
            key_states: Key tensor
            value_states: Value tensor
            layer_idx: Layer index for cache
            cos, sin: RoPE embeddings for cache kwargs

        Returns:
            (updated_key_states, updated_value_states)
        """
        pass

    def get_hidden_size(self, attn: nn.Module) -> int:
        """Get hidden size from attention module."""
        if hasattr(attn, 'hidden_size'):
            return attn.hidden_size
        # Fallback: compute from num_heads * head_dim
        return attn.num_heads * attn.head_dim


class LlamaAdapter_v4_40(AttentionPatchAdapter):
    """
    Llama attention adapter for transformers 4.40.x - 4.44.x.

    Key characteristics:
    - rotary_emb(value_states, position_ids) - uses position_ids directly
    - apply_rotary_pos_emb(q, k, cos, sin) - no position_ids in apply
    - Cache update via past_key_value.update()
    """

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("llama", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(
        self,
        attn: nn.Module,
        value_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Llama uses position_ids directly
        return attn.rotary_emb(value_states, position_ids)

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def update_cache(
        self,
        past_key_value: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return key_states, value_states

        cache_kwargs = {"sin": sin, "cos": cos}
        if hasattr(past_key_value, 'update'):
            return past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)
        return key_states, value_states


class QwenAdapter_v4_40(AttentionPatchAdapter):
    """
    Qwen2 attention adapter for transformers 4.40.x - 4.44.x.

    Key characteristics:
    - rotary_emb(value_states, seq_len=kv_seq_len) - uses seq_len kwarg
    - apply_rotary_pos_emb(q, k, cos, sin, position_ids) - needs position_ids
    - Uses attn.layer_idx for cache operations
    - Uses attn.hidden_size for output reshape
    """

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("qwen", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(
        self,
        attn: nn.Module,
        value_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Qwen uses seq_len kwarg
        if seq_len is None:
            seq_len = value_states.shape[-2]
        return attn.rotary_emb(value_states, seq_len=seq_len)

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    def update_cache(
        self,
        past_key_value: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return key_states, value_states

        cache_kwargs = {"sin": sin, "cos": cos}
        return past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)


class MistralAdapter_v4_40(AttentionPatchAdapter):
    """
    Mistral attention adapter for transformers 4.40.x - 4.44.x.

    Key characteristics:
    - Same as Llama: rotary_emb(value_states, position_ids)
    - apply_rotary_pos_emb(q, k, cos, sin) - no position_ids
    - Uses attn.layer_idx for cache
    """

    def __init__(self, version: Tuple[int, int, int]):
        super().__init__("mistral", version)

    def get_rope_functions(self) -> Tuple[Callable, Callable]:
        from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv
        return apply_rotary_pos_emb, repeat_kv

    def compute_rope(
        self,
        attn: nn.Module,
        value_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mistral uses position_ids like Llama
        return attn.rotary_emb(value_states, position_ids)

    def apply_rope(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        apply_rotary_pos_emb, _ = self.get_rope_functions()
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)

    def update_cache(
        self,
        past_key_value: Any,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past_key_value is None:
            return key_states, value_states

        cache_kwargs = {"sin": sin, "cos": cos}
        return past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)


# ============================================================================
# Stub adapters for future versions (raise clear errors)
# ============================================================================

class LlamaAdapter_v4_45(AttentionPatchAdapter):
    """
    STUB: Llama adapter for transformers 4.45.x.

    This version has breaking API changes. Not yet implemented.
    """

    def __init__(self, version: Tuple[int, int, int]):
        raise UnsupportedVersionError(
            version, "llama",
            "transformers 4.45.x has breaking attention API changes. "
            "Downgrade to transformers >= 4.40.0, < 4.45.0, or wait for AG-SAR update."
        )

    def get_rope_functions(self):
        raise NotImplementedError("Not supported for 4.45.x")

    def compute_rope(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 4.45.x")

    def apply_rope(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 4.45.x")

    def update_cache(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 4.45.x")


class LlamaAdapter_v5(AttentionPatchAdapter):
    """
    STUB: Llama adapter for transformers 5.x.

    Major version with potential significant changes. Not yet implemented.
    """

    def __init__(self, version: Tuple[int, int, int]):
        raise UnsupportedVersionError(
            version, "llama",
            "transformers 5.x is not yet supported. "
            "Please use transformers >= 4.40.0, < 4.45.0."
        )

    def get_rope_functions(self):
        raise NotImplementedError("Not supported for 5.x")

    def compute_rope(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 5.x")

    def apply_rope(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 5.x")

    def update_cache(self, *args, **kwargs):
        raise NotImplementedError("Not supported for 5.x")


# ============================================================================
# Adapter Factory
# ============================================================================

_ADAPTER_REGISTRY: Dict[str, Dict[Tuple[int, int], type]] = {
    "llama": {
        (4, 40): LlamaAdapter_v4_40,
        (4, 41): LlamaAdapter_v4_40,
        (4, 42): LlamaAdapter_v4_40,
        (4, 43): LlamaAdapter_v4_40,
        (4, 44): LlamaAdapter_v4_40,
        (4, 45): LlamaAdapter_v4_45,  # Stub
        (5, 0): LlamaAdapter_v5,       # Stub
    },
    "qwen": {
        (4, 40): QwenAdapter_v4_40,
        (4, 41): QwenAdapter_v4_40,
        (4, 42): QwenAdapter_v4_40,
        (4, 43): QwenAdapter_v4_40,
        (4, 44): QwenAdapter_v4_40,
    },
    "mistral": {
        (4, 40): MistralAdapter_v4_40,
        (4, 41): MistralAdapter_v4_40,
        (4, 42): MistralAdapter_v4_40,
        (4, 43): MistralAdapter_v4_40,
        (4, 44): MistralAdapter_v4_40,
    },
}


def get_adapter(architecture: str, version: Optional[Tuple[int, int, int]] = None) -> AttentionPatchAdapter:
    """
    Get the appropriate adapter for an architecture and transformers version.

    Args:
        architecture: Model architecture ("llama", "qwen", "mistral")
        version: (major, minor, patch) tuple. Auto-detected if None.

    Returns:
        AttentionPatchAdapter instance

    Raises:
        UnsupportedVersionError: If version is not supported
        ValueError: If architecture is unknown

    Example:
        >>> adapter = get_adapter("llama")
        >>> apply_rope, repeat_kv = adapter.get_rope_functions()
    """
    if version is None:
        version = get_transformers_version()

    # Normalize architecture name
    arch_lower = architecture.lower()
    if arch_lower in ("qwen2", "qwen2_moe"):
        arch_lower = "qwen"
    elif arch_lower in ("mixtral",):
        arch_lower = "mistral"

    if arch_lower not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Supported: {list(_ADAPTER_REGISTRY.keys())}"
        )

    # Check version compatibility
    check_version_supported(version, arch_lower)

    # Find matching adapter
    major, minor, _ = version
    version_key = (major, minor)

    arch_adapters = _ADAPTER_REGISTRY[arch_lower]
    if version_key not in arch_adapters:
        # Try to find closest supported version
        supported = sorted(arch_adapters.keys())
        raise UnsupportedVersionError(
            version, arch_lower,
            f"No adapter for transformers {major}.{minor}.x. "
            f"Supported minor versions: {[f'{m}.{n}' for m, n in supported]}"
        )

    adapter_class = arch_adapters[version_key]
    return adapter_class(version)


def is_version_supported(architecture: str, version: Optional[Tuple[int, int, int]] = None) -> bool:
    """
    Check if a version is supported without raising an exception.

    Args:
        architecture: Model architecture
        version: Version tuple (auto-detected if None)

    Returns:
        True if supported, False otherwise
    """
    try:
        get_adapter(architecture, version)
        return True
    except (UnsupportedVersionError, ValueError):
        return False
