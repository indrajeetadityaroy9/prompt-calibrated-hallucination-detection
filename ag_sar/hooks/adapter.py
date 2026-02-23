"""Architecture adapter for accessing model-specific attributes."""

from dataclasses import dataclass
from typing import Optional


def _resolve_path(obj, dot_path: str):
    """Traverse a dot-separated attribute path (e.g. 'model.layers')."""
    for attr in dot_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _has_path(obj, dot_path: str) -> bool:
    """Check if a dot-separated attribute path exists."""
    try:
        _resolve_path(obj, dot_path)
        return True
    except AttributeError:
        return False


# Architecture patterns: (layers_path, final_norm_path, lm_head_path, post_attn_norm_attr)
_ARCH_PATTERNS = [
    # LLaMA / Mistral / Qwen / Gemma
    ("model.layers", "model.norm", "lm_head", "post_attention_layernorm"),
    # Phi-3 / Phi-2
    ("model.layers", "model.norm", "lm_head", "post_layernorm"),
    # Phi-1 / Phi-1.5 (MixFormer)
    ("model.layers", "model.final_layernorm", "lm_head", "post_attention_layernorm"),
    # GPT-2 / GPT-Neo
    ("transformer.h", "transformer.ln_f", "lm_head", "ln_2"),
    # Falcon
    ("transformer.h", "transformer.ln_f", "lm_head", "post_attention_layernorm"),
    # GPT-NeoX / Pythia
    ("gpt_neox.layers", "gpt_neox.final_layer_norm", "embed_out", "post_attention_layernorm"),
]


@dataclass
class ModelAdapter:
    """Architecture adapter for accessing model components across model families."""
    post_attn_norm_attr: str = "post_attention_layernorm"
    layers_path: str = "model.layers"
    final_norm_path: str = "model.norm"
    lm_head_path: str = "lm_head"

    @classmethod
    def from_model(cls, model) -> "ModelAdapter":
        """Auto-detect architecture by probing known patterns."""
        for layers_path, norm_path, head_path, norm_attr in _ARCH_PATTERNS:
            if not _has_path(model, layers_path):
                continue
            if not _has_path(model, norm_path):
                continue
            if not _has_path(model, head_path):
                continue
            layers = _resolve_path(model, layers_path)
            if len(layers) == 0:
                continue
            if not hasattr(layers[0], norm_attr):
                continue
            return cls(
                post_attn_norm_attr=norm_attr,
                layers_path=layers_path,
                final_norm_path=norm_path,
                lm_head_path=head_path,
            )
        raise ValueError(
            f"Unsupported architecture: {type(model).__name__}. "
            f"Could not find matching layer/norm/head pattern."
        )

    def get_layers(self, model) -> list:
        """Get the list of decoder layers."""
        return list(_resolve_path(model, self.layers_path))

    def get_final_norm(self, model):
        """Get the final layer norm module."""
        return _resolve_path(model, self.final_norm_path)

    def get_lm_head(self, model):
        """Get the language model head."""
        return _resolve_path(model, self.lm_head_path)

    def get_post_attn_norm(self, layer):
        """Get the post-attention norm module from a layer."""
        return getattr(layer, self.post_attn_norm_attr)
