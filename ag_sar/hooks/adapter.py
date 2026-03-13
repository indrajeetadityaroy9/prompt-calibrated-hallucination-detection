"""Architecture adapter for accessing model-specific attributes via nn.Module.get_submodule()."""

from dataclasses import dataclass

import torch.nn as nn


# Architecture patterns: (layers_path, final_norm_path, lm_head_path, post_attn_norm_attr)
_ARCH_PATTERNS = [
    # LLaMA / Mistral / Qwen / Gemma
    ("model.layers", "model.norm", "lm_head", "post_attention_layernorm"),
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
    """Architecture adapter for accessing model components across model families.

    Uses nn.Module.get_submodule() for dot-path traversal instead of manual
    getattr chains. Raises AttributeError on invalid paths with clear messages.
    """
    post_attn_norm_attr: str = "post_attention_layernorm"
    layers_path: str = "model.layers"
    final_norm_path: str = "model.norm"
    lm_head_path: str = "lm_head"

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelAdapter":
        """Auto-detect architecture by probing known patterns."""
        for layers_path, norm_path, head_path, norm_attr in _ARCH_PATTERNS:
            try:
                layers = model.get_submodule(layers_path)
            except AttributeError:
                continue
            if hasattr(layers[0], norm_attr):
                model.get_submodule(norm_path)
                model.get_submodule(head_path)
                return cls(
                    post_attn_norm_attr=norm_attr,
                    layers_path=layers_path,
                    final_norm_path=norm_path,
                    lm_head_path=head_path,
                )
        raise ValueError(f"Unsupported architecture: {type(model).__name__}")

    def get_layers(self, model: nn.Module) -> list[nn.Module]:
        """Get the list of decoder layers."""
        return list(model.get_submodule(self.layers_path))

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        """Get the final layer norm module."""
        return model.get_submodule(self.final_norm_path)

    def get_lm_head(self, model: nn.Module) -> nn.Module:
        """Get the language model head."""
        return model.get_submodule(self.lm_head_path)

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        """Get the post-attention norm module from a layer."""
        return layer.get_submodule(self.post_attn_norm_attr)
