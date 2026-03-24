from dataclasses import dataclass

import torch.nn as nn


_ARCH_PATTERNS = [
    ("model.layers", "model.norm", "lm_head", "post_attention_layernorm"),           # LLaMA/Mistral/Qwen/Gemma
    ("model.layers", "model.final_layernorm", "lm_head", "post_attention_layernorm"),# Phi
    ("transformer.h", "transformer.ln_f", "lm_head", "ln_2"),                        # GPT-2/GPT-Neo
    ("transformer.h", "transformer.ln_f", "lm_head", "post_attention_layernorm"),    # Falcon
    ("gpt_neox.layers", "gpt_neox.final_layer_norm", "embed_out", "post_attention_layernorm"),  # GPT-NeoX/Pythia
]


@dataclass
class ModelAdapter:

    post_attn_norm_attr: str = "post_attention_layernorm"
    layers_path: str = "model.layers"
    final_norm_path: str = "model.norm"
    lm_head_path: str = "lm_head"

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelAdapter":
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
        return list(model.get_submodule(self.layers_path))

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.get_submodule(self.final_norm_path)

    def get_lm_head(self, model: nn.Module) -> nn.Module:
        return model.get_submodule(self.lm_head_path)

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule(self.post_attn_norm_attr)
