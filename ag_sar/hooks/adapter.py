"""Architecture adapter for accessing model-specific attributes."""

from dataclasses import dataclass


@dataclass
class ModelAdapter:
    """Architecture adapter for accessing post-attention norm across model families."""
    post_attn_norm_attr: str = "post_attention_layernorm"

    @classmethod
    def from_model(cls, model):
        """Auto-detect the post-attention norm attribute name."""
        layer0 = model.model.layers[0]
        if hasattr(layer0, "post_attention_layernorm"):
            return cls()
        elif hasattr(layer0, "post_layernorm"):
            return cls(post_attn_norm_attr="post_layernorm")
        raise ValueError(f"Unsupported architecture: {type(layer0).__name__}")

    def get_post_attn_norm(self, layer):
        """Get the post-attention norm module from a layer."""
        return getattr(layer, self.post_attn_norm_attr)
