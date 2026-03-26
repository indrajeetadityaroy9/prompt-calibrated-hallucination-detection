from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from ag_sar.config import LayerHiddenStates


def _is_norm(module: nn.Module) -> bool:
    return isinstance(module, nn.LayerNorm) or "norm" in type(module).__name__.lower()


@dataclass
class ModelAdapter:

    layers_path: str
    final_norm_path: str
    lm_head_path: str
    post_attn_norm_attr: str

    @classmethod
    def from_model(cls, model: nn.Module) -> "ModelAdapter":
        lm_head = model.get_output_embeddings()
        lm_head_path = next(n for n, m in model.named_modules() if m is lm_head)

        layers_path, layers = max(
            ((n, m) for n, m in model.named_modules() if isinstance(m, nn.ModuleList)),
            key=lambda x: len(x[1]),
        )

        parent_path = layers_path.rsplit(".", 1)[0] if "." in layers_path else ""
        parent = model.get_submodule(parent_path) if parent_path else model
        norm_children = [
            (f"{parent_path}.{n}" if parent_path else n)
            for n, m in parent.named_children() if _is_norm(m)
        ]
        final_norm_path = norm_children[-1]

        block_norms = [n for n, m in layers[0].named_children() if _is_norm(m)]
        post_attn_norm_attr = block_norms[1]

        return cls(
            layers_path=layers_path,
            final_norm_path=final_norm_path,
            lm_head_path=lm_head_path,
            post_attn_norm_attr=post_attn_norm_attr,
        )

    def get_layers(self, model: nn.Module) -> list[nn.Module]:
        return list(model.get_submodule(self.layers_path))

    def get_final_norm(self, model: nn.Module) -> nn.Module:
        return model.get_submodule(self.final_norm_path)

    def get_lm_head(self, model: nn.Module) -> nn.Module:
        return model.get_submodule(self.lm_head_path)

    def get_post_attn_norm(self, layer: nn.Module) -> nn.Module:
        return layer.get_submodule(self.post_attn_norm_attr)


class LayerHooks:

    def __init__(self, layer_idx: int, store: dict, adapter: ModelAdapter):
        self.layer_idx = layer_idx
        self._store = store
        self.adapter = adapter
        self._h_resid_attn: Tensor | None = None
        self._handles: list = []

    def install(self, layer):
        post_attn_norm = self.adapter.get_post_attn_norm(layer)

        h1 = post_attn_norm.register_forward_pre_hook(self._capture_resid_attn)
        self._handles.append(h1)

        h2 = layer.register_forward_hook(self._capture_resid_mlp_and_store)
        self._handles.append(h2)

    def _capture_resid_attn(self, module, args):
        self._h_resid_attn = args[0]

    def _capture_resid_mlp_and_store(self, module, args, output):
        self._store[self.layer_idx] = LayerHiddenStates(
            h_resid_attn=self._h_resid_attn[:, -1, :].detach().bfloat16(),
            h_resid_mlp=output[0][:, -1, :].detach().bfloat16(),
        )
        self._h_resid_attn = None

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
