"""Prefill-phase hooks for context capture and prompt statistics."""

from dataclasses import dataclass
from typing import Dict, List
import torch
from torch import Tensor

from .adapter import ModelAdapter


class PrefillContextHook:
    """
    Temporary hook installed ONLY during prefill to capture context embeddings.

    Captures the full sequence output at one layer, slicing context positions.
    Preserves the "single decoding pass" claim by NOT running a second forward.
    """

    def __init__(
        self,
        context_start: int,
        context_end: int,
        buffer_ref: List[Tensor],
    ):
        self.context_start = context_start
        self.context_end = context_end
        self.buffer_ref = buffer_ref
        self._handle = None

    def install(self, layer):
        """Install hook on the specified layer."""
        self._handle = layer.register_forward_hook(self._capture_context)

    def _capture_context(self, module, args, output):
        """Capture context positions from this layer's output during prefill."""
        context_hidden = output[0][0, self.context_start:self.context_end, :]
        self.buffer_ref.append(context_hidden.detach().bfloat16())

    def remove(self):
        """Remove the hook."""
        self._handle.remove()
        self._handle = None


@dataclass
class PrefillStatistics:
    """Statistics computed from prompt tail for prompt-anchored normalization."""
    sigma: float
    raw_values: 'np.ndarray'


class PrefillStatisticsHook:
    """Captures prompt-tail hidden states → candidate-restricted JSD statistics for PIT normalization."""

    def __init__(
        self,
        lm_head: torch.nn.Module,
        final_norm: torch.nn.Module,
        window_size: int = None,
        *,
        adapter: ModelAdapter,
    ):
        self.lm_head = lm_head
        self.final_norm = final_norm
        # Default: minimum for stable order statistics (median of 4 values).
        # Always overridden by adaptive_window() in Detector._prefill().
        self.window_size = window_size if window_size is not None else 4
        self.adapter = adapter

        self._captured_hidden: Tensor = None  # type: ignore[assignment]
        self._pre_mlp_hidden: Tensor = None  # type: ignore[assignment]
        self._handle = None
        self._pre_handle = None

    def install(self, layer):
        """Install hooks to capture both pre-MLP and post-MLP states."""
        post_attn_norm = self.adapter.get_post_attn_norm(layer)
        self._pre_handle = post_attn_norm.register_forward_pre_hook(
            self._capture_pre_mlp
        )
        self._handle = layer.register_forward_hook(self._capture_hidden)

    def _capture_pre_mlp(self, module, args):
        """Capture input to post_attention_layernorm (pre-MLP residual stream)."""
        self._pre_mlp_hidden = args[0].detach()

    def _capture_hidden(self, module, args, output):
        """Capture full sequence hidden states during prefill."""
        self._captured_hidden = output[0].detach()

    def remove(self):
        """Remove installed hooks."""
        self._handle.remove()
        self._handle = None
        self._pre_handle.remove()
        self._pre_handle = None

    def compute_statistics(self, prompt_length: int) -> PrefillStatistics:
        """Tail-sampled candidate-restricted JSD statistics with adaptive top-k per position."""
        import torch.nn.functional as F
        import numpy as np
        from ..numerics import safe_jsd, entropy_adaptive_nucleus

        tail_start = max(0, prompt_length - self.window_size)
        tail_end = prompt_length

        tail_hidden = self._captured_hidden[:, tail_start:tail_end, :]

        lm_head_device = next(self.lm_head.parameters()).device
        lm_head_dtype = next(self.lm_head.parameters()).dtype

        with torch.no_grad():
            tail_hidden_moved = tail_hidden.to(device=lm_head_device, dtype=lm_head_dtype)
            normalized = self.final_norm(tail_hidden_moved)
            tail_logits = self.lm_head(normalized)

        pre_mlp_tail = self._pre_mlp_hidden[:, tail_start:tail_end, :]

        with torch.no_grad():
            pre_mlp_moved = pre_mlp_tail.to(device=lm_head_device, dtype=lm_head_dtype)
            pre_normalized = self.final_norm(pre_mlp_moved)
            pre_logits = self.lm_head(pre_normalized).float()

        jsd_values = []
        for pos in range(tail_logits.shape[1]):
            post_logits_pos = tail_logits[0, pos].float()
            pre_logits_pos = pre_logits[0, pos]

            post_probs_full = F.softmax(post_logits_pos, dim=-1)
            sorted_probs, sorted_indices = torch.sort(post_probs_full, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            mass = entropy_adaptive_nucleus(post_probs_full)
            k = int((cumsum < mass).sum().item()) + 1
            k = max(2, k)

            topk_indices = sorted_indices[:k]

            post_cand = F.softmax(post_logits_pos[topk_indices], dim=-1)
            pre_cand = F.softmax(pre_logits_pos[topk_indices], dim=-1)

            jsd_values.append(safe_jsd(pre_cand, post_cand))

        values = np.array(jsd_values)

        self._captured_hidden = None  # type: ignore[assignment]
        self._pre_mlp_hidden = None  # type: ignore[assignment]

        return PrefillStatistics(
            sigma=float(np.std(values)),
            raw_values=np.sort(values),
        )
