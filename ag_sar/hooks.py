"""
Hook system for capturing hidden states during generation.

Implements 3-point capture per layer:
1. h_resid_attn: Residual after attention add (before post_attn_norm)
2. h_mlp_in: MLP input (post_attn_norm output, normalized)
3. h_resid_mlp: Residual after MLP add (layer output)

Signal semantics:
- JSD(h_resid_attn -> h_resid_mlp): "MLP-induced shift on residual stream"
- JSD(h_mlp_in -> h_resid_mlp): "MLP transformation" (normalized input vs output)
"""

from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
import torch
from torch import Tensor
if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer


@dataclass
class LayerHiddenStates:
    """All 3 capture points for one layer."""

    h_resid_attn: Tensor  # Residual after attention add (before post_attn_norm)
    h_mlp_in: Tensor  # MLP input (post_attn_norm output, normalized)
    h_resid_mlp: Tensor  # Residual after MLP add (layer output)


class EphemeralHiddenBuffer:
    """
    Per-step buffer capturing 3 points per layer.

    Ephemeral: cleared after each token's signals are computed.
    This prevents memory accumulation during generation.

    Usage:
        1. Install hooks before forward pass
        2. Run forward pass (hooks populate buffer)
        3. Call compute_signals with candidate set
        4. Buffer is automatically cleared after compute_signals
    """

    def __init__(self):
        self.layer_states: Dict[int, LayerHiddenStates] = {}

    def store(
        self,
        layer_idx: int,
        h_resid_attn: Tensor,
        h_mlp_in: Tensor,
        h_resid_mlp: Tensor,
    ):
        """
        Store fp16 hidden states (last position only, detached not cloned).

        Args:
            layer_idx: Index of the layer
            h_resid_attn: Residual after attention [batch, seq, hidden]
            h_mlp_in: MLP input after norm [batch, seq, hidden]
            h_resid_mlp: Residual after MLP [batch, seq, hidden]
        """
        self.layer_states[layer_idx] = LayerHiddenStates(
            h_resid_attn=h_resid_attn[:, -1, :].detach().half(),
            h_mlp_in=h_mlp_in[:, -1, :].detach().half(),
            h_resid_mlp=h_resid_mlp[:, -1, :].detach().half(),
        )

    def get_states(self) -> Dict[int, LayerHiddenStates]:
        """Get all captured layer states."""
        return self.layer_states

    def clear(self):
        """Clear all stored states to free memory."""
        self.layer_states.clear()

    def __len__(self) -> int:
        """Number of layers with captured states."""
        return len(self.layer_states)


class LayerHooks:
    """
    Install hooks to capture all 3 points for a single layer.

    Hook architecture for LLaMA:
    - pre_hook on post_attention_layernorm: capture input (h_resid_attn)
    - forward_hook on post_attention_layernorm: capture output (h_mlp_in)
    - forward_hook on decoder layer: capture output (h_resid_mlp)

    The hooks store states to an EphemeralHiddenBuffer.
    """

    def __init__(self, layer_idx: int, buffer: EphemeralHiddenBuffer):
        """
        Initialize hooks for a layer.

        Args:
            layer_idx: Index of the layer to hook
            buffer: Shared buffer to store captured states
        """
        self.layer_idx = layer_idx
        self.buffer = buffer
        self._h_resid_attn: Tensor = None  # type: ignore[assignment]
        self._h_mlp_in: Tensor = None  # type: ignore[assignment]
        self._handles: List = []

    def install(self, layer: "LlamaDecoderLayer"):
        """
        Install hooks on the layer.

        Args:
            layer: LlamaDecoderLayer to hook
        """
        # Hook 1: pre-hook on post_attention_layernorm to get h_resid_attn
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(
            self._capture_resid_attn
        )
        self._handles.append(h1)

        # Hook 2: forward hook on post_attention_layernorm to get h_mlp_in
        h2 = layer.post_attention_layernorm.register_forward_hook(
            self._capture_mlp_in
        )
        self._handles.append(h2)

        # Hook 3: forward hook on layer to get h_resid_mlp
        h3 = layer.register_forward_hook(
            self._capture_resid_mlp_and_store
        )
        self._handles.append(h3)

    def _capture_resid_attn(self, module, args):
        """Pre-hook: input to post_attention_layernorm = h_resid_attn."""
        # args[0] is the input tensor
        self._h_resid_attn = args[0]

    def _capture_mlp_in(self, module, args, output):
        """Forward hook: output of post_attention_layernorm = h_mlp_in."""
        self._h_mlp_in = output

    def _capture_resid_mlp_and_store(self, module, args, output):
        """Forward hook: layer output = h_resid_mlp. Store all 3 to buffer."""
        # Layer output is tuple (hidden_states, ...) or just hidden_states
        h_resid_mlp = output[0] if isinstance(output, tuple) else output

        # Store all 3 points to buffer
        self.buffer.store(
            self.layer_idx,
            self._h_resid_attn,
            self._h_mlp_in,
            h_resid_mlp,
        )

        # Clear temporary references
        self._h_resid_attn = None  # type: ignore[assignment]
        self._h_mlp_in = None  # type: ignore[assignment]

    def remove(self):
        """Remove all installed hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


class PrefillContextHook:
    """
    Temporary hook installed ONLY during prefill to capture context embeddings.

    This is a separate hook from the 3-point decode hooks because:
    - We need the full sequence output (not just last position)
    - We only need one layer (not the full layer subset)
    - We only run it during prefill (not during decode)

    This preserves the "single decoding pass" claim by NOT running a second forward.
    """

    def __init__(
        self,
        context_start: int,
        context_end: int,
        buffer_ref: List[Tensor],
    ):
        """
        Initialize prefill context hook.

        Args:
            context_start: Start index of context tokens
            context_end: End index of context tokens (exclusive)
            buffer_ref: Mutable list to store captured embeddings
        """
        self.context_start = context_start
        self.context_end = context_end
        self.buffer_ref = buffer_ref
        self._handle = None

    def install(self, layer):
        """Install hook on the specified layer."""
        self._handle = layer.register_forward_hook(self._capture_context)

    def _capture_context(self, module, args, output):
        """Capture context positions from this layer's output during prefill."""
        hidden = output[0] if isinstance(output, tuple) else output

        # Slice context positions and store (fp16, detached)
        context_hidden = hidden[0, self.context_start:self.context_end, :]
        self.buffer_ref.append(context_hidden.detach().half())

    def remove(self):
        """Remove the hook."""
        self._handle.remove()
        self._handle = None


@dataclass
class PrefillStatistics:
    """Statistics computed from prompt tail for prompt-anchored normalization."""

    # Per-signal statistics
    mu: Dict[str, float]  # Mean per signal
    sigma: Dict[str, float]  # Std per signal

    # Metadata
    n_tokens: int  # Number of tokens used for statistics
    signals: tuple  # Which signals were computed


class PrefillStatisticsHook:
    """
    Captures hidden states from prompt tail for prompt-anchored normalization.

    Key design decisions:
    1. Only processes last `window_size` tokens (default 64) for efficiency
    2. Captures full sequence in prefill (not just last token)
    3. Computes candidate-restricted JSD statistics
    4. Memory-efficient: stores only statistics, not per-token values
    5. Uses adaptive top-k based on distribution concentration (95% cumulative mass)

    This implements the "Tail Sampling" strategy that provides a robust
    baseline for the "local thermodynamic temperature" right before generation.
    """

    def __init__(
        self,
        lm_head: torch.nn.Module,
        final_norm: torch.nn.Module,
        window_size: int = 64,
    ):
        """
        Initialize prefill statistics hook.

        Args:
            lm_head: Model's language model head for logit projection
            final_norm: Model's final layer norm for proper Logit Lens
            window_size: Number of tail tokens to sample (default 64)
        """
        self.lm_head = lm_head
        self.final_norm = final_norm
        self.window_size = window_size

        # Captured states during prefill
        self._captured_hidden: Tensor = None  # type: ignore[assignment]
        self._pre_mlp_hidden: Tensor = None  # type: ignore[assignment]
        self._handle = None
        self._pre_handle = None

    def install(self, layer):
        """Install hooks to capture both pre-MLP and post-MLP states."""
        # Capture pre-MLP state: the INPUT to post_attention_layernorm
        # This is the residual stream (h_resid_attn) BEFORE normalization,
        # which matches what CandidateJSDSignal uses for JSD computation.
        self._pre_handle = layer.post_attention_layernorm.register_forward_pre_hook(
            self._capture_pre_mlp
        )
        # Capture post-MLP state (layer output)
        self._handle = layer.register_forward_hook(self._capture_hidden)

    def _capture_pre_mlp(self, module, args):
        """Capture input to post_attention_layernorm (pre-MLP residual stream)."""
        # args[0] is the input tensor (residual stream before normalization)
        # This matches h_resid_attn in the 3-point capture for JSD
        self._pre_mlp_hidden = args[0].detach()

    def _capture_hidden(self, module, args, output):
        """Capture full sequence hidden states during prefill."""
        hidden = output[0] if isinstance(output, tuple) else output
        # Store the full sequence (we'll extract tail in compute_statistics)
        self._captured_hidden = hidden.detach()

    def remove(self):
        """Remove installed hooks."""
        self._handle.remove()
        self._handle = None
        self._pre_handle.remove()
        self._pre_handle = None

    def compute_statistics(self, prompt_length: int) -> PrefillStatistics:
        """
        Compute prompt statistics from captured hidden states.

        Uses tail sampling: only the last `window_size` tokens of the prompt.
        Computes candidate-restricted JSD with adaptive top-k per position.

        Args:
            prompt_length: Total number of prompt tokens

        Returns:
            PrefillStatistics with mu and sigma for pos (JSD)
        """
        import torch.nn.functional as F
        import numpy as np
        from .numerics import safe_jsd

        # Determine tail window
        tail_start = max(0, prompt_length - self.window_size)
        tail_end = prompt_length
        n_tokens = tail_end - tail_start

        if n_tokens < 5:
            tail_start = 0

        # Extract tail hidden states [1, window_size, hidden_dim]
        tail_hidden = self._captured_hidden[:, tail_start:tail_end, :]

        # Get device and dtype from lm_head for proper placement
        lm_head_device = next(self.lm_head.parameters()).device
        lm_head_dtype = next(self.lm_head.parameters()).dtype

        # Project to logits using final_norm + lm_head (proper Logit Lens)
        with torch.no_grad():
            tail_hidden_moved = tail_hidden.to(device=lm_head_device, dtype=lm_head_dtype)
            normalized = self.final_norm(tail_hidden_moved)
            tail_logits = self.lm_head(normalized)  # [1, window_size, vocab_size]

        # Also project pre-MLP states for JSD computation
        pre_mlp_tail = self._pre_mlp_hidden[:, tail_start:tail_end, :]

        with torch.no_grad():
            pre_mlp_moved = pre_mlp_tail.to(device=lm_head_device, dtype=lm_head_dtype)
            pre_normalized = self.final_norm(pre_mlp_moved)
            pre_logits = self.lm_head(pre_normalized).float()

        # Candidate-restricted JSD per position with adaptive top-k
        jsd_values = []
        for pos in range(tail_logits.shape[1]):
            post_logits_pos = tail_logits[0, pos].float()  # [vocab]
            pre_logits_pos = pre_logits[0, pos]             # [vocab]

            # Adaptive top-k via 95% cumulative probability mass
            post_probs_full = F.softmax(post_logits_pos, dim=-1)
            sorted_probs, sorted_indices = torch.sort(post_probs_full, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            k = int((cumsum < 0.95).sum().item()) + 1
            k = max(2, k)

            # Get top-k candidates from post-MLP logits
            topk_indices = sorted_indices[:k]

            # Restrict to candidates and re-normalize
            post_cand = F.softmax(post_logits_pos[topk_indices], dim=-1)
            pre_cand = F.softmax(pre_logits_pos[topk_indices], dim=-1)

            jsd_values.append(safe_jsd(pre_cand, post_cand))

        values = np.array(jsd_values)
        mu = {"pos": float(np.mean(values))}
        sigma = {"pos": float(np.std(values))}

        # Clear captured states to free memory
        self._captured_hidden = None  # type: ignore[assignment]
        self._pre_mlp_hidden = None  # type: ignore[assignment]

        return PrefillStatistics(
            mu=mu,
            sigma=sigma,
            n_tokens=n_tokens,
            signals=("pos",),
        )


