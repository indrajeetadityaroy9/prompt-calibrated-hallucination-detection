"""
DSG (Decoupled Spectral Grounding) Detector.

Causal decomposition of hallucination risk via 5 signals:
- CUS: Lookback ratio bimodality — is the model looking at context? (attention)
- POS: Parametric override — is the FFN overriding what attention found? (transformation)
- DPS: Dual-subspace projection — does the representation live in context or reasoning space? (geometry)
- DoLa: Layer-contrast — did late layers add factual content? (factuality)
- CGD: Context-grounding direction — is generation moving toward or away from context? (activation steering)

Fusion: Entropy-gated weighted mean with prompt-anchored calibration.
"""

from typing import Any, List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from ..config import DSGConfig, DSGTokenSignals, DetectionResult, SIGNAL_REGISTRY
from ..hooks import (
    EphemeralHiddenBuffer,
    LayerHooks,
    PrefillContextHook,
    PrefillStatisticsHook,
)
from ..signals.context_grounding import DualSubspaceGrounding
from ..signals.topk_jsd import CandidateJSDSignal
from ..signals.copying_heads import (
    identify_copying_heads,
    compute_layer_affinity,
    ContextUtilizationSignal,
)
from ..aggregation.prompt_anchored import PromptAnchoredAggregator
from ..aggregation.span_merger import SpanMerger
from ..calibration import (
    get_layer_indices,
    build_input,
    adaptive_window,
    self_calibrate,
)


class CandidateSetManager:
    """Manages top-k candidate sets for signal computation."""

    def __init__(self, topk: int = 128):
        self.topk = topk
        self._prev_topk = None

    def build(self, logits, emitted_token_id):
        current_topk = torch.topk(logits, min(self.topk, len(logits))).indices
        candidates = set(current_topk.tolist())
        if self._prev_topk is not None:
            candidates.update(self._prev_topk.tolist())
        candidates.add(emitted_token_id)
        self._prev_topk = current_topk
        return torch.tensor(sorted(candidates), device=logits.device, dtype=torch.long)

    def reset(self):
        self._prev_topk = None

    @staticmethod
    def adaptive_topk(logits):
        """Adaptive top-k via 95% cumulative probability mass."""
        probs = torch.softmax(logits.float(), dim=-1)
        sorted_probs = probs.sort(descending=True).values
        k = int((sorted_probs.cumsum(-1) < 0.95).sum().item()) + 1
        return max(2, k)


class DSGDetector:
    """
    Decoupled Spectral Grounding detector.

    Pipeline:
    1. __init__: Precompute reasoning subspace SVD (once per model load)
    2. prefill(): Identify copying heads, compute context SVD, prompt stats,
       prompt center, magnitude tau
    3. detect(): CUS + POS + DPS + DoLa + CGD -> Entropy-Gated -> p90
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: DSGConfig = DSGConfig(),
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        # H100 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Model components
        self.lm_head = model.lm_head
        self.final_norm = model.model.norm
        self.num_layers = len(model.model.layers)

        # Layer selection
        self._hookable_layers = get_layer_indices(config, self.num_layers)

        # Signal computers
        self.dps_signal = DualSubspaceGrounding(
            lm_head_weight=self.lm_head.weight,
        )
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)

        # Candidate set manager
        self.candidate_manager = CandidateSetManager(topk=128)

        # Aggregator (5 signals, entropy-gated fusion)
        self.aggregator = PromptAnchoredAggregator(
            active_signals={"cus", "pos", "dps", "dola", "cgd"},
        )

        # Hidden state buffer (hooks still work for hidden states)
        self.hidden_buffer = EphemeralHiddenBuffer()

        # State (set during prefill)
        self.cus_signal = None
        self._prompt_stats = None
        self._prefill_hidden = None
        self._hooks: List = []

    def _prefill(
        self,
        input_ids: Tensor,
        context_start: int,
        context_end: int,
        prompt_len: int,
    ) -> Tuple[Any, Tensor]:
        """
        Run prefill: identify copying heads, compute context SVD, prompt stats,
        prompt center, and magnitude tau.
        Returns (past_key_values, last_logits).
        """
        window_size = adaptive_window(prompt_len)

        # Enable eager attention + output for CUS (restored in _cleanup)
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        # Install hidden state hooks (persist across prefill + generation; removed in _cleanup)
        for layer_idx in self._hookable_layers:
            layer = self.model.model.layers[layer_idx]
            hook = LayerHooks(layer_idx, self.hidden_buffer)
            hook.install(layer)
            self._hooks.append(hook)

        # Context hidden state hook (for DPS)
        context_buffer = []
        mid_layer_idx = self.num_layers // 2
        context_hook = PrefillContextHook(context_start, context_end, context_buffer)
        context_hook.install(self.model.model.layers[mid_layer_idx])

        # Prefill statistics hook
        stats_layer_idx = max(self._hookable_layers)
        stats_hook = PrefillStatisticsHook(
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            window_size=window_size,
        )
        stats_hook.install(self.model.model.layers[stats_layer_idx])

        # Hidden state capture hook for DPS/CGD prompt stats
        prefill_hidden_buffer = []

        def capture_hidden(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            prefill_hidden_buffer.append(hidden.detach())

        stats_layer = self.model.model.layers[stats_layer_idx]
        capture_handle = stats_layer.register_forward_hook(capture_hidden)

        # Per-layer tail capture for DoLa/DPS calibration.
        # Each decoder layer produces genuinely different hidden states —
        # needed for cross-layer signals (DoLa JSD, DPS multi-layer projection).
        # Memory: ~(n_hookable_layers × tail_tokens × hidden_dim × 2) bytes ≈ 13 MB
        cal_tail_start = max(0, prompt_len - window_size)
        _tail_per_layer = {}

        def _make_tail_hook(lidx, start, end, buf):
            def fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                buf[lidx] = h[0, start:end, :].detach()
            return fn

        _tail_handles = []
        for layer_idx in self._hookable_layers:
            layer = self.model.model.layers[layer_idx]
            h = layer.register_forward_hook(
                _make_tail_hook(layer_idx, cal_tail_start, prompt_len, _tail_per_layer)
            )
            _tail_handles.append(h)

        # Run prefill forward pass
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

        # Store prefill hidden states for prompt stat computation
        if prefill_hidden_buffer:
            self._prefill_hidden = prefill_hidden_buffer[0]
        capture_handle.remove()

        # Remove tail capture hooks (data is in _tail_per_layer)
        for h in _tail_handles:
            h.remove()
        del _tail_handles

        # A. Copying heads from model output attentions
        if outputs.attentions is not None:
            affinities = {}
            for layer_idx in self._hookable_layers:
                if layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[layer_idx]
                    affinity = compute_layer_affinity(attn, context_start, context_end)
                    affinities[layer_idx] = affinity.detach().cpu()

            copying_heads, _ = identify_copying_heads(affinities)
            self.cus_signal = ContextUtilizationSignal(
                copying_heads, prompt_len,
                n_layers=self.num_layers,
            )

        # B. Context SVD (DPS)
        if context_buffer:
            self.dps_signal.set_context_basis(context_buffer[0])
            self.jsd_signal.set_context_basis(self.dps_signal.context_basis)

        # C. Prompt center (CGD) — non-context prompt tokens
        if self._prefill_hidden is not None:
            prompt_mask = torch.ones(prompt_len, dtype=torch.bool)
            prompt_mask[context_start:context_end] = False
            if prompt_mask.any():
                prompt_hidden = self._prefill_hidden[0, prompt_mask, :]
                self.dps_signal.set_prompt_center(prompt_hidden)

        # D. Magnitude tau (DPS gate)
        if self._prefill_hidden is not None and self.dps_signal._context_center is not None:
            self.dps_signal.set_magnitude_tau(
                self._prefill_hidden[0],
                self.dps_signal._context_center,
            )

        # E. Prompt statistics
        prefill_stats = stats_hook.compute_statistics(prompt_len)
        jsd_mu = prefill_stats.mu.get("pos", 0.0)
        jsd_sigma = prefill_stats.sigma.get("pos", 1.0)
        self.jsd_signal.set_prompt_jsd_stats(jsd_mu, jsd_sigma)

        # Self-calibrate: derive all prompt stats from actual prefill data
        self._prompt_stats = self_calibrate(
            jsd_signal=self.jsd_signal,
            dps_signal=self.dps_signal,
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            num_layers=self.num_layers,
            prompt_len=prompt_len,
            tail_per_layer=_tail_per_layer,
            prefill_hidden=self._prefill_hidden,
        )

        # Clean up prefill hidden states and per-layer tail data
        self._prefill_hidden = None
        del _tail_per_layer

        # Remove prefill-only hooks (hidden hooks persist for generation)
        context_hook.remove()
        stats_hook.remove()
        self.hidden_buffer.clear()

        return past_key_values, last_logits

    def compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        attentions: Tuple = None,
    ) -> DSGTokenSignals:
        """Compute all 5 DSG signals for a single generation token."""
        layer_states = self.hidden_buffer.get_states()

        # Adaptive topk
        self.candidate_manager.topk = CandidateSetManager.adaptive_topk(logits)
        candidate_set = self.candidate_manager.build(logits, emitted_token_id)

        # DPS — with adaptive layer selection and magnitude gate
        dps_hidden = {idx: states.h_resid_mlp for idx, states in layer_states.items()}
        dps = self.dps_signal.compute_dps(dps_hidden, self.num_layers)

        # POS
        pos = self.jsd_signal.compute_pos(layer_states, candidate_set)

        # DoLa — layer-contrast score
        dola = self.jsd_signal.compute_dola_score(layer_states, emitted_token_id, candidate_set)

        # CGD — context-grounding direction from final layer
        cgd = 0.5
        if layer_states:
            final_layer = max(layer_states.keys())
            cgd = self.dps_signal.compute_grounding_direction(
                layer_states[final_layer].h_resid_mlp
            )

        # CUS — lookback ratio bimodality from model output attentions
        cus = 0.5
        if attentions is not None and self.cus_signal is not None:
            attn_slices = {}
            for layer_idx in self._hookable_layers:
                if layer_idx < len(attentions) and attentions[layer_idx] is not None:
                    attn = attentions[layer_idx]
                    attn_slices[layer_idx] = attn[0, :, -1, :]
            cus = self.cus_signal.compute_lookback_ratio_signal(
                attn_slices, self.cus_signal.prompt_len
            )

        # Clear hidden buffer
        self.hidden_buffer.clear()

        return DSGTokenSignals(cus=cus, pos=pos, dps=dps, dola=dola, cgd=cgd)

    def _cleanup(self):
        """Remove all generation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.model.config._attn_implementation = "sdpa"
        self.model.config.output_attentions = False

    def detect(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """
        Generate text with DSG hallucination detection.

        Returns DetectionResult with per-token risk scores from 5 signals.
        """
        self.candidate_manager.reset()

        input_ids, context_start, context_end, prompt_len = build_input(
            self.tokenizer, context, question, self.device
        )
        attention_mask = torch.ones_like(input_ids)

        try:
            # Prefill
            past_key_values, next_logits = self._prefill(
                input_ids, context_start, context_end, prompt_len
            )

            all_ids = input_ids.clone()
            token_results: List[DSGTokenSignals] = []

            # First token
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            signals = self.compute_token_signals(
                next_logits.squeeze(0), next_token.item(), attentions=None
            )
            token_results.append(signals)

            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)
            cache_position = torch.tensor([prompt_len], device=self.device)

            # Decode loop
            for step in range(1, max_new_tokens):
                cache_position = cache_position + 1
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )

                with torch.no_grad():
                    outputs = self.model(**model_inputs, return_dict=True)

                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1, keepdim=True)

                signals = self.compute_token_signals(
                    next_logits.squeeze(0),
                    next_token.item(),
                    attentions=outputs.attentions,
                )
                token_results.append(signals)

                all_ids = torch.cat([all_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        finally:
            self._cleanup()

        # Aggregate
        generated_text = self.tokenizer.decode(
            all_ids[0, prompt_len:], skip_special_tokens=True
        )

        response_signals = {
            "cus": np.array([s.cus for s in token_results]),
            "pos": np.array([s.pos for s in token_results]),
            "dps": np.array([s.dps for s in token_results]),
            "dola": np.array([s.dola for s in token_results]),
            "cgd": np.array([s.cgd for s in token_results]),
        }

        agg_result = self.aggregator.compute_risk(self._prompt_stats, response_signals)
        token_risks = agg_result.token_risks.tolist()
        response_risk = agg_result.risk

        # Spans — adaptive threshold
        merger = SpanMerger.adaptive(token_risks)
        risky_spans = merger.find_spans(token_risks)

        is_flagged = response_risk >= 0.5

        return DetectionResult(
            generated_text=generated_text,
            token_signals=token_results,
            token_risks=token_risks,
            risky_spans=risky_spans,
            response_risk=response_risk,
            is_flagged=is_flagged,
            num_tokens=len(token_results),
            prompt_length=prompt_len,
        )
