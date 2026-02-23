"""
AG-SAR Hallucination Detector.

Causal decomposition of hallucination risk via 6 signals:
- CUS: Lookback ratio bimodality — is the model looking at context? (attention)
- POS: Parametric override — is the FFN overriding what attention found? (transformation)
- DPS: Dual-subspace projection — does the representation live in context or reasoning space? (geometry)
- DoLa: Layer-contrast — did late layers add factual content? (factuality)
- CGD: Context-grounding direction — is generation moving toward or away from context? (activation steering)
- STD: Semantic trajectory dynamics — is the layer trajectory smooth or oscillatory? (trajectory)

Fusion: Entropy-gated weighted mean with prompt-anchored calibration.
"""

from typing import List, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from .config import TokenSignals, DetectionResult
from .hooks import (
    EphemeralHiddenBuffer,
    LayerHooks,
    ModelAdapter,
    PrefillContextHook,
    PrefillStatisticsHook,
)
from .signals.dps import DualSubspaceGrounding
from .signals._jsd_base import CandidateJSDSignal
from .signals.std import SemanticTrajectoryDynamics
from .signals.cus import (
    identify_copying_heads,
    compute_layer_affinity,
    ContextUtilizationSignal,
)
from .aggregation.fusion import PromptAnchoredAggregator
from .aggregation.spans import SpanMerger
from .calibration import (
    adaptive_window,
    self_calibrate,
    select_informative_dps_layers,
)


class CandidateSetManager:
    """Manages top-k candidate sets for signal computation."""

    def __init__(self, topk: int = None):
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
        """Adaptive top-k via entropy-adaptive nucleus mass."""
        from .numerics import entropy_adaptive_nucleus
        probs = torch.softmax(logits.float(), dim=-1)
        mass = entropy_adaptive_nucleus(probs)
        sorted_probs = probs.sort(descending=True).values
        k = int((sorted_probs.cumsum(-1) < mass).sum().item()) + 1
        return max(2, k)


class Detector:
    """
    AG-SAR hallucination detector.

    Pipeline:
    1. __init__: Precompute reasoning subspace SVD (once per model load)
    2. prefill(): Identify copying heads, compute context SVD, prompt stats,
       prompt center, magnitude tau
    3. detect(): CUS + POS + DPS + DoLa + CGD -> Entropy-Gated -> p90
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

        # TF32 optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Architecture adapter (auto-detects norm attribute names)
        self.adapter = ModelAdapter.from_model(model)

        # Model components (resolved via adapter for architecture portability)
        self.lm_head = self.adapter.get_lm_head(model)
        self.final_norm = self.adapter.get_final_norm(model)
        self._layers = self.adapter.get_layers(model)
        self.num_layers = len(self._layers)

        # All layers hooked
        self._hookable_layers = list(range(self.num_layers))

        # Signal computers
        self.dps_signal = DualSubspaceGrounding(
            lm_head_weight=self.lm_head.weight,
        )
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)
        self.std_signal = SemanticTrajectoryDynamics(self.num_layers)

        # Candidate set manager
        self.candidate_manager = CandidateSetManager()

        # Aggregator (5 signals, entropy-gated fusion)
        self.aggregator = PromptAnchoredAggregator()

        # Hidden state buffer (hooks still work for hidden states)
        self.hidden_buffer = EphemeralHiddenBuffer()

        # State (set during prefill)
        self.cus_signal: ContextUtilizationSignal  # Set in _prefill
        self._prompt_stats = None
        self._prefill_hidden = None
        self._hooks: List = []

    def _prefill(
        self,
        input_ids: Tensor,
        context_mask: Tensor,
        prompt_len: int,
    ) -> Tuple:
        """
        Run prefill: identify copying heads, compute context SVD, prompt stats,
        prompt center, and magnitude tau.
        Returns (past_key_values, last_logits).
        """
        context_mask = context_mask.to(input_ids.device)
        window_size = adaptive_window(prompt_len)

        # Enable eager attention + output for CUS (restored in _cleanup)
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        # Install hidden state hooks (persist across prefill + generation; removed in _cleanup)
        for layer_idx in self._hookable_layers:
            layer = self._layers[layer_idx]
            hook = LayerHooks(layer_idx, self.hidden_buffer, adapter=self.adapter)
            hook.install(layer)
            self._hooks.append(hook)

        # Context hidden state hooks at 3 candidate layers — select best by spectral gap
        from .numerics import EPS as _eps
        candidate_layers = sorted(set([
            self.num_layers // 3,
            self.num_layers // 2,
            2 * self.num_layers // 3,
        ]))
        context_buffers = {}
        context_hooks = []
        for layer_idx in candidate_layers:
            buf = []
            hook = PrefillContextHook(context_mask, buf)
            hook.install(self._layers[layer_idx])
            context_buffers[layer_idx] = buf
            context_hooks.append(hook)

        # Prefill statistics hook
        stats_layer_idx = max(self._hookable_layers)
        stats_hook = PrefillStatisticsHook(
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            window_size=window_size,
            adapter=self.adapter,
        )
        stats_hook.install(self._layers[stats_layer_idx])

        # Hidden state capture hook for DPS/CGD prompt stats
        prefill_hidden_buffer = []

        def capture_hidden(module, input, output):
            prefill_hidden_buffer.append(output[0].detach())

        stats_layer = self._layers[stats_layer_idx]
        capture_handle = stats_layer.register_forward_hook(capture_hidden)

        # Per-layer tail capture for DoLa/DPS calibration.
        # Each decoder layer produces genuinely different hidden states —
        # needed for cross-layer signals (DoLa JSD, DPS multi-layer projection).
        # Memory: ~(n_hookable_layers × tail_tokens × hidden_dim × 2) bytes ≈ 13 MB
        cal_tail_start = max(0, prompt_len - window_size)
        _tail_per_layer = {}

        def _make_tail_hook(lidx, start, end, buf):
            def fn(module, args, output):
                buf[lidx] = output[0][0, start:end, :].detach()
            return fn

        _tail_handles = []
        for layer_idx in self._hookable_layers:
            layer = self._layers[layer_idx]
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

        # Initialize adaptive topk from prefill logits
        self.candidate_manager.topk = CandidateSetManager.adaptive_topk(last_logits.squeeze(0))

        # Store prefill hidden states for prompt stat computation
        self._prefill_hidden = prefill_hidden_buffer[0]
        capture_handle.remove()

        # Remove tail capture hooks (data is in _tail_per_layer)
        for h in _tail_handles:
            h.remove()
        del _tail_handles

        # A. Copying heads from model output attentions
        affinities = {}
        for layer_idx in self._hookable_layers:
            attn = outputs.attentions[layer_idx]
            affinity = compute_layer_affinity(attn, context_mask)
            affinities[layer_idx] = affinity.detach().cpu()

        copying_heads, affinity_map = identify_copying_heads(affinities)
        self.cus_signal = ContextUtilizationSignal(
            copying_heads, affinity_map, prompt_len,
            n_layers=self.num_layers,
        )

        # B. Context SVD (DPS) — select layer with largest spectral gap S[0]/S[1]
        best_layer = candidate_layers[len(candidate_layers) // 2]  # default: middle
        best_gap = -1.0
        for layer_idx, buf in context_buffers.items():
            if buf and buf[0].shape[0] >= 2:
                h = buf[0].float()
                S = torch.linalg.svdvals(h - h.mean(dim=0, keepdim=True))
                if len(S) > 1:
                    gap = (S[0] / (S[1] + _eps)).item()
                    if gap > best_gap:
                        best_gap, best_layer = gap, layer_idx
        self.dps_signal.set_context_basis(context_buffers[best_layer][0])
        self.jsd_signal.set_context_basis(self.dps_signal.context_basis)
        self.std_signal.set_context_basis(self.dps_signal.context_basis)
        for hook in context_hooks:
            hook.remove()

        # C. Prompt center (CGD) — non-context prompt tokens
        non_context_mask = ~context_mask[:prompt_len]
        non_context_hidden = self._prefill_hidden[0, non_context_mask, :]
        if non_context_hidden.shape[0] == 0:
            # All tokens are context → use context mean as anchor.
            # CGD returns 0.5 (uninformative); entropy gating suppresses it.
            self.dps_signal.set_prompt_center(self._prefill_hidden[0, context_mask[:prompt_len], :])
        else:
            self.dps_signal.set_prompt_center(non_context_hidden)

        # D. Magnitude tau (DPS gate)
        self.dps_signal.set_magnitude_tau(
            self._prefill_hidden[0],
            self.dps_signal._context_center,
        )

        # E. Prompt statistics (sigma + sorted reference values for PIT)
        prefill_stats = stats_hook.compute_statistics(prompt_len)
        self.jsd_signal._prompt_jsd_sigma = prefill_stats.sigma
        self.jsd_signal._prompt_jsd_values = prefill_stats.raw_values

        # Self-calibrate: derive all prompt stats from actual prefill data
        self._prompt_stats = self_calibrate(
            jsd_signal=self.jsd_signal,
            dps_signal=self.dps_signal,
            std_signal=self.std_signal,
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            num_layers=self.num_layers,
            prompt_len=prompt_len,
            tail_per_layer=_tail_per_layer,
            prefill_hidden=self._prefill_hidden,
        )

        # Data-driven DPS layer selection (replaces middle-third heuristic)
        _first_key = next(iter(_tail_per_layer))
        _n_tail = _tail_per_layer[_first_key].shape[0]
        dps_layers = select_informative_dps_layers(
            self.dps_signal, _tail_per_layer, self.num_layers, _n_tail,
        )
        self.dps_signal.set_dps_layers(dps_layers)

        # Clean up prefill hidden states and per-layer tail data
        self._prefill_hidden = None
        del _tail_per_layer

        # Remove prefill-only hooks (hidden hooks persist for generation)
        stats_hook.remove()
        # NOTE: Do NOT clear hidden_buffer here. The prefill forward pass captured
        # last-position hidden states via LayerHooks, which are needed by the first
        # compute_token_signals() call. The buffer is cleared inside
        # compute_token_signals() after use (line 332).

        return past_key_values, last_logits

    def compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        attentions: Tuple = None,
    ) -> TokenSignals:
        """Compute all 6 signals for a single generation token."""
        layer_states = self.hidden_buffer.get_states()

        # Guard: if hooks failed to capture states, return neutral signals
        if not layer_states:
            self.hidden_buffer.clear()
            return TokenSignals()

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
        final_layer = max(layer_states.keys())
        cgd = self.dps_signal.compute_grounding_direction(
            layer_states[final_layer].h_resid_mlp
        )

        # STD — semantic trajectory dynamics using h_mlp_in (post-norm)
        std_hidden = {idx: states.h_mlp_in for idx, states in layer_states.items()}
        std = self.std_signal.compute_std(std_hidden)

        # CUS — lookback ratio bimodality from model output attentions
        cus = 0.5
        if attentions is not None:
            attn_slices = {}
            for layer_idx in self._hookable_layers:
                attn = attentions[layer_idx]
                attn_slices[layer_idx] = attn[0, :, -1, :]
            cus = self.cus_signal.compute_lookback_ratio_signal(
                attn_slices, self.cus_signal.prompt_len
            )

        # Clear hidden buffer
        self.hidden_buffer.clear()

        return TokenSignals(cus=cus, pos=pos, dps=dps, dola=dola, cgd=cgd, std=std)

    def _cleanup(self):
        """Remove all generation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.model.config._attn_implementation = "sdpa"
        self.model.config.output_attentions = False

    def _build_input_with_mask(
        self,
        context: str,
        question: str,
        template: str,
    ) -> Tuple[Tensor, Tensor, int]:
        """Build input_ids + context_mask from template segments.

        Mask is built from known integer boundaries (no subsequence matching).
        Returns (input_ids [1, seq_len], context_mask [seq_len], prompt_len).
        """
        parts = template.split("{context}")
        prefix_str = parts[0]
        rest = parts[1].split("{question}")
        middle_str = rest[0]
        suffix_str = rest[1]

        bos = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else []
        prefix = self.tokenizer.encode(prefix_str, add_special_tokens=False)
        ctx_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        middle = self.tokenizer.encode(middle_str, add_special_tokens=False)
        q_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        suffix = self.tokenizer.encode(suffix_str, add_special_tokens=False)

        ctx_start = len(bos) + len(prefix)
        ctx_end = ctx_start + len(ctx_tokens)
        tokens = bos + prefix + ctx_tokens + middle + q_tokens + suffix

        context_mask = torch.zeros(len(tokens), dtype=torch.bool)
        context_mask[ctx_start:ctx_end] = True

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return input_ids, context_mask, len(tokens)

    def _aggregate_results(
        self,
        token_results: List[TokenSignals],
        all_ids: Tensor,
        prompt_len: int,
    ) -> DetectionResult:
        """Shared aggregation for detect_from_tokens and score."""
        generated_text = self.tokenizer.decode(
            all_ids[0, prompt_len:], skip_special_tokens=True
        )

        response_signals = {
            "cus": np.array([s.cus for s in token_results]),
            "pos": np.array([s.pos for s in token_results]),
            "dps": np.array([s.dps for s in token_results]),
            "dola": np.array([s.dola for s in token_results]),
            "cgd": np.array([s.cgd for s in token_results]),
            "std": np.array([s.std for s in token_results]),
        }

        agg_result = self.aggregator.compute_risk(self._prompt_stats, response_signals)
        token_risks = agg_result.token_risks.tolist()
        response_risk = agg_result.risk

        # Spans — adaptive threshold
        merger = SpanMerger.adaptive(token_risks)
        risky_spans = merger.find_spans(token_risks)

        # Flag when response risk exceeds the bimodality-adaptive Tukey fence
        is_flagged = response_risk >= merger.threshold

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

    def detect_from_tokens(
        self,
        input_ids: Tensor,
        context_mask: Tensor,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """
        Format-agnostic detection from pre-tokenized input + context mask.

        Args:
            input_ids: [1, seq_len] token IDs
            context_mask: [seq_len] boolean mask (True = context token)
            max_new_tokens: maximum tokens to generate
        """
        self.candidate_manager.reset()
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        try:
            # Prefill
            past_key_values, next_logits = self._prefill(
                input_ids, context_mask, prompt_len
            )

            all_ids = input_ids.clone()
            token_results: List[TokenSignals] = []

            # First token: no generation-time attention available (prefill logits only).
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
            cache_position = torch.tensor([prompt_len - 1], device=self.device)

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

        return self._aggregate_results(token_results, all_ids, prompt_len)

    def detect(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
        prompt_template: str = None,
    ) -> DetectionResult:
        """
        Generate text with hallucination detection.

        Args:
            question: The question to answer.
            context: The context/passage to ground against.
            max_new_tokens: Maximum tokens to generate.
            prompt_template: Template with {context} and {question} placeholders.
                Defaults to "Context: {context}\\n\\nQuestion: {question}\\n\\nAnswer:".
        """
        if prompt_template is None:
            prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_ids, context_mask, prompt_len = self._build_input_with_mask(
            context, question, prompt_template
        )
        return self.detect_from_tokens(input_ids, context_mask, max_new_tokens)

    def _find_subsequence(self, haystack: List[int], needle: List[int]) -> int:
        """Find first occurrence of needle as subsequence in haystack. Returns start index or -1."""
        n, m = len(haystack), len(needle)
        for i in range(n - m + 1):
            if haystack[i:i + m] == needle:
                return i
        return -1

    def score(
        self,
        prompt: str,
        response_text: str,
        context_text: str,
    ) -> DetectionResult:
        """
        Score pre-existing text via teacher-forced decode.

        Args:
            prompt: The full prompt string (including context and question).
            response_text: The model's response to score.
            context_text: The context/passage within the prompt for grounding.

        Returns DetectionResult with per-token risk scores from 6 signals.
        """
        self.candidate_manager.reset()

        # Tokenize full prompt+response as one string for correct BPE at junction
        full_text = prompt + response_text
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens)
        response_ids = full_tokens[prompt_len:]

        if len(response_ids) == 0:
            raise ValueError("Response text produces no tokens after prompt.")

        # Find context within prompt via token subsequence matching
        ctx_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        ctx_start = self._find_subsequence(prompt_tokens, ctx_ids)

        context_mask = torch.zeros(prompt_len, dtype=torch.bool)
        if ctx_start >= 0:
            context_mask[ctx_start:ctx_start + len(ctx_ids)] = True
        else:
            # Fallback: find context by character position and count tokens
            char_pos = prompt.find(context_text)
            if char_pos >= 0:
                prefix = prompt[:char_pos]
                prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
                ctx_token_start = len(prefix_tokens)
                ctx_token_end = ctx_token_start + len(ctx_ids)
                ctx_token_end = min(ctx_token_end, prompt_len)
                context_mask[ctx_token_start:ctx_token_end] = True
            else:
                raise ValueError("Context text not found in prompt.")

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        try:
            # Prefill
            past_key_values, first_logits = self._prefill(
                input_ids, context_mask, prompt_len
            )

            all_ids = input_ids.clone()
            token_results: List[TokenSignals] = []

            # First response token: teacher-forced, no generation-time attention
            token_id = response_ids[0]
            signals = self.compute_token_signals(
                first_logits.squeeze(0), token_id, attentions=None
            )
            token_results.append(signals)

            next_token = torch.tensor([[token_id]], device=self.device)
            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
            ], dim=1)
            cache_position = torch.tensor([prompt_len - 1], device=self.device)

            # Teacher-forced decode loop for remaining tokens
            for step in range(1, len(response_ids)):
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
                logits = outputs.logits[:, -1, :]

                # Teacher-forced: use actual response token, not argmax
                token_id = response_ids[step]
                signals = self.compute_token_signals(
                    logits.squeeze(0),
                    token_id,
                    attentions=outputs.attentions,
                )
                token_results.append(signals)

                next_token = torch.tensor([[token_id]], device=self.device)
                all_ids = torch.cat([all_ids, next_token], dim=1)
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)
        finally:
            self._cleanup()

        return self._aggregate_results(token_results, all_ids, prompt_len)
