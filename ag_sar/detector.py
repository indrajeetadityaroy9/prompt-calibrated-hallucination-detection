"""
AG-SAR Hallucination Detector.

3-signal decomposition of hallucination risk:
- CUS: Lookback ratio bimodality — is the model looking at context? (attention)
- POS: Parametric override — is the FFN overriding what attention found? (transformation)
- DPS: Dual-subspace projection — does the representation live in context or reasoning space? (geometry)

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
)
from .signals.dps import DualSubspaceGrounding
from .signals._jsd_base import CandidateJSDSignal
from .signals.cus import compute_cus
from .aggregation.fusion import PromptAnchoredAggregator
from .aggregation.spans import SpanMerger
from .calibration import adaptive_window, self_calibrate
from .numerics import effective_rank


class Detector:
    """
    AG-SAR hallucination detector.

    Pipeline:
    1. __init__: Precompute reasoning subspace SVD (once per model load)
    2. prefill: Compute context SVD, prompt stats, prompt center, magnitude tau
    3. detect/score: CUS + POS + DPS -> Entropy-Gated fusion -> spans
    """

    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        self.adapter = ModelAdapter.from_model(model)
        self.lm_head = self.adapter.get_lm_head(model)
        self.final_norm = self.adapter.get_final_norm(model)
        self._layers = self.adapter.get_layers(model)
        self.num_layers = len(self._layers)
        self._hookable_layers = list(range(self.num_layers))

        self.dps_signal = DualSubspaceGrounding(lm_head_weight=self.lm_head.weight)
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)
        self.aggregator = PromptAnchoredAggregator()
        self.hidden_buffer = EphemeralHiddenBuffer()

        self._prompt_stats = None
        self._cus_prompt_len: int = 0
        self._hooks: List = []

    def _prefill(
        self,
        input_ids: Tensor,
        context_mask: Tensor,
        prompt_len: int,
    ) -> Tuple:
        """Run prefill: compute context SVD, prompt center, magnitude tau, prompt stats."""
        context_mask = context_mask.to("cuda")
        window_size = adaptive_window(prompt_len)

        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        # Install 2-point layer hooks (persist through generation)
        for layer_idx in self._hookable_layers:
            hook = LayerHooks(layer_idx, self.hidden_buffer, adapter=self.adapter)
            hook.install(self._layers[layer_idx])
            self._hooks.append(hook)

        # Context hook on midpoint layer
        ctx_layer_idx = self.num_layers // 2
        ctx_buf = []
        ctx_hook = PrefillContextHook(context_mask, ctx_buf)
        ctx_hook.install(self._layers[ctx_layer_idx])

        # Prefill hidden capture (final layer, for prompt center + magnitude tau)
        prefill_hidden_buf = []

        def capture_hidden(module, input, output):
            prefill_hidden_buf.append(output[0].detach())

        final_layer = self._layers[max(self._hookable_layers)]
        capture_handle = final_layer.register_forward_hook(capture_hidden)

        # Per-layer tail capture for calibration (2-point: h_resid_attn + h_resid_mlp)
        cal_tail_start = max(0, prompt_len - window_size)
        tail_per_layer = {}
        _tail_attn_tmp = {}

        def _make_tail_pre_hook(lidx, start, end, tmp):
            def fn(module, args):
                tmp[lidx] = args[0][0, start:end, :].detach()
            return fn

        def _make_tail_hook(lidx, start, end, buf, tmp):
            def fn(module, args, output):
                buf[lidx] = {
                    "h_resid_attn": tmp.pop(lidx),
                    "h_resid_mlp": output[0][0, start:end, :].detach(),
                }
            return fn

        tail_handles = []
        for layer_idx in self._hookable_layers:
            post_attn_norm = self.adapter.get_post_attn_norm(self._layers[layer_idx])
            h1 = post_attn_norm.register_forward_pre_hook(
                _make_tail_pre_hook(layer_idx, cal_tail_start, prompt_len, _tail_attn_tmp)
            )
            tail_handles.append(h1)
            h2 = self._layers[layer_idx].register_forward_hook(
                _make_tail_hook(layer_idx, cal_tail_start, prompt_len, tail_per_layer, _tail_attn_tmp)
            )
            tail_handles.append(h2)

        # Forward pass
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

        # Capture prefill hidden, remove temporary hooks
        prefill_hidden = prefill_hidden_buf[0]
        capture_handle.remove()
        for h in tail_handles:
            h.remove()

        # Context basis from midpoint layer
        self.dps_signal.set_context_basis(ctx_buf[0])
        self.jsd_signal.set_context_basis(self.dps_signal.context_basis)
        ctx_hook.remove()

        # Prompt center (non-context tokens)
        non_context_mask = ~context_mask[:prompt_len]
        non_context_hidden = prefill_hidden[0, non_context_mask, :]
        if non_context_hidden.shape[0] == 0:
            self.dps_signal.set_prompt_center(prefill_hidden[0, context_mask[:prompt_len], :])
        else:
            self.dps_signal.set_prompt_center(non_context_hidden)

        # Magnitude tau
        self.dps_signal.set_magnitude_tau(prefill_hidden[0], self.dps_signal._context_center)

        # CUS prompt length (needed during generation)
        self._cus_prompt_len = prompt_len

        # Self-calibrate
        self._prompt_stats = self_calibrate(
            dps_signal=self.dps_signal,
            jsd_signal=self.jsd_signal,
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            prompt_len=prompt_len,
            tail_per_layer=tail_per_layer,
        )

        return past_key_values, last_logits

    def compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        attentions: Tuple = None,
    ) -> TokenSignals:
        """Compute all 3 signals for a single generation token."""
        layer_states = self.hidden_buffer.get_states()

        # Candidate set via effective rank
        probs = torch.softmax(logits.float(), dim=-1)
        k = max(2, effective_rank(probs))
        cand = torch.topk(logits, min(k, len(logits))).indices
        cand = torch.unique(torch.cat([cand, torch.tensor([emitted_token_id], device="cuda")]))

        # DPS — all-layer mean
        dps_hidden = {idx: states.h_resid_mlp for idx, states in layer_states.items()}
        dps = self.dps_signal.compute_dps(dps_hidden)

        # POS — JSD-weighted directional override
        pos = self.jsd_signal.compute_pos(layer_states, cand)

        # CUS — lookback ratio bimodality
        cus = 0.5
        if attentions is not None:
            attn_slices = {
                layer_idx: attentions[layer_idx][0, :, -1, :]
                for layer_idx in self._hookable_layers
            }
            cus = compute_cus(attn_slices, self._cus_prompt_len)

        self.hidden_buffer.clear()
        return TokenSignals(cus=cus, pos=pos, dps=dps)

    def _cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _build_input_with_mask(
        self,
        context: str,
        question: str,
        template: str,
    ) -> Tuple[Tensor, Tensor, int]:
        """Build input_ids + context_mask from template segments."""
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

        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
        return input_ids, context_mask, len(tokens)

    def _aggregate_results(
        self,
        token_results: List[TokenSignals],
        all_ids: Tensor,
        prompt_len: int,
    ) -> DetectionResult:
        generated_text = self.tokenizer.decode(
            all_ids[0, prompt_len:], skip_special_tokens=True
        )

        response_signals = {
            "cus": np.array([s.cus for s in token_results]),
            "pos": np.array([s.pos for s in token_results]),
            "dps": np.array([s.dps for s in token_results]),
        }

        agg_result = self.aggregator.compute_risk(self._prompt_stats, response_signals)
        token_risks = agg_result.token_risks.tolist()
        response_risk = agg_result.risk

        merger = SpanMerger.adaptive(token_risks)
        risky_spans = merger.find_spans(token_risks)
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

    def _generation_loop(
        self,
        input_ids: Tensor,
        context_mask: Tensor,
        response_ids: List[int] = None,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """Shared generation loop. Teacher-forced if response_ids provided, greedy otherwise."""
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        past_key_values, next_logits = self._prefill(input_ids, context_mask, prompt_len)

        all_ids = input_ids.clone()
        token_results: List[TokenSignals] = []

        # Determine total steps
        n_steps = len(response_ids) if response_ids is not None else max_new_tokens

        # First token
        if response_ids is not None:
            token_id = response_ids[0]
        else:
            token_id = next_logits.argmax(dim=-1).item()

        signals = self.compute_token_signals(
            next_logits.squeeze(0), token_id, attentions=None
        )
        token_results.append(signals)

        next_token = torch.tensor([[token_id]], device="cuda")
        all_ids = torch.cat([all_ids, next_token], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device="cuda", dtype=attention_mask.dtype)
        ], dim=1)
        cache_position = torch.tensor([prompt_len - 1], device="cuda")

        # Remaining tokens
        for step in range(1, n_steps):
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

            if response_ids is not None:
                token_id = response_ids[step]
            else:
                token_id = logits.argmax(dim=-1).item()

            signals = self.compute_token_signals(
                logits.squeeze(0), token_id, attentions=outputs.attentions,
            )
            token_results.append(signals)

            next_token = torch.tensor([[token_id]], device="cuda")
            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device="cuda", dtype=attention_mask.dtype)
            ], dim=1)

            if response_ids is None and token_id == self.tokenizer.eos_token_id:
                break

        self._cleanup()
        return self._aggregate_results(token_results, all_ids, prompt_len)

    def detect_from_tokens(
        self,
        input_ids: Tensor,
        context_mask: Tensor,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """Format-agnostic detection from pre-tokenized input + context mask."""
        return self._generation_loop(input_ids, context_mask, max_new_tokens=max_new_tokens)

    def detect(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
        prompt_template: str = None,
    ) -> DetectionResult:
        """Generate text with hallucination detection."""
        if prompt_template is None:
            prompt_template = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        input_ids, context_mask, _ = self._build_input_with_mask(
            context, question, prompt_template
        )
        return self.detect_from_tokens(input_ids, context_mask, max_new_tokens)

    def score(
        self,
        prompt: str,
        response_text: str,
        context_text: str,
    ) -> DetectionResult:
        """Score pre-existing text via teacher-forced decode."""
        full_text = prompt + response_text
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_len = len(prompt_tokens)
        response_ids = full_tokens[prompt_len:]

        ctx_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        ctx_start = _find_subsequence(prompt_tokens, ctx_ids)

        context_mask = torch.zeros(prompt_len, dtype=torch.bool)
        context_mask[ctx_start:ctx_start + len(ctx_ids)] = True

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
        return self._generation_loop(input_ids, context_mask, response_ids=response_ids)


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    n, m = len(haystack), len(needle)
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1
