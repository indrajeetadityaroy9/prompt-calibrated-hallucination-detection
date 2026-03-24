import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from .config import TokenSignals, DetectionResult
from .hooks import (
    EphemeralHiddenBuffer,
    LayerHooks,
    ModelAdapter,
)
from .signals.psp import PromptSubspaceProjection
from .signals._jsd_base import CandidateJSDSignal
from .signals.ent import compute_ent
from .signals.spt import SpectralPhaseTransition
from .aggregation.fusion import PromptAnchoredAggregator
from .aggregation.spans import SpanMerger
from .calibration import adaptive_window, self_calibrate
from .numerics import effective_rank


class Detector:

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

        self.adapter = ModelAdapter.from_model(model)
        self.lm_head = self.adapter.get_lm_head(model)
        self.final_norm = self.adapter.get_final_norm(model)
        self._layers = self.adapter.get_layers(model)
        self.num_layers = len(self._layers)

        self.psp_signal = PromptSubspaceProjection()
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)
        self.aggregator = PromptAnchoredAggregator()
        self.hidden_buffer = EphemeralHiddenBuffer()

        self.prompt_stats: dict | None = None
        self._hooks: list[LayerHooks] = []

    def _prefill(
        self,
        input_ids: Tensor,
        prompt_len: int,
    ) -> tuple:
        window_size = adaptive_window(prompt_len)

        for layer_idx in range(self.num_layers):
            hook = LayerHooks(layer_idx, self.hidden_buffer, adapter=self.adapter)
            hook.install(self._layers[layer_idx])
            self._hooks.append(hook)

        mid_layer_idx = self.num_layers // 2
        prompt_hidden_buf = []

        def capture_prompt_hidden(module, input, output):
            prompt_hidden_buf.append(output[0][0, :prompt_len, :].detach())

        mid_handle = self._layers[mid_layer_idx].register_forward_hook(capture_prompt_hidden)

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
        for layer_idx in range(self.num_layers):
            post_attn_norm = self.adapter.get_post_attn_norm(self._layers[layer_idx])
            h1 = post_attn_norm.register_forward_pre_hook(
                _make_tail_pre_hook(layer_idx, cal_tail_start, prompt_len, _tail_attn_tmp)
            )
            tail_handles.append(h1)
            h2 = self._layers[layer_idx].register_forward_hook(
                _make_tail_hook(layer_idx, cal_tail_start, prompt_len, tail_per_layer, _tail_attn_tmp)
            )
            tail_handles.append(h2)

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

        mid_handle.remove()
        for h in tail_handles:
            h.remove()

        prompt_hidden = prompt_hidden_buf[0]
        prompt_S = self.psp_signal.calibrate(prompt_hidden)

        spt_window = effective_rank(prompt_S)
        hidden_dim = prompt_hidden.shape[-1]
        self.spt_signal = SpectralPhaseTransition(hidden_dim, spt_window)

        self.prompt_stats = self_calibrate(
            psp_signal=self.psp_signal,
            jsd_signal=self.jsd_signal,
            spt_signal=self.spt_signal,
            lm_head=self.lm_head,
            final_norm=self.final_norm,
            prompt_len=prompt_len,
            tail_per_layer=tail_per_layer,
            mid_layer_idx=mid_layer_idx,
        )

        self.spt_signal.reset()
        self.spt_signal.seed(tail_per_layer[mid_layer_idx]["h_resid_mlp"])

        return past_key_values, last_logits

    def compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        attentions: tuple | None = None,
        seq_len: int = 0,
    ) -> TokenSignals:
        layer_states = self.hidden_buffer.get_states()

        probs = torch.softmax(logits.float(), dim=-1)
        k = max(2, effective_rank(probs))
        cand = torch.topk(logits, k).indices
        cand = torch.unique(torch.cat([cand, torch.tensor([emitted_token_id], device="cuda")]))

        ent = 0.5
        if attentions is not None:
            attn_tensor = torch.stack(attentions)[:, 0, :, -1, :]
            ent = compute_ent(attn_tensor, seq_len)

        mlp = self.jsd_signal.compute_mlp_jsd(layer_states, cand)

        psp_hidden = {idx: states.h_resid_mlp for idx, states in layer_states.items()}
        psp = self.psp_signal.compute_psp(psp_hidden)

        mid_layer = self.num_layers // 2
        self.spt_signal.push(layer_states[mid_layer].h_resid_mlp)
        spt, spectral_gap = self.spt_signal.compute_spt()

        self.hidden_buffer.clear()
        return TokenSignals(ent=ent, mlp=mlp, psp=psp, spt=spt, spectral_gap=spectral_gap)

    def _cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _aggregate_results(
        self,
        token_results: list[TokenSignals],
        all_ids: Tensor,
        prompt_len: int,
    ) -> DetectionResult:
        generated_text = self.tokenizer.decode(
            all_ids[0, prompt_len:], skip_special_tokens=True
        )

        _SIG_NAMES = ("ent", "mlp", "psp", "spt", "spectral_gap")
        n = len(token_results)
        signals_matrix = np.empty((n, 5))
        for i, s in enumerate(token_results):
            signals_matrix[i] = (s.ent, s.mlp, s.psp, s.spt, s.spectral_gap)
        response_signals = {name: signals_matrix[:, j] for j, name in enumerate(_SIG_NAMES)}

        agg_result = self.aggregator.compute_risk(self.prompt_stats, response_signals)
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
        response_ids: list[int] | None = None,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        past_key_values, next_logits = self._prefill(input_ids, prompt_len)

        all_ids = input_ids.clone()
        token_results: list[TokenSignals] = []

        n_steps = len(response_ids) if response_ids is not None else max_new_tokens

        if response_ids is not None:
            token_id = response_ids[0]
        else:
            token_id = next_logits.argmax(dim=-1).item()

        signals = self.compute_token_signals(
            next_logits.squeeze(0), token_id, attentions=None, seq_len=prompt_len + 1
        )
        token_results.append(signals)

        next_token = torch.tensor([[token_id]], device="cuda")
        all_ids = torch.cat([all_ids, next_token], dim=1)
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((1, 1), device="cuda", dtype=attention_mask.dtype)
        ], dim=1)
        cache_position = torch.tensor([prompt_len], device="cuda")

        for step in range(1, n_steps):
            cache_position = cache_position + 1
            seq_len = prompt_len + step + 1
            model_inputs = self.model.prepare_inputs_for_generation(
                input_ids=next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_attentions=True, return_dict=True)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if response_ids is not None:
                token_id = response_ids[step]
            else:
                token_id = logits.argmax(dim=-1).item()

            signals = self.compute_token_signals(
                logits.squeeze(0), token_id, attentions=outputs.attentions, seq_len=seq_len,
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
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        return self._generation_loop(input_ids, max_new_tokens=max_new_tokens)

    def detect(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)],
            dtype=torch.long, device="cuda",
        )
        return self._generation_loop(input_ids, max_new_tokens=max_new_tokens)

    def score(
        self,
        prompt: str,
        response_text: str,
    ) -> DetectionResult:
        full_text = prompt + response_text
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
        prompt_len = len(prompt_tokens)
        response_ids = full_tokens[prompt_len:]

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
        return self._generation_loop(input_ids, response_ids=response_ids)
