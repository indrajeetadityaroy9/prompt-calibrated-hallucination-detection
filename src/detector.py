import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from src.calibration import self_calibrate
from src.config import LayerHiddenStates, TokenSignals, DetectionResult
from src.fusion import CalibrationStats, compute_cusum_risks
from src.hooks import LayerHooks, ModelAdapter
from src.numerics import effective_rank, information_flow_regularity
from src.signals import SpectralAnalyzer, compute_ent, compute_mlp_jsd

_SIGNAL_FIELDS = ("rho", "phi", "spf", "mlp", "ent")


class Detector:

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.adapter = ModelAdapter.from_model(model)
        self.lm_head = self.adapter.get_lm_head(model)
        self.final_norm = self.adapter.get_final_norm(model)
        self._layers = self.adapter.get_layers(model)
        self.num_layers = len(self._layers)
        self.spectral_analyzer = SpectralAnalyzer()
        self._layer_states: dict[int, LayerHiddenStates] = {}
        self.prompt_stats: CalibrationStats | None = None
        self._hooks: list[LayerHooks] = []

    def _prefill(self, input_ids: Tensor, prompt_len: int) -> tuple:
        for layer_idx in range(self.num_layers):
            hook = LayerHooks(layer_idx, self._layer_states, adapter=self.adapter)
            hook.install(self._layers[layer_idx])
            self._hooks.append(hook)

        cal_tail_start = 0
        tail_per_layer = {}
        _tail_attn_tmp = {}

        def _make_tail_pre_hook(lidx, start, end, tmp):
            def fn(module, args):
                tmp[lidx] = args[0][0, start:end, :].detach()
            return fn

        def _make_tail_hook(lidx, start, end, buf, tmp):
            def fn(module, args, output):
                buf[lidx] = {"h_resid_attn": tmp.pop(lidx), "h_resid_mlp": output[0][0, start:end, :].detach()}
            return fn

        tail_handles = []
        for layer_idx in range(self.num_layers):
            post_attn_norm = self.adapter.get_post_attn_norm(self._layers[layer_idx])
            tail_handles.append(post_attn_norm.register_forward_pre_hook(_make_tail_pre_hook(layer_idx, cal_tail_start, prompt_len, _tail_attn_tmp)))
            tail_handles.append(self._layers[layer_idx].register_forward_hook(_make_tail_hook(layer_idx, cal_tail_start, prompt_len, tail_per_layer, _tail_attn_tmp)))

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=True, return_dict=True, output_attentions=True)

        for h in tail_handles:
            h.remove()

        layer_keys = sorted(tail_per_layer.keys())
        H_all = torch.stack([tail_per_layer[li]["h_resid_mlp"] for li in layer_keys], dim=1).float()
        self.spectral_analyzer.calibrate(H_all)

        self.prompt_stats = self_calibrate(spectral_analyzer=self.spectral_analyzer, lm_head=self.lm_head, final_norm=self.final_norm, tail_per_layer=tail_per_layer, prefill_attentions=outputs.attentions, cal_tail_start=cal_tail_start)

        return outputs.past_key_values, outputs.logits[:, -1, :], outputs.attentions

    def compute_token_signals(self, logits: Tensor, emitted_token_id: int, attentions: tuple, seq_len: int = 0) -> TokenSignals:
        layer_states = self._layer_states
        sorted_keys = sorted(layer_states.keys())

        H_token = torch.stack([layer_states[k].h_resid_mlp.float().squeeze() for k in sorted_keys])
        rho, spf = self.spectral_analyzer.compute(H_token)

        diffs = H_token[1:] - H_token[:-1]
        fi = diffs.norm(dim=-1) ** 2 / (H_token[:-1].norm(dim=-1) ** 2 + torch.finfo(H_token.dtype).eps)
        phi = information_flow_regularity(fi)

        probs = torch.softmax(logits.float(), dim=-1)
        cand = torch.topk(logits, effective_rank(probs)).indices
        cand = torch.unique(torch.cat([cand, torch.tensor([emitted_token_id], device=self.device)]))
        mlp = compute_mlp_jsd(layer_states, cand, self.lm_head, self.final_norm)

        ent = compute_ent(torch.stack(attentions)[:, 0, :, -1, :], seq_len)

        self._layer_states.clear()
        return TokenSignals(rho=rho, phi=phi, spf=spf, mlp=mlp, ent=ent)

    def _cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _aggregate_results(self, token_results: list[TokenSignals], all_ids: Tensor, prompt_len: int) -> DetectionResult:
        generated_text = self.tokenizer.decode(all_ids[0, prompt_len:], skip_special_tokens=True)
        signal_matrix = np.column_stack([[getattr(s, f) for s in token_results] for f in _SIGNAL_FIELDS])
        token_risks, cusum_values, response_risk, is_flagged, spans = compute_cusum_risks(signal_matrix, self.prompt_stats)

        return DetectionResult(generated_text=generated_text, token_signals=token_results, token_risks=token_risks, cusum_values=cusum_values, risky_spans=spans, response_risk=response_risk, is_flagged=is_flagged, num_tokens=len(token_results), prompt_length=prompt_len)

    def _generation_loop(self, input_ids: Tensor, max_new_tokens: int, response_ids: list[int] | None = None) -> DetectionResult:
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        past_key_values, next_logits, prefill_attentions = self._prefill(input_ids, prompt_len)
        all_ids = input_ids.clone()
        token_results: list[TokenSignals] = []
        n_steps = len(response_ids) if response_ids is not None else max_new_tokens

        token_id = response_ids[0] if response_ids is not None else next_logits.argmax(dim=-1).item()
        token_results.append(self.compute_token_signals(next_logits.squeeze(0), token_id, attentions=prefill_attentions, seq_len=prompt_len))
        del prefill_attentions

        next_token = torch.tensor([[token_id]], device=self.device)
        all_ids = torch.cat([all_ids, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)
        cache_position = torch.tensor([prompt_len], device=self.device)

        for step in range(1, n_steps):
            cache_position = cache_position + 1
            seq_len = prompt_len + step + 1
            model_inputs = self.model.prepare_inputs_for_generation(input_ids=next_token, past_key_values=past_key_values, attention_mask=attention_mask, cache_position=cache_position, use_cache=True)

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_attentions=True, return_dict=True)

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            token_id = response_ids[step] if response_ids is not None else logits.argmax(dim=-1).item()

            token_results.append(self.compute_token_signals(logits.squeeze(0), token_id, attentions=outputs.attentions, seq_len=seq_len))

            next_token = torch.tensor([[token_id]], device=self.device)
            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)

            if response_ids is None and token_id == self.tokenizer.eos_token_id:
                break

        self._cleanup()
        return self._aggregate_results(token_results, all_ids, prompt_len)

    def detect(self, prompt: str, max_new_tokens: int) -> DetectionResult:
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_special_tokens=True)], dtype=torch.long, device=self.device)
        return self._generation_loop(input_ids, max_new_tokens)
