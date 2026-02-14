"""
DSG (Decoupled Spectral Grounding) Detector.

Causal decomposition of hallucination risk:
- CUS: Is the model looking at context? (attention)
- POS: Is the FFN overriding what attention found? (transformation)
- DPS: Does the representation live in context or reasoning space? (geometry)

Fusion: Causal Noisy-OR with prompt-anchored calibration.
"""

from typing import Any, Dict, List, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

import math
from ..config import DSGConfig, DSGTokenSignals, DetectionResult, PrefillCalibration
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
from ..numerics import mad_sigma, otsu_threshold
from ..aggregation.span_merger import SpanMerger
from ..aggregation.conformal import ConformalCalibrator


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
        """
        Adaptive top-k via 95% cumulative probability mass.

        No min/max bounds — the mass threshold determines k naturally.
        Floor at k=2 (need emitted token + one alternative).
        For peaked distributions k is small; for flat distributions k is large.
        """
        probs = torch.softmax(logits.float(), dim=-1)
        sorted_probs = probs.sort(descending=True).values
        k = int((sorted_probs.cumsum(-1) < 0.95).sum().item()) + 1
        return max(2, k)


class DSGDetector:
    """
    Decoupled Spectral Grounding detector.

    Pipeline:
    1. __init__: Precompute reasoning subspace SVD (once per model load)
    2. prefill(): Identify copying heads, compute context SVD, prompt stats
    3. detect(): CUS + POS + DPS -> Causal Noisy-OR -> p90

    Attention weights are extracted from model outputs (outputs.attentions)
    rather than layer hooks, since transformers 4.45+ returns only hidden_states
    from LlamaDecoderLayer.forward().
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: DSGConfig = DSGConfig(),
        conformal: ConformalCalibrator = None,
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
        self._hookable_layers = self._get_layer_indices()

        # Signal computers
        self.dps_signal = DualSubspaceGrounding(
            lm_head_weight=self.lm_head.weight,
        )
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)

        # Candidate set manager
        self.candidate_manager = CandidateSetManager(topk=128)

        # Aggregator (always p90, DSG signals)
        self.aggregator = PromptAnchoredAggregator(
            active_signals={"cus", "pos", "dps"},
        )

        # Conformal calibrator (optional)
        self.conformal = conformal

        # Hidden state buffer (hooks still work for hidden states)
        self.hidden_buffer = EphemeralHiddenBuffer()

        # State (set during prefill)
        self.cus_signal = None
        self._prompt_stats = None
        self._prefill_hidden = None
        self._dps_active_layers = None  # Adaptive layer selection for DPS
        self._hooks: List = []

        print(f"DSG Detector initialized: {len(self._hookable_layers)} layers, "
              f"reasoning basis shape: {self.dps_signal._reasoning_basis.shape}")

    def _get_layer_indices(self) -> List[int]:
        if self.config.layer_subset == "all":
            return list(range(self.num_layers))
        elif self.config.layer_subset == "last_third":
            start = self.num_layers - (self.num_layers // 3)
            return list(range(start, self.num_layers))
        elif isinstance(self.config.layer_subset, list):
            return self.config.layer_subset
        return list(range(self.num_layers))

    def _build_input(
        self, context: str, question: str,
    ) -> Tuple[Tensor, int, int, int]:
        """Build input_ids. Returns (input_ids, context_start, context_end, prompt_len)."""
        prefix = self.tokenizer.encode("Context: ", add_special_tokens=False)
        ctx_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        sep = self.tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
        q_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        suffix = self.tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

        bos = self.tokenizer.bos_token_id
        bos_len = 1 if bos is not None else 0

        context_start = bos_len + len(prefix)
        context_end = context_start + len(ctx_tokens)

        tokens = (([bos] if bos is not None else [])
                  + prefix + ctx_tokens + sep + q_tokens + suffix)

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return input_ids, context_start, context_end, len(tokens)

    @staticmethod
    def _adaptive_window(prompt_len: int) -> int:
        """
        Adaptive tail window: sqrt(prompt_len), clamped to [16, prompt_len//2].

        For a stationary process, standard error decreases as 1/sqrt(n).
        Using n ~ sqrt(L) balances estimation accuracy against locality.
        """
        window = int(math.ceil(math.sqrt(prompt_len)))
        return max(16, min(window, prompt_len // 2, prompt_len))

    def _select_informative_layers(self, layer_states_per_token):
        """
        Select informative layers for DPS via JSD variance + Otsu threshold.

        Layers with high JSD variance across prefill tokens are actively
        differentiating between token types — these are the informative layers.
        Uses Otsu's method for parameter-free bimodal separation.

        Fallback: top-third by variance if Otsu selects < 3 layers.
        """
        if not layer_states_per_token:
            return None

        # Compute per-layer JSD variance across all observed tokens
        layer_variances = {}
        for layer_idx in layer_states_per_token:
            jsds = layer_states_per_token[layer_idx]
            if len(jsds) >= 2:
                layer_variances[layer_idx] = float(np.var(jsds))

        if len(layer_variances) < 3:
            return None

        var_values = np.array(list(layer_variances.values()))
        threshold = otsu_threshold(var_values)

        selected = [idx for idx, v in layer_variances.items() if v >= threshold]

        if len(selected) < 3:
            sorted_layers = sorted(layer_variances.items(), key=lambda x: x[1], reverse=True)
            n_select = max(3, len(sorted_layers) // 3)
            selected = [idx for idx, _ in sorted_layers[:n_select]]

        return sorted(selected)

    def _self_calibrate(self, outputs, context_start, context_end, prompt_len):
        """
        Self-calibrating prompt statistics. No hardcoded priors.

        Strategy per signal:
        - DPS: Compute from prefill tail. MAD-based robust sigma.
        - CUS: Direct mode — CUS ∈ [0,1] with semantic meaning, no z-score.
        - POS: Compute from prefill JSD statistics (already in PrefillStatisticsHook).
               MAD-based robust sigma.

        Returns dict suitable for PromptAnchoredAggregator.compute_risk().
        """
        window = self._adaptive_window(prompt_len)
        tail_start = max(0, prompt_len - window)

        # DPS: project tail hidden states onto context/reasoning subspaces
        dps_values = []
        if hasattr(self, '_prefill_hidden') and self._prefill_hidden is not None:
            for t in range(tail_start, prompt_len):
                h = self._prefill_hidden[:, t, :].squeeze(0)
                h_centered = h.float() - self.dps_signal._context_center.squeeze(0).float()
                h_norm = torch.norm(h_centered) + 1e-10
                V_ctx = self.dps_signal._context_basis.float()
                V_rsn = self.dps_signal._reasoning_basis.float()
                proj_ctx = V_ctx.T @ (V_ctx @ h_centered)
                s_ctx = torch.norm(proj_ctx) / h_norm
                proj_rsn = V_rsn.T @ (V_rsn @ h_centered)
                s_rsn = torch.norm(proj_rsn) / h_norm
                dps_val = (s_rsn / (s_ctx + s_rsn + 1e-10)).item()
                dps_values.append(dps_val)

        stats = {}

        # DPS: robust statistics from prefill tail
        if dps_values and len(dps_values) >= 2:
            dps_arr = np.array(dps_values)
            sigma_robust = mad_sigma(dps_arr)
            sigma_std = float(np.std(dps_arr))
            sigma = max(sigma_robust, sigma_std)
            # Data-proportional floor: 10% of robust sigma
            sigma_floor = 0.1 * sigma_robust if sigma_robust > 0 else 0.01
            stats["dps"] = {
                "mu": float(np.mean(dps_arr)),
                "sigma": max(sigma, sigma_floor),
            }
        else:
            # Range-based fallback for very short prompts
            if dps_values:
                stats["dps"] = {"mu": float(dps_values[0]), "sigma": 0.1}
            else:
                stats["dps"] = {"mu": 0.5, "sigma": 0.3}

        # CUS: direct mode — value IS the probability, no z-score needed.
        # CUS has a systematic distributional shift between prefill and generation
        # that makes z-scoring unsound. CUS ∈ [0,1] with semantic meaning.
        stats["cus"] = {"mode": "direct"}

        # POS: use JSD stats from PrefillStatisticsHook (already computed and
        # stored via set_prompt_jsd_stats). Apply MAD-based robustness.
        # POS override is bounded [0,1], use JSD variance as proxy for POS variance.
        if hasattr(self.jsd_signal, '_prompt_jsd_mu') and self.jsd_signal._prompt_jsd_mu is not None:
            pos_mu = self.jsd_signal._prompt_jsd_mu
            pos_sigma = self.jsd_signal._prompt_jsd_sigma if self.jsd_signal._prompt_jsd_sigma else 0.1
            stats["pos"] = {"mu": pos_mu, "sigma": max(pos_sigma, 0.01)}
        else:
            stats["pos"] = {"mu": 0.0, "sigma": 0.1}

        return stats

    def _prefill(
        self,
        input_ids: Tensor,
        context_start: int,
        context_end: int,
        prompt_len: int,
    ) -> Tuple[Any, Tensor]:
        """
        Run prefill: identify copying heads, compute context SVD, collect prompt stats.
        Returns (past_key_values, last_logits).
        """
        # Adaptive window: sqrt(prompt_len)
        window_size = self._adaptive_window(prompt_len)

        # Save attention config
        orig_output_attentions = self.model.config.output_attentions
        orig_attn_impl = self.model.config._attn_implementation

        # Enable eager attention + output for CUS
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        # Install hidden state hooks
        hidden_hooks = []
        for layer_idx in self._hookable_layers:
            layer = self.model.model.layers[layer_idx]
            hook = LayerHooks(layer_idx, self.hidden_buffer)
            hook.install(layer)
            hidden_hooks.append(hook)

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

        # Hidden state capture hook for DPS prompt stats
        prefill_hidden_buffer = []

        def capture_hidden(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            prefill_hidden_buffer.append(hidden.detach())

        stats_layer = self.model.model.layers[stats_layer_idx]
        capture_handle = stats_layer.register_forward_hook(capture_hidden)

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

        # A. Copying heads from model output attentions (not hooks)
        if outputs.attentions is not None:
            affinities = {}
            for layer_idx in self._hookable_layers:
                if layer_idx < len(outputs.attentions):
                    attn = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
                    affinity = compute_layer_affinity(attn, context_start, context_end)
                    affinities[layer_idx] = affinity.detach().cpu()

            copying_heads, head_affinities = identify_copying_heads(affinities)
            self.cus_signal = ContextUtilizationSignal(
                copying_heads, prompt_len,
                n_layers=self.num_layers,
                head_affinities=head_affinities,
            )

        # B. Context SVD (DPS)
        if context_buffer:
            self.dps_signal.set_context_basis(context_buffer[0])
            self.jsd_signal.set_context_basis(self.dps_signal.context_basis)

        # C. Prompt statistics
        prefill_stats = stats_hook.compute_statistics(prompt_len)
        jsd_mu = prefill_stats.mu.get("pos", 0.0)
        jsd_sigma = prefill_stats.sigma.get("pos", 1.0)
        self.jsd_signal.set_prompt_jsd_stats(jsd_mu, jsd_sigma)

        # Self-calibrate: derive all prompt stats from actual prefill data
        self._prompt_stats = self._self_calibrate(
            outputs, context_start, context_end, prompt_len
        )

        # Clean up prefill hidden states
        self._prefill_hidden = None

        # Remove prefill hooks
        for h in hidden_hooks:
            h.remove()
        context_hook.remove()
        stats_hook.remove()
        self.hidden_buffer.clear()

        # Restore attention config, then set up for generation
        self.model.config._attn_implementation = orig_attn_impl
        self.model.config.output_attentions = orig_output_attentions

        # For generation: keep output_attentions=True for CUS
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        # Install generation hidden state hooks
        for layer_idx in self._hookable_layers:
            layer = self.model.model.layers[layer_idx]
            hook = LayerHooks(layer_idx, self.hidden_buffer)
            hook.install(layer)
            self._hooks.append(hook)

        return past_key_values, last_logits

    def compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        attentions: Tuple = None,
    ) -> DSGTokenSignals:
        """Compute all 3 DSG signals for a single generation token."""
        layer_states = self.hidden_buffer.get_states()

        # Adaptive topk
        self.candidate_manager.topk = CandidateSetManager.adaptive_topk(logits)
        candidate_set = self.candidate_manager.build(logits, emitted_token_id)

        # DPS — with adaptive layer selection when available
        dps_hidden = {idx: states.h_resid_mlp for idx, states in layer_states.items()}
        dps = self.dps_signal.compute_dps(
            dps_hidden, self.num_layers,
            active_layers=self._dps_active_layers,
        )

        # POS
        pos = self.jsd_signal.compute_pos(layer_states, candidate_set)

        # CUS — from model output attentions
        cus = 0.0
        if attentions is not None:
            # Build attention slices: layer_idx -> [num_heads, kv_len]
            attn_slices = {}
            for layer_idx in self._hookable_layers:
                if layer_idx < len(attentions) and attentions[layer_idx] is not None:
                    attn = attentions[layer_idx]  # [batch, heads, seq_len, kv_len]
                    attn_slices[layer_idx] = attn[0, :, -1, :]  # last position
            cus = self.cus_signal.compute_cus(attn_slices)

        # Clear hidden buffer
        self.hidden_buffer.clear()

        return DSGTokenSignals(cus=cus, pos=pos, dps=dps)

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

        Args:
            question: The question/prompt
            context: Context for grounded generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            DetectionResult with risk scores
        """
        self.candidate_manager.reset()

        input_ids, context_start, context_end, prompt_len = self._build_input(
            context, question
        )
        attention_mask = torch.ones_like(input_ids)

        try:
            # Prefill
            past_key_values, next_logits = self._prefill(
                input_ids, context_start, context_end, prompt_len
            )

            all_ids = input_ids.clone()
            token_results: List[DSGTokenSignals] = []

            # First token — no attentions available from prefill output
            # (prefill attentions were used for copying head identification)
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

                # Pass attentions from model output for CUS
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
        }

        agg_result = self.aggregator.compute_risk(self._prompt_stats, response_signals)
        token_risks = agg_result.token_risks.tolist()
        response_risk = agg_result.risk

        # Spans — adaptive threshold
        merger = SpanMerger.adaptive(token_risks)
        risky_spans = merger.find_spans(token_risks)

        # Use conformal threshold if calibrated, else fixed threshold
        if self.conformal is not None and self.conformal._threshold is not None:
            is_flagged, conf_threshold = self.conformal.predict(response_risk)
        else:
            is_flagged = response_risk >= self.config.response_flag_threshold
            conf_threshold = None

        return DetectionResult(
            generated_text=generated_text,
            token_signals=token_results,
            token_risks=token_risks,
            risky_spans=risky_spans,
            response_risk=response_risk,
            is_flagged=is_flagged,
            conformal_threshold=conf_threshold,
            num_tokens=len(token_results),
            prompt_length=prompt_len,
        )
