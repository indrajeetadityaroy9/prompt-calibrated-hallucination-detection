"""
Evaluation modes for hallucination detection.

Two modes:
- FORCED_DECODING: Stepwise with ground-truth tokens (for RAGTruth span labels)
- GENERATION: Standard argmax generation (for HaluEval response labels)
"""

from enum import Enum
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
from torch import Tensor

from ..config import DSGConfig, DSGTokenSignals, DetectionResult
from ..hooks import EphemeralHiddenBuffer, LayerHooks, PrefillContextHook, PrefillStatisticsHook
from ..signals.context_grounding import DualSubspaceGrounding
from ..signals.topk_jsd import CandidateJSDSignal
from ..signals.copying_heads import (
    identify_copying_heads,
    compute_layer_affinity,
    ContextUtilizationSignal,
)
from ..icml.dsg_detector import CandidateSetManager


class EvaluationMode(Enum):
    """Evaluation mode selection."""
    FORCED_DECODING = "forced_decoding"  # Stepwise with ground-truth tokens
    GENERATION = "generation"  # Standard argmax generation


class ForcedDecodingEvaluator:
    """
    Score a PROVIDED response using FORCED DECODING with full DSG signals.

    This is "stepwise teacher forcing" - we iterate through the provided response
    tokens one at a time, feeding the ground-truth next token (instead of argmax).

    Mirrors the DSGDetector pipeline:
    1. Prefill: identify copying heads, compute context SVD, collect prompt stats
    2. For each response token:
       a. Run forward pass with KV-cache (output_attentions=True)
       b. Capture 3-point hidden states via hooks
       c. Compute CUS, POS, DPS from hidden states, attentions, and logits
       d. Feed ground-truth next token (not argmax)
    3. Return per-token DSGTokenSignals for alignment with labels
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: DSGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = next(model.parameters()).device

        self.lm_head = model.lm_head
        self.final_norm = model.model.norm
        self.num_layers = len(model.model.layers)

        # Get hookable layers
        self._hookable_layers = self._get_hookable_layers()

        # Signal computers (DPS + POS are model-level; CUS is per-input)
        self.dps_signal = DualSubspaceGrounding(lm_head_weight=self.lm_head.weight)
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)
        self.candidate_manager = CandidateSetManager(topk=128)

        # State (per-input)
        self.buffer = EphemeralHiddenBuffer()
        self.cus_signal: ContextUtilizationSignal = None
        self._hooks: List[LayerHooks] = []
        self._prompt_stats: Dict[str, Dict[str, float]] = {}

    def _get_hookable_layers(self) -> List[int]:
        """Get layer indices to hook."""
        if self.config.layer_subset == "last_third":
            start = self.num_layers - (self.num_layers // 3)
            return list(range(start, self.num_layers))
        elif self.config.layer_subset == "last_quarter":
            start = self.num_layers - (self.num_layers // 4)
            return list(range(start, self.num_layers))
        elif self.config.layer_subset == "all":
            return list(range(self.num_layers))
        elif isinstance(self.config.layer_subset, list):
            return self.config.layer_subset
        return list(range(self.num_layers))

    def _install_hooks(self):
        """Install hooks on all specified layers."""
        for layer_idx in self._hookable_layers:
            layer = self.model.model.layers[layer_idx]
            hook = LayerHooks(layer_idx, self.buffer)
            hook.install(layer)
            self._hooks.append(hook)

    def _remove_hooks(self):
        """Remove all installed hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _build_input_from_segments(
        self,
        context: str,
        question: str,
    ) -> Tuple[Tensor, int, int]:
        """Build input_ids from separately tokenized segments."""
        prefix_tokens = self.tokenizer.encode("Context: ", add_special_tokens=False)
        context_tokens = self.tokenizer.encode(context, add_special_tokens=False) if context else []
        separator_tokens = self.tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

        bos_id = self.tokenizer.bos_token_id
        bos_len = 1 if bos_id is not None else 0

        context_start = bos_len + len(prefix_tokens)
        context_end = context_start + len(context_tokens)

        all_tokens = (
            ([bos_id] if bos_id is not None else [])
            + prefix_tokens
            + context_tokens
            + separator_tokens
            + question_tokens
            + suffix_tokens
        )

        input_ids = torch.tensor([all_tokens], dtype=torch.long)
        return input_ids, context_start, context_end

    def _compute_step_signals(
        self,
        logits: Tensor,
        gt_token_id: int,
        attentions: Tuple = None,
    ) -> DSGTokenSignals:
        """Compute all 3 DSG signals for a single forced-decoding step."""
        layer_states = self.buffer.get_states()

        # Adaptive topk + candidate set
        self.candidate_manager.topk = CandidateSetManager.adaptive_topk(logits)
        candidate_set = self.candidate_manager.build(logits, gt_token_id)

        # DPS
        dps_hidden = {idx: states.h_resid_mlp for idx, states in layer_states.items()}
        dps = self.dps_signal.compute_dps(dps_hidden, self.num_layers)

        # POS
        pos = self.jsd_signal.compute_pos(layer_states, candidate_set)

        # CUS from attention outputs
        cus = 0.0
        if attentions is not None and self.cus_signal is not None:
            attn_slices = {}
            for layer_idx in self._hookable_layers:
                if layer_idx < len(attentions) and attentions[layer_idx] is not None:
                    attn = attentions[layer_idx]  # [batch, heads, seq, kv]
                    attn_slices[layer_idx] = attn[0, :, -1, :]
            cus = self.cus_signal.compute_cus(attn_slices)

        self.buffer.clear()
        return DSGTokenSignals(cus=cus, pos=pos, dps=dps)

    def evaluate(
        self,
        context: str,
        question: str,
        response: str,
    ) -> List[DSGTokenSignals]:
        """
        Evaluate a provided response using forced decoding with full DSG signals.

        Args:
            context: The context string
            question: The question/prompt
            response: The response to evaluate (ground-truth)

        Returns:
            List of DSGTokenSignals, one per response token.
        """
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        if not response_tokens:
            return []

        input_ids, context_start, context_end = self._build_input_from_segments(context, question)
        input_ids = input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        # Save and set attention config
        orig_output_attentions = self.model.config.output_attentions
        orig_attn_impl = self.model.config._attn_implementation
        self.model.config._attn_implementation = "eager"
        self.model.config.output_attentions = True

        self.candidate_manager.reset()
        token_results: List[DSGTokenSignals] = []

        try:
            # === PHASE 1: PREFILL ===
            # Install hidden state hooks
            self._install_hooks()

            # Context hook for DPS
            context_buffer: List[Tensor] = []
            context_hook = None
            if context and context_end > context_start:
                mid_layer_idx = self.num_layers // 2
                context_hook = PrefillContextHook(context_start, context_end, context_buffer)
                context_hook.install(self.model.model.layers[mid_layer_idx])

            # POS statistics hook
            import math as _math
            window_size = max(16, min(int(_math.ceil(_math.sqrt(prompt_length))), prompt_length // 2, prompt_length))
            stats_layer_idx = max(self._hookable_layers)
            stats_hook = PrefillStatisticsHook(
                lm_head=self.lm_head,
                final_norm=self.final_norm,
                window_size=window_size,
            )
            stats_hook.install(self.model.model.layers[stats_layer_idx])

            # Run prefill
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values

            # A. Copying heads from attention outputs -> CUS
            if outputs.attentions is not None:
                affinities = {}
                for layer_idx in self._hookable_layers:
                    if layer_idx < len(outputs.attentions):
                        attn = outputs.attentions[layer_idx]
                        affinity = compute_layer_affinity(attn, context_start, context_end)
                        affinities[layer_idx] = affinity.detach().cpu()
                copying_heads, head_affinities = identify_copying_heads(affinities)
                self.cus_signal = ContextUtilizationSignal(
                    copying_heads, prompt_length,
                    n_layers=self.num_layers,
                    head_affinities=head_affinities,
                )

            # B. Context SVD -> DPS
            if context_buffer:
                self.dps_signal.set_context_basis(context_buffer[0])
                self.jsd_signal.set_context_basis(self.dps_signal.context_basis)

            # C. Prompt JSD statistics -> POS threshold
            prefill_stats = stats_hook.compute_statistics(prompt_length)
            jsd_mu = prefill_stats.mu.get("pos", 0.0)
            jsd_sigma = prefill_stats.sigma.get("pos", 1.0)
            self.jsd_signal.set_prompt_jsd_stats(jsd_mu, jsd_sigma)

            # Store prompt stats: CUS uses direct mode (no z-score),
            # POS/DPS use prompt-anchored z-scoring
            self._prompt_stats = {
                "cus": {"mode": "direct"},
                "pos": {"mu": jsd_mu, "sigma": max(jsd_sigma, 0.01)},
                "dps": {"mu": 0.5, "sigma": 0.3},  # DPS stats computed if hidden states available
            }

            # Remove prefill-only hooks
            if context_hook is not None:
                context_hook.remove()
            stats_hook.remove()
            self.buffer.clear()

            # === PHASE 2: FORCED DECODING LOOP ===
            cache_position = torch.tensor([prompt_length - 1], device=self.device)

            for i, gt_token_id in enumerate(response_tokens):
                next_logits = outputs.logits[:, -1, :]
                gt_token = torch.tensor([[gt_token_id]], device=self.device)

                # Compute DSG signals from this step's hidden states + attentions
                signals = self._compute_step_signals(
                    next_logits.squeeze(0), gt_token_id, attentions=outputs.attentions
                )
                token_results.append(signals)

                # Update cache position and feed ground-truth token
                cache_position = cache_position + 1
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
                ], dim=1)

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids=gt_token,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    cache_position=cache_position,
                    use_cache=True,
                )

                with torch.no_grad():
                    outputs = self.model(**model_inputs, return_dict=True)
                past_key_values = outputs.past_key_values

        finally:
            self._remove_hooks()
            self.model.config._attn_implementation = orig_attn_impl
            self.model.config.output_attentions = orig_output_attentions

        return token_results


class GenerationEvaluator:
    """
    Score Llama's own GENERATED response using DSGDetector.

    Labels not available from RAGTruth (would need re-annotation).

    Use for:
    - HaluEval (binary response-level labels, not span-specific)
    - Manual inspection
    - Downstream gating applications
    """

    def __init__(
        self,
        detector,
    ):
        """
        Initialize generation evaluator.

        Args:
            detector: DSGDetector instance
        """
        self.detector = detector

    def evaluate(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """
        Generate and evaluate response using DSGDetector.detect().

        Args:
            context: The context string
            question: The question/prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            DetectionResult from the detector
        """
        return self.detector.detect(
            prompt=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )
