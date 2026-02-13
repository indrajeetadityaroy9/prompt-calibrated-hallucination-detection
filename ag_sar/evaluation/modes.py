"""
Evaluation modes for hallucination detection.

Two modes:
- FORCED_DECODING: Stepwise with ground-truth tokens (for RAGTruth span labels)
- GENERATION: Standard argmax generation (for HaluEval response labels)
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple
import torch
from torch import Tensor

from ..config import TokenSignals, DetectorConfig
from ..hooks import EphemeralHiddenBuffer, LayerHooks, PrefillContextHook
from ..signals import CandidateJSDSignal, ContextGroundingSignal



class EvaluationMode(Enum):
    """Evaluation mode selection."""
    FORCED_DECODING = "forced_decoding"  # Stepwise with ground-truth tokens
    GENERATION = "generation"  # Standard argmax generation


class ForcedDecodingEvaluator:
    """
    Score a PROVIDED response using FORCED DECODING.

    This is "stepwise teacher forcing" - we iterate through the provided response
    tokens one at a time, feeding the ground-truth next token (instead of argmax).

    This approach:
    - Uses the SAME hook/buffer logic as generation mode
    - Ensures signals computed during evaluation match deployment behavior
    - Requires multiple forward passes (one per response token) - acceptable for evaluation

    Process:
    1. Prefill on prompt + context (same as generation)
    2. For each response token position:
       a. Run forward pass with KV-cache
       b. Capture 3-point hidden states via hooks
       c. Compute signals using candidate set from logits
       d. Feed ground-truth next token (not argmax)
    3. Return per-token signals for alignment with labels
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: DetectorConfig,
        signal_computer=None,  # TODO: Replace with DSG signal computer
        candidate_manager=None,
    ):
        """
        Initialize forced decoding evaluator.

        Args:
            model: HuggingFace LLaMA model
            tokenizer: HuggingFace tokenizer
            config: Detector configuration
            signal_computer: Signal computer instance
            candidate_manager: Candidate set manager instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.signal_computer = signal_computer
        self.candidate_manager = candidate_manager
        self.device = next(model.parameters()).device

        # Get hookable layers
        self._hookable_layers = self._get_hookable_layers()

        # State
        self.buffer = EphemeralHiddenBuffer()
        self._hooks: List[LayerHooks] = []
        self._context_hook: Optional[PrefillContextHook] = None
        self._prefill_context_buffer: List[Tensor] = []

    def _get_hookable_layers(self) -> List[int]:
        """Get layer indices to hook."""
        num_layers = len(self.model.model.layers)

        if self.config.layer_subset == "last_third":
            start = num_layers - (num_layers // 3)
            return list(range(start, num_layers))
        elif self.config.layer_subset == "last_quarter":
            start = num_layers - (num_layers // 4)
            return list(range(start, num_layers))
        elif self.config.layer_subset == "all":
            return list(range(num_layers))
        elif isinstance(self.config.layer_subset, list):
            return self.config.layer_subset
        else:
            raise ValueError(f"Unknown layer_subset: {self.config.layer_subset}")

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

        if self._context_hook is not None:
            self._context_hook.remove()
            self._context_hook = None

    def _build_input_from_segments(
        self,
        context: Optional[str],
        question: str,
    ) -> Tuple[Tensor, int, int]:
        """Build input_ids from separately tokenized segments."""
        prefix_tokens = self.tokenizer.encode("Context: ", add_special_tokens=False)

        if context:
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        else:
            context_tokens = []

        separator_tokens = self.tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

        bos_len = 1
        prefix_len = len(prefix_tokens)

        context_start = bos_len + prefix_len
        context_end = context_start + len(context_tokens)

        all_tokens = (
            [self.tokenizer.bos_token_id]
            + prefix_tokens
            + context_tokens
            + separator_tokens
            + question_tokens
            + suffix_tokens
        )

        input_ids = torch.tensor([all_tokens], dtype=torch.long)
        return input_ids, context_start, context_end

    def _install_context_hook(self, context_start: int, context_end: int):
        """Install temporary hook to capture context during prefill."""
        # TODO: Configure context hook layer for DSG
        layer_idx = len(self.model.model.layers) // 2

        layer = self.model.model.layers[layer_idx]
        self._prefill_context_buffer = []
        self._context_hook = PrefillContextHook(
            context_start, context_end, self._prefill_context_buffer
        )
        self._context_hook.install(layer)

    def evaluate(
        self,
        context: str,
        question: str,
        response: str,
    ) -> List[TokenSignals]:
        """
        Evaluate a provided response using forced decoding.

        Args:
            context: The context string
            question: The question/prompt
            response: The response to evaluate (ground-truth)

        Returns:
            List of TokenSignals, one per response token
        """
        # Reset state
        self.candidate_manager.reset()

        # Tokenize response
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)

        if not response_tokens:
            return []

        # Build prompt input
        input_ids, context_start, context_end = self._build_input_from_segments(
            context, question
        )
        input_ids = input_ids.to(self.device)

        # Initialize attention_mask
        attention_mask = torch.ones((1, input_ids.shape[1]), device=self.device)

        # Install hooks
        self._install_hooks()

        # Install context hook if we have context
        if context and context_end > context_start:
            self._install_context_hook(context_start, context_end)

        token_results: List[TokenSignals] = []

        try:
            # === PREFILL ===
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values

            # Cache context embeddings
            # TODO: Replace with DSG context subspace computation
            self._prefill_context_buffer = []
            if self._context_hook is not None:
                self._context_hook.remove()
                self._context_hook = None

            # Clear buffer from prefill
            self.buffer.clear()

            # Initialize cache_position for decode loop
            prompt_length = input_ids.shape[1]
            cache_position = torch.tensor([prompt_length - 1], device=self.device)

            # === FORCED DECODING LOOP ===
            for i, gt_token_id in enumerate(response_tokens):
                # Get logits from current outputs
                next_logits = outputs.logits[:, -1, :]
                gt_token = torch.tensor([[gt_token_id]], device=self.device)

                # Build candidate set and compute signals
                candidate_set = self.candidate_manager.build(
                    next_logits.squeeze(0), gt_token_id
                )

                # Get layer states from buffer
                layer_states = self.buffer.get_states()

                # Compute signals
                # TODO: Replace with DSG signal computation (CUS, POS, DPS)
                jsd_value = 0.0
                if self.signal_computer is not None:
                    jsd_value = self.signal_computer.compute_aggregated(
                        layer_states, candidate_set, aggregation="mean"
                    ) if layer_states else 0.0

                token_results.append(TokenSignals(jsd_cand=jsd_value))

                # Clear buffer
                self.buffer.clear()

                # Update cache position
                cache_position = cache_position + 1

                # Feed ground-truth token using prepare_inputs_for_generation
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.device)
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

        return token_results


class GenerationEvaluator:
    """
    Score Llama's own GENERATED response.

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
            detector: HallucinationDetector instance
        """
        self.detector = detector

    def evaluate(
        self,
        context: str,
        question: str,
        max_new_tokens: int = 256,
    ):
        """
        Generate and evaluate response.

        Args:
            context: The context string
            question: The question/prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            DetectionResult from the detector
        """
        return self.detector.generate(
            prompt=question,
            context=context,
            max_new_tokens=max_new_tokens,
        )
