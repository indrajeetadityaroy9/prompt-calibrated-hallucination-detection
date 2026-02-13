"""
AG-SAR Engine: The Core Pipeline.

Architectural maturity:
- Type-safe interfaces
- Pipeline pattern
- Separation of concerns
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from .config import DetectorConfig, TokenSignals, DetectionResult
from .hooks import (
    EphemeralHiddenBuffer,
    LayerHooks,
    PrefillContextHook,
    PrefillStatisticsHook,
    PrefillStatistics,
)
from .signals import (
    CandidateJSDSignal,
    ContextGroundingSignal,
)
from .aggregation.span_merger import SpanMerger
from .aggregation.prompt_anchored import PromptAnchoredAggregator



class CandidateSetManager:
    """
    Manages candidate set construction for each token.

    Candidate set includes:
    - Top-k tokens from current step's logits
    - Emitted token (always)
    - Previous step's top-k (optional)
    """

    def __init__(self, config: DetectorConfig):
        """
        Initialize candidate set manager.

        Args:
            config: Detector configuration
        """
        self.topk = config.candidate_topk
        self.include_prev = config.include_prev_topk
        self._prev_topk: Optional[Tensor] = None

    def build(self, logits: Tensor, emitted_token_id: int) -> Tensor:
        """
        Build candidate set for current token.

        Args:
            logits: Full vocabulary logits [vocab_size]
            emitted_token_id: The token ID that was/will be emitted

        Returns:
            Candidate set indices [candidate_size]
        """
        # Get top-k from current logits
        current_topk = torch.topk(logits, min(self.topk, len(logits))).indices

        # Start with current top-k
        candidates = set(current_topk.tolist())

        # Add previous step's top-k
        if self.include_prev and self._prev_topk is not None:
            candidates.update(self._prev_topk.tolist())

        # Always include emitted token
        candidates.add(emitted_token_id)

        # Update previous top-k for next step
        self._prev_topk = current_topk

        # Convert to sorted tensor
        candidate_list = sorted(list(candidates))
        return torch.tensor(candidate_list, device=logits.device, dtype=torch.long)

    def reset(self):
        """Reset state (call at start of new sequence)."""
        self._prev_topk = None


class AGSAR:
    """
    AG-SAR: Authorization & Grounding via Signal Analysis & Retrieval.
    
    The core engine for zero-shot hallucination detection.
    Implements the pipeline: Input -> Model(Hooks) -> SignalExtraction -> Aggregation -> RiskScore.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: Optional[DetectorConfig] = None,
    ):
        """
        Initialize the AG-SAR engine.

        Args:
            model: HuggingFace LLaMA model
            tokenizer: HuggingFace tokenizer
            config: Detector configuration (optional, uses defaults if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DetectorConfig()
        self.device = next(model.parameters()).device

        # Get lm_head and final_norm for proper Logit Lens projection
        self.lm_head = model.lm_head
        # CRITICAL: Use model's learned final LayerNorm (with γ weights)
        # for proper Logit Lens projection. The lm_head expects features
        # normalized by this layer, not by parameter-free rms_norm.
        self.final_norm = model.model.norm

        # Determine which layers to hook
        self._hookable_layers = self._get_layer_indices()

        # Initialize components
        self.buffer = EphemeralHiddenBuffer()
        self.candidate_manager = CandidateSetManager(self.config)

        # Initialize signal computers
        self._init_signal_computers()

        # Initialize aggregator (Prompt-Anchored - The Standard Model, ICML-ready)
        # TODO: Update active_signals for DSG (cus, pos, dps)
        active_signals = set(self.config.prompt_anchored_signals)
        self.anchored_aggregator = PromptAnchoredAggregator(active_signals=active_signals)

        # Span Merger (threshold set dynamically if not configured)
        # Default threshold will be computed from token risk distribution
        self._span_merger_threshold = self.config.span_build_threshold

        # State
        self._hooks: List[LayerHooks] = []
        self._context_hook: Optional[PrefillContextHook] = None
        self._prefill_context_buffer: List[Tensor] = []
        self._prefill_stats_hook: Optional[PrefillStatisticsHook] = None
        self._prefill_statistics: Optional[PrefillStatistics] = None


        print(
            f"Initialized AG-SAR Engine with {len(self._hookable_layers)} layers"
        )

    def _get_layer_indices(self) -> List[int]:
        """Get layer indices to hook based on configuration."""
        num_layers = len(self.model.model.layers)

        if self.config.layer_subset == "last_third":
            start = num_layers - (num_layers // 3)
            layer_subset = list(range(start, num_layers))
        elif self.config.layer_subset == "last_quarter":
            start = num_layers - (num_layers // 4)
            layer_subset = list(range(start, num_layers))
        elif self.config.layer_subset == "all":
            layer_subset = list(range(num_layers))
        elif isinstance(self.config.layer_subset, list):
            layer_subset = self.config.layer_subset
        else:
            raise ValueError(f"Unknown layer_subset: {self.config.layer_subset}")

        # Handle 70B multi-GPU mode
        if self._is_multi_gpu():
            layer_subset = self._apply_multi_gpu_intersection(layer_subset)

        return layer_subset

    def _is_multi_gpu(self) -> bool:
        """Check if model is distributed across multiple GPUs."""
        devices = set()
        for layer in self.model.model.layers:
            devices.add(next(layer.parameters()).device)
        return len(devices) > 1

    def _apply_multi_gpu_intersection(self, layer_subset: List[int]) -> List[int]:
        """
        Apply 70B compatible-layers-only intersection.
        """
        lm_head_device = next(self.lm_head.parameters()).device

        # Find layers on same device as lm_head
        compatible_layers = set()
        for idx, layer in enumerate(self.model.model.layers):
            layer_device = next(layer.parameters()).device
            if layer_device == lm_head_device:
                compatible_layers.add(idx)

        # Intersection
        layer_subset_set = set(layer_subset)
        hook_layers = layer_subset_set & compatible_layers

        if not hook_layers:
            if self.config.multi_gpu_mode == "compatible_only":
                if compatible_layers:
                    print(
                        f"No overlap between layer_subset ({sorted(layer_subset_set)}) "
                        f"and compatible_layers ({sorted(compatible_layers)}). "
                        f"Falling back to compatible_layers only."
                    )
                    hook_layers = compatible_layers
                else:
                    raise RuntimeError(
                        "No layers colocated with lm_head. "
                        "Set multi_gpu_mode='allow_transfer' or use single-GPU."
                    )
            else:  # allow_transfer
                print(
                    f"Using layer_subset ({sorted(layer_subset_set)}) with "
                    f"cross-device transfer (may impact performance)."
                )
                hook_layers = layer_subset_set

        result = sorted(hook_layers)
        print(f"70B mode: hooking {len(result)} layers on {lm_head_device}")
        return result

    def _init_signal_computers(self):
        """Initialize signal computation modules.

        TODO: Replace with DSG signal initialization (CUS, POS, DPS).
        Currently only initializes JSD signal (retained for POS).
        """
        self.jsd_signal = CandidateJSDSignal(self.lm_head, self.final_norm)

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
        """
        Build input_ids from separately tokenized segments.

        Returns (input_ids, context_start_idx, context_end_idx).
        """
        # Tokenize each segment separately (no special tokens)
        prefix_tokens = self.tokenizer.encode("Context: ", add_special_tokens=False)

        if context:
            context_tokens = self.tokenizer.encode(context, add_special_tokens=False)
        else:
            context_tokens = []

        separator_tokens = self.tokenizer.encode("\n\nQuestion: ", add_special_tokens=False)
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode("\n\nAnswer:", add_special_tokens=False)

        # Handle BOS token - some models (e.g., Qwen2) don't have bos_token_id
        bos_token = self.tokenizer.bos_token_id
        if bos_token is None:
            # Try to use a reasonable alternative
            if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
                bos_token = self.tokenizer.cls_token_id
            else:
                # No BOS token available - skip it
                bos_token = None

        # Compute context boundaries BEFORE concatenation
        bos_len = 1 if bos_token is not None else 0
        prefix_len = len(prefix_tokens)

        context_start = bos_len + prefix_len
        context_end = context_start + len(context_tokens)

        # Concatenate with optional BOS
        if bos_token is not None:
            all_tokens = (
                [bos_token]
                + prefix_tokens
                + context_tokens
                + separator_tokens
                + question_tokens
                + suffix_tokens
            )
        else:
            all_tokens = (
                prefix_tokens
                + context_tokens
                + separator_tokens
                + question_tokens
                + suffix_tokens
            )

        input_ids = torch.tensor([all_tokens], dtype=torch.long)

        return input_ids, context_start, context_end

    def _compute_token_signals(
        self,
        logits: Tensor,
        emitted_token_id: int,
        token_hidden: Optional[Tensor] = None,
    ) -> TokenSignals:
        """
        Compute all signals for a token.

        TODO: Replace with DSG signal computation (CUS, POS, DPS).
        Currently a stub that computes JSD only.
        """
        # Build candidate set
        candidate_set = self.candidate_manager.build(logits, emitted_token_id)

        # Get layer states from buffer
        layer_states = self.buffer.get_states()

        # Compute JSD across layers (retained for POS)
        jsd_value = self.jsd_signal.compute_aggregated(
            layer_states, candidate_set, aggregation="mean"
        ) if layer_states else 0.0

        # Clear buffer after computing signals
        self.buffer.clear()

        return TokenSignals(jsd_cand=jsd_value)

    def _aggregate_results(
        self,
        token_results: List[TokenSignals],
        all_ids: Tensor,
        prompt_length: int,
        generated_text: str,
        prompt_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> DetectionResult:
        """
        Aggregate token-level signals into detection result.

        Uses prompt-anchored aggregation (The Standard Model) if configured,
        otherwise falls back to legacy Noisy-OR.

        Args:
            token_results: List of TokenSignals for each generated token
            all_ids: Full sequence tensor (prompt + generated)
            prompt_length: Length of prompt in tokens
            generated_text: Decoded generated text
            prompt_stats: Statistics from prompt (μ, σ per signal) for anchored normalization
        """
        # === Prompt-Anchored Aggregation (The Standard Model) ===
        # Collect response signals into arrays
        response_signals = self._collect_signal_arrays(token_results)

        # Use provided prompt_stats or empty (aggregator will handle gracefully)
        if prompt_stats is None:
            prompt_stats = {}

        # Compute risk using anchored aggregator
        agg_result = self.anchored_aggregator.compute_risk(prompt_stats, response_signals)

        # Extract token risks and response risk from aggregator
        token_risks = agg_result.token_risks.tolist() if len(agg_result.token_risks) > 0 else []
        response_risk = agg_result.risk

        print(
            f"Prompt-anchored aggregation: risk={response_risk:.3f}, "
            f"anchor_stats={agg_result.anchor_stats}"
        )

        # Detect Risky Spans (dynamic threshold if not configured)
        span_threshold = self._span_merger_threshold
        if span_threshold is None and token_risks:
            # Dynamic threshold: use median + 1 std of token risks
            mean_risk = sum(token_risks) / len(token_risks)
            var_risk = sum((r - mean_risk) ** 2 for r in token_risks) / len(token_risks)
            std_risk = var_risk ** 0.5
            span_threshold = min(0.9, mean_risk + std_risk)  # Capped at 0.9 for stability

        if span_threshold is not None and span_threshold > 0:
            span_merger = SpanMerger(threshold=span_threshold, min_span_length=1, max_gap=1)
            risky_spans = span_merger.find_spans(token_risks)
        else:
            risky_spans = []

        # Dynamic response flagging threshold
        response_flag_threshold = self.config.response_flag_threshold
        if response_flag_threshold is None:
            # Default: flag if max token risk > 0.5 (more than 50% probability)
            response_flag_threshold = 0.5

        # Compute topk_mass statistics
        topk_masses = [s.topk_mass for s in token_results if s.topk_mass is not None]
        mean_topk_mass = sum(topk_masses) / len(topk_masses) if topk_masses else None
        min_topk_mass = min(topk_masses) if topk_masses else None

        return DetectionResult(
            generated_text=generated_text,
            token_signals=token_results,
            token_risks=token_risks,
            risky_spans=risky_spans,
            response_risk=response_risk,
            is_flagged=response_risk >= response_flag_threshold,
            num_tokens=len(token_results),
            num_claim_tokens=len(token_results),
            prompt_length=prompt_length,
            mean_topk_mass=mean_topk_mass,
            min_topk_mass=min_topk_mass,
        )

    def _convert_prefill_stats_to_signals(
        self, prefill_stats: PrefillStatistics
    ) -> Dict[str, Dict[str, float]]:
        """
        Convert PrefillStatistics to format expected by aggregator.

        Instead of passing raw signal arrays (which the aggregator would
        compute μ/σ from), we pass the pre-computed statistics directly.
        The aggregator's _compute_anchor_stats method is updated to accept this.

        Args:
            prefill_stats: Statistics computed from prompt tail

        Returns:
            Dict mapping signal name to {"mu_prompt": float, "sigma_prompt": float}
        """
        result = {}
        for sig in prefill_stats.signals:
            if sig in prefill_stats.mu and sig in prefill_stats.sigma:
                result[sig] = {
                    "mu_prompt": prefill_stats.mu[sig],
                    "sigma_prompt": prefill_stats.sigma[sig],
                    "n_tokens": prefill_stats.n_tokens,
                }
        return result

    def _collect_signal_arrays(
        self, token_results: List[TokenSignals]
    ) -> Dict[str, np.ndarray]:
        """
        Collect signal values from token results into arrays for aggregation.
        """
        signals = self.config.prompt_anchored_signals
        signal_arrays = {sig: [] for sig in signals}

        # Map signal names to TokenSignals attributes
        # TODO: Update for DSG signals (cus, pos, dps)
        signal_attr_map = {
            "jsd_cand": "jsd_cand",
            "jsd": "jsd_cand",  # Alias
        }

        for token_sig in token_results:
            for sig in signals:
                attr = signal_attr_map.get(sig, sig)
                value = getattr(token_sig, attr, None)
                if value is not None:
                    signal_arrays[sig].append(value)
                else:
                    signal_arrays[sig].append(0.0)

        return {sig: np.array(vals) for sig, vals in signal_arrays.items()}

    def generate(
        self,
        prompt: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> DetectionResult:
        """
        Generate text with hallucination detection (The Pipeline).

        Args:
            prompt: The question/prompt to answer
            context: Optional context for grounded generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            DetectionResult with risk scores
        """
        # === STANDARD PIPELINE ===

        # 1. Setup
        self.candidate_manager.reset()
        self._prefill_statistics = None  # Reset stale statistics from previous calls

        input_ids, context_start, context_end = self._build_input_from_segments(
            context, prompt
        )
        input_ids = input_ids.to(self.device)
        prompt_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        # 2. Hooks
        self._install_hooks()

        # Install prefill statistics hook for prompt-anchored normalization
        if self.config.use_prompt_anchored:
            # Use a layer near the end for statistics (last third)
            stats_layer_idx = max(self._hookable_layers) if self._hookable_layers else len(self.model.model.layers) - 1
            self._prefill_stats_hook = PrefillStatisticsHook(
                lm_head=self.lm_head,
                final_norm=self.final_norm,
                window_size=64,  # Tail sampling: last 64 tokens
                signals=self.config.prompt_anchored_signals,
            )
            self._prefill_stats_hook.install(self.model.model.layers[stats_layer_idx])

        if context and context_end > context_start:
            # Install temp hook for context
            # TODO: Configure context hook layer for DSG
            layer_idx = len(self.model.model.layers) // 2
            self._prefill_context_buffer = []
            self._context_hook = PrefillContextHook(context_start, context_end, self._prefill_context_buffer)
            self._context_hook.install(self.model.model.layers[layer_idx])

        # 3. Generation Loop
        all_ids = input_ids.clone()
        token_results: List[TokenSignals] = []
        cache_position = torch.arange(prompt_length, device=self.device)

        try:
            # Prefill
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            # Compute prompt statistics for prompt-anchored normalization
            if self.config.use_prompt_anchored and self._prefill_stats_hook is not None:
                self._prefill_statistics = self._prefill_stats_hook.compute_statistics(prompt_length)
                self._prefill_stats_hook.remove()
                self._prefill_stats_hook = None
                print(
                    f"Computed prefill statistics from {self._prefill_statistics.n_tokens} tokens: "
                    f"μ={self._prefill_statistics.mu}, σ={self._prefill_statistics.sigma}"
                )

            # Cache context embeddings from prefill
            # TODO: Replace with DSG context subspace computation
            if context and context_end > context_start and self._prefill_context_buffer:
                self._prefill_context_buffer = []
                if self._context_hook:
                    self._context_hook.remove()
                    self._context_hook = None

            # First token
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            token_signals = self._compute_token_signals(next_logits.squeeze(0), next_token.item())
            token_results.append(token_signals)

            all_ids = torch.cat([all_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)
            cache_position = torch.tensor([prompt_length], device=self.device)

            # Decode
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

                token_signals = self._compute_token_signals(next_logits.squeeze(0), next_token.item())
                token_results.append(token_signals)

                all_ids = torch.cat([all_ids, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)], dim=1)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        finally:
            self._remove_hooks()

        # 4. Post-processing
        generated_text = self.tokenizer.decode(all_ids[0, prompt_length:], skip_special_tokens=True)

        # 5. Aggregation
        # Convert prefill statistics to prompt_stats format for aggregator
        prompt_stats = None
        if self._prefill_statistics is not None:
            prompt_stats = self._convert_prefill_stats_to_signals(self._prefill_statistics)

        return self._aggregate_results(
            token_results, all_ids, prompt_length, generated_text,
            prompt_stats=prompt_stats,
        )

