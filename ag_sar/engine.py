"""
AG-SAR Engine - Single-pass hallucination detection.

Orchestrates attention extraction, authority flow, unified gating, and
semantic dispersion into a single forward pass with O(N) memory.
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn

from .config import AGSARConfig
from .modeling import ModelAdapter
from .measures import (
    compute_semantic_authority,
    compute_semantic_authority_v3,
    compute_semantic_authority_v4,
    compute_semantic_authority_v5,
    compute_semantic_authority_v6,
    compute_semantic_authority_v7,
    compute_semantic_authority_v8,
    compute_semantic_authority_v9,
    compute_semantic_authority_v10,
    compute_semantic_authority_v11,
    compute_semantic_authority_v12,
    compute_semantic_authority_v13,
    compute_semantic_authority_v15,
    compute_semantic_authority_v16,
    compute_semantic_authority_v19,
    compute_varentropy,
    compute_local_intrinsic_dimension,
)
from .utils import get_model_dtype, get_model_device


class AGSAR:
    """
    AG-SAR Uncertainty Quantification Engine.

    Detects extrinsic hallucinations (context violations) by analyzing
    attention patterns in a single forward pass.

    Core equation:
        Uncertainty(t) = 1 - Authority(t)
        Authority(t) = Gate(t) × Flow(t) + (1 - Gate(t)) × Trust(t)

    Example:
        >>> agsar = AGSAR(model, tokenizer)
        >>> uncertainty = agsar.compute_uncertainty(prompt, response)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[AGSARConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AGSARConfig()

        self.dtype = get_model_dtype(model)
        self.device = get_model_device(model)
        self._compute_device = self._get_compute_device(model)

        # Detect layer count and select semantic layers
        num_layers = self._detect_num_layers(model)
        start_layer = max(0, num_layers - self.config.semantic_layers)
        self._semantic_layer_indices = list(range(start_layer, num_layers))

        # Initialize adapter with hooks
        self._adapter = ModelAdapter(
            model=model,
            layers=self._semantic_layer_indices,
            dtype=self.dtype,
        )
        self._adapter.register()

        # Cache embedding matrix
        self._embed_matrix = self._extract_embedding_matrix(model)

        # Calibration state
        self._calibration: Optional[Dict[str, float]] = None

    def _detect_num_layers(self, model: nn.Module) -> int:
        if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
            return model.config.num_hidden_layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return len(model.transformer.h)
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return len(model.model.layers)
        return 12

    def _extract_embedding_matrix(self, model: nn.Module) -> Optional[torch.Tensor]:
        try:
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                return output_embeddings.weight.detach()
        except Exception:
            if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
                return model.lm_head.weight.detach()
        return None

    def _get_compute_device(self, model: nn.Module) -> torch.device:
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            device_map = model.hf_device_map
            for key in ['lm_head', 'embed_out', 'output']:
                if key in device_map:
                    device_id = device_map[key]
                    if isinstance(device_id, int):
                        return torch.device(f"cuda:{device_id}")
                    elif isinstance(device_id, str) and device_id != 'cpu':
                        return torch.device(device_id)
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return self.device

    def _tokenize(self, prompt: str, response: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not response or not response.strip():
            raise ValueError("response must be non-empty")

        prompt_enc = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
        response_enc = self.tokenizer(response, return_tensors='pt', add_special_tokens=False)

        input_ids = torch.cat([prompt_enc['input_ids'], response_enc['input_ids']], dim=1)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, prompt_enc['input_ids'].size(1)

    def _aggregate_authority(
        self,
        authority_tokens: torch.Tensor,
        logits: torch.Tensor,
    ) -> float:
        """
        Aggregate token-level authority to sequence-level score.

        Args:
            authority_tokens: (B, T) token-level authority scores
            logits: (B, T, V) logits for varentropy computation

        Returns:
            Aggregated authority score in [0, 1]
        """
        method = self.config.aggregation_method

        if method == "mean":
            # Standard statistical mean (Manakul et al., Kuhn et al.)
            # Most defensible baseline, reduces estimator variance
            return authority_tokens.mean().item()

        elif method == "importance_weighted":
            # Varentropy-weighted mean (SOTA agnostic approach)
            # Upweights "thinking" tokens (high V), downweights "filler" (low V)
            # Insight: High-V tokens are structural choice points where model decisions matter
            varentropy = compute_varentropy(logits)

            # Weights in [1.0, 2.0] range via tanh saturation
            weights = 1.0 + torch.tanh(varentropy)

            # Weighted mean
            weighted_sum = (authority_tokens * weights).sum()
            total_weight = weights.sum()

            return (weighted_sum / (total_weight + 1e-8)).item()

        elif method == "percentile_10":
            # Weakest-link aggregation (safety-critical mode)
            # If one token is bad, the whole sequence is flagged
            # High variance, but appropriate for API calls, math proofs
            return torch.quantile(authority_tokens.flatten(), 0.10).item()

        elif method == "min":
            # "Weakest Link" principle
            # If ANY token has low faithfulness, flag the entire sequence
            # Captures localized hallucination bursts (V spikes at fabrication)
            # Note: v4's compute_semantic_authority_v4 sets masked tokens to 1.0
            # so they don't affect the min calculation
            return authority_tokens.min().item()

        elif method == "geometric_mean":
            # "Spectral Coherence" - Principled Matrix Misalignment aggregation
            # Treats sequence as probabilistic chain of independent faithfulness.
            # Geometric mean = exp(mean(log(scores))) = "Average Likelihood of Truth"
            #
            # Key properties:
            # - [0.9, 0.9, 0.9] → 0.9 (same as mean for uniform)
            # - [0.9, 0.9, 0.1] → 0.43 (one lie punished, mean=0.63)
            # - [0.9, 0.9, 0.01] → 0.20 (fabrication crushed, mean=0.60)
            #
            # Mathematically: If scores are singular values of alignment matrix,
            # geometric mean measures "Volume of Truth" (determinant^(1/n)).
            # Hallucination is volume collapse (one σ_i → 0).
            eps = 1e-6
            scores = authority_tokens.float().clamp(min=eps)  # Prevent log(0)
            log_mean = scores.log().mean()
            return log_mean.exp().item()

        elif method == "sliding_window_min":
            # "Phrase-Level Weakest Link" - v9 recommended aggregation
            # Assumes hallucinations are SPANS (phrases), not single tokens.
            #
            # Key insight:
            # - Single noisy token (0.2) in a fact: 5-token window avg stays high (~0.8)
            # - 5-token lie phrase: window avg drops (~0.2), min catches it
            #
            # This balances:
            # - min: Too harsh on long facts (1 noisy token → low score)
            # - mean: Too lenient on short lies (3-token lie washed out in 50 tokens)
            window_size = 5

            # Handle short sequences (fallback to min)
            if authority_tokens.numel() < window_size:
                return authority_tokens.min().item()

            # Flatten for 1D unfold
            tokens_flat = authority_tokens.flatten()

            # Create sliding windows: [num_windows, window_size]
            windows = tokens_flat.unfold(dimension=0, size=window_size, step=1)

            # Mean within each window (smooths single-token noise)
            window_scores = windows.mean(dim=1)

            # Min across windows (catches any low-scoring phrase)
            return window_scores.min().item()

        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def calibrate_on_prompt(self, prompt: str) -> Dict[str, float]:
        """
        Calibrate adaptive thresholds from prompt statistics.

        Computes baseline dispersion, gate, and varentropy statistics
        to enable model-agnostic threshold adaptation.

        Args:
            prompt: Source context to calibrate on.

        Returns:
            Dict with calibration statistics and adaptive thresholds.
        """
        from .measures.semantics import compute_semantic_dispersion
        from .ops import compute_mlp_divergence, fused_stability_gate

        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")

        # Tokenize and forward
        prompt_enc = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=True)
        input_ids = prompt_enc['input_ids'].to(self._compute_device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # Get hidden states
        last_layer = self._semantic_layer_indices[-1]
        h_attn = self._adapter.capture.attn_outputs.get(last_layer)
        h_block = self._adapter.capture.block_outputs.get(last_layer)

        if h_attn is None or h_block is None:
            raise RuntimeError("Hidden states not captured")

        h_attn = h_attn.to(self._compute_device)
        h_block = h_block.to(self._compute_device)

        # Divergence statistics (for adaptive Z-score gating)
        divergence = compute_mlp_divergence(h_attn, h_block)
        div_mu = divergence.mean().item()
        div_sigma = max(divergence.std().item(), 0.01)

        # Legacy gate stats (kept for backward compatibility)
        gate_values = fused_stability_gate(h_attn, h_block, sensitivity=1.0)
        gate_mu = gate_values.mean().item()
        gate_sigma = max(gate_values.std().item(), 0.01)

        # Dispersion statistics (last N tokens for efficiency)
        window = min(input_ids.size(1), self.config.calibration_window)
        logits = outputs.logits[:, -window:, :]
        dispersion = compute_semantic_dispersion(logits, self._embed_matrix, k=5, method="nucleus_variance")
        disp_mu = dispersion.mean().item()
        disp_sigma = max(dispersion.std().item(), 0.01)

        # Varentropy statistics
        varentropy = compute_varentropy(logits)
        var_mu = varentropy.mean().item()
        var_sigma = max(varentropy.std().item(), 0.01)

        # LID statistics (v7)
        if h_block is not None:
            lid = compute_local_intrinsic_dimension(
                h_block[:, -window:, :],
                window_size=self.config.lid_window_size,
            )
            lid_mu = lid.mean().item()
            lid_sigma = max(lid.std().item(), 0.01)
        else:
            lid_mu, lid_sigma = 0.5, 0.1  # Fallback defaults

        # Compute adaptive thresholds using sigma multiplier
        sigma = self.config.sigma_multiplier
        self._calibration = {
            # Divergence stats for adaptive Z-score gating
            'divergence_mu': div_mu,
            'divergence_sigma': div_sigma,
            # Legacy gate stats
            'gate_mu': gate_mu,
            'gate_sigma': gate_sigma,
            'dispersion_mu': disp_mu,
            'dispersion_sigma': disp_sigma,
            'varentropy_mu': var_mu,
            'varentropy_sigma': var_sigma,
            'adaptive_cpg_gate_threshold': gate_mu + sigma * gate_sigma,
            # Dispersion threshold: flag when ABOVE baseline (use +1.0 sigma, not negative)
            'adaptive_cpg_dispersion_threshold': disp_mu + 1.0 * disp_sigma,
            'adaptive_cpg_varentropy_threshold': var_mu + sigma * var_sigma,
            # v3 Epiplexity stats (varentropy is the proxy)
            'epiplexity_mu': var_mu,
            'epiplexity_sigma': var_sigma,
            # v7 LID stats (geometric manifold adherence)
            'lid_mu': lid_mu,
            'lid_sigma': lid_sigma,
        }

        return self._calibration

    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        return_details: bool = False,
    ) -> Union[float, Dict[str, Any]]:
        """
        Compute uncertainty score for a prompt-response pair.

        Args:
            prompt: Source context/question.
            response: Generated text to evaluate.
            return_details: If True, return dict with component scores.

        Returns:
            float: Uncertainty in [0, 1]. Higher = more likely hallucination.
            Or dict with detailed breakdown if return_details=True.
        """
        input_ids, attention_mask, response_start = self._tokenize(prompt, response)
        input_ids = input_ids.to(self._compute_device)
        attention_mask = attention_mask.to(self._compute_device)

        # Forward pass with hook capture
        Q_stack, K_stack, _, model_output = self._adapter.extract(input_ids, attention_mask)

        # Get attention and hidden states from last semantic layer
        last_layer = self._semantic_layer_indices[-1]
        attn = self._adapter.capture.attention_weights.get(last_layer)
        h_attn = self._adapter.capture.attn_outputs.get(last_layer)
        h_block = self._adapter.capture.block_outputs.get(last_layer)

        # Extract Q/K states for FlashAuthority (v3 optimization)
        q_states = self._adapter.capture.query_states.get(last_layer)
        k_states = self._adapter.capture.key_states.get(last_layer)

        if attn is None:
            raise RuntimeError("Attention weights not captured")

        attn = attn.to(self._compute_device)
        h_attn = h_attn.to(self._compute_device) if h_attn is not None else None
        h_block = h_block.to(self._compute_device) if h_block is not None else None
        q_states = q_states.to(self._compute_device) if q_states is not None else None
        k_states = k_states.to(self._compute_device) if k_states is not None else None
        logits = model_output.logits.to(self._compute_device)
        embed_matrix = self._embed_matrix.to(self._compute_device)

        # Get sequence length for details
        seq_len = input_ids.size(1)

        # Compute authority (version dispatch)
        if self.config.version == 19:
            # v19: Hinge-Risk Architecture - Zero-Shot Cross-Dataset SOTA
            #
            # Master Equation:
            #   R_deception = Mean(JSD)           # Systemic FFN override
            #   R_confusion = Max(V_norm × D)     # Burst epistemic collapse
            #   Risk_seq = Max(R_deception, R_confusion)
            #
            # Key Innovation: Topology-Aware Aggregation
            # - Deception (RAGTruth): Systemic pattern → Mean catches distributed override
            # - Confusion (HaluEval): Burst pattern → Max catches single-token collapse
            #
            # V×D Conjunction: Only penalizes when BOTH signals are high
            #   * High V alone: Protected (valid reasoning)
            #   * High D alone: Protected (rare words)
            #   * High V AND High D: Penalized (confused generation)
            #
            # Cross-dataset SOTA (Avg AUROC = 0.68):
            # - HaluEval QA: 0.84   (V×D catches confusion)
            # - HaluEval Summ: 0.61 (V×D catches sustained confusion)
            # - RAGTruth QA: 0.72   (JSD catches deception)
            # - RAGTruth Summ: 0.56 (JSD catches systemic override)
            if h_attn is None or h_block is None:
                raise RuntimeError("v19 requires h_attn and h_block.")

            # Import for signal computation
            from .measures.entropy import compute_varentropy
            from .measures.semantics import compute_semantic_dispersion
            from .ops import compute_logit_divergence_jsd

            B, S_total, V = logits.shape
            S_resp = S_total - response_start

            if S_resp <= 0:
                uncertainty = torch.zeros(B, device=logits.device)
                if return_details:
                    return {"uncertainty": 0.0, "R_deception": 0.0, "R_confusion": 0.0, "response_tokens": 0}
                return 0.0

            # Slice to response
            h_attn_resp = h_attn[:, response_start:, :]
            h_block_resp = h_block[:, response_start:, :]
            logits_resp = logits[:, response_start:, :].contiguous()

            # === SIGNAL 1: DECEPTION (JSD) ===
            # Detects FFN override - "confident lies"
            jsd = compute_logit_divergence_jsd(
                h_attn_resp, h_block_resp, embed_matrix, top_k=self.config.jsd_top_k
            )  # [B, S_resp] in [0, 1]

            # === SIGNAL 2: CONFUSION (V × D) ===
            # Detects epistemic collapse with semantic confirmation
            # Only fires when BOTH V and D are high (conjunction)
            varentropy = compute_varentropy(logits_resp, attention_mask=None)
            V_norm = torch.tanh(varentropy / self.config.varentropy_scale)  # [B, S_resp] in [0, 1]

            dispersion = compute_semantic_dispersion(
                logits_resp, embed_matrix, k=5, method="nucleus_variance", top_p=0.95
            )  # [B, S_resp] in [0, 1]

            confusion_risk = V_norm * dispersion  # [B, S_resp] - conjunction

            # === TOPOLOGY-AWARE AGGREGATION ===
            # Deception (systemic) → Mean: catches distributed FFN override
            # Confusion (burst) → Max: catches single-token epistemic collapse
            R_deception = jsd.mean(dim=1)  # [B]
            R_confusion = confusion_risk.max(dim=1).values  # [B]

            # === FUSION: Max of detectors ===
            # If EITHER detector fires, flag the sequence
            uncertainty = torch.max(R_deception, R_confusion)  # [B]

            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "R_deception": R_deception[0].item() if R_deception.numel() == 1 else R_deception,
                    "R_confusion": R_confusion[0].item() if R_confusion.numel() == 1 else R_confusion,
                    "response_tokens": S_resp,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 17:
            # v17: Thermodynamic Gating (Adaptive Regime Selection)
            # Uses sequence-level varentropy to determine which detector to trust:
            # - High V (>1.75): Confusion Regime → Trust R_confusion (V×(1-A))
            # - Low V (<1.75): Deception Regime → Trust R_deception (JSD)
            #
            # This is "Maxwell's Demon" - using entropy to sort signals into correct buckets.
            # Replaces the failed "Length Heuristic" (v13) with "State-of-Matter Heuristic".
            if h_attn is None or h_block is None:
                raise RuntimeError("v17 requires h_attn and h_block.")

            # Reuse v16's signal computation
            signals = compute_semantic_authority_v16(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                varentropy_scale=self.config.varentropy_scale,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )

            risk_deception = signals['risk_deception']    # [B, S_resp]
            risk_confusion = signals['risk_confusion']    # [B, S_resp]
            varentropy = signals['varentropy']            # [B, S_resp]
            resp_len = risk_deception.shape[1]

            # === 1. DETERMINE REGIME (Thermodynamics) ===
            # HaluEval: V_mean ~3.0 (High Energy → Confusion)
            # RAGTruth: V_mean ~1.0 (Low Energy → Deception)
            # Threshold 1.75 separates the regimes cleanly
            V_seq_mean = varentropy.mean(dim=1)  # [B]

            # Gate: 0.0 = Deception (Trust JSD), 1.0 = Confusion (Trust V×(1-A))
            # Sigmoid centered at varentropy_threshold, slope varentropy_slope
            gate = torch.sigmoid((V_seq_mean - self.config.varentropy_threshold) * self.config.varentropy_slope)  # [B]

            # === 2. COMPUTE COMPONENT SCORES ===
            # R_deception (JSD) → Mean (systemic)
            R_deception = risk_deception.mean(dim=1)  # [B]

            # R_confusion (V×(1-A)) → Percentile_90 (burst)
            k = max(1, int(risk_confusion.shape[1] * 0.10))
            top_k_vals, _ = torch.topk(risk_confusion, k, dim=1)
            R_confusion = top_k_vals.min(dim=1).values  # [B] - 90th percentile

            # === 3. THERMODYNAMIC FUSION ===
            # Smoothly interpolate based on regime
            # High gate → Trust R_confusion, Low gate → Trust R_deception
            final_risk = gate * R_confusion + (1.0 - gate) * R_deception  # [B]

            if return_details:
                return {
                    "uncertainty": final_risk[0].item() if final_risk.numel() == 1 else final_risk,
                    "R_deception": R_deception[0].item() if R_deception.numel() == 1 else R_deception,
                    "R_confusion": R_confusion[0].item() if R_confusion.numel() == 1 else R_confusion,
                    "V_mean": V_seq_mean[0].item() if V_seq_mean.numel() == 1 else V_seq_mean,
                    "gate": gate[0].item() if gate.numel() == 1 else gate,
                    "response_tokens": resp_len,
                }
            return final_risk[0].item() if final_risk.numel() == 1 else final_risk

        elif self.config.version == 16:
            # v16: Grounded-Risk Architecture
            # Risk_seq = max(R_deception, R_confusion)
            # - R_deception = Mean(JSD) - FFN override detection
            # - R_confusion = Percentile_90(V × (1-A)) - Ungrounded uncertainty
            #
            # Key Innovation - Authority Shield (replaces failed Dispersion Shield):
            # - High V is only penalized if ALSO ungrounded (low Authority)
            # - Valid Summary: High V, High A → Low (1-A) → Low Risk
            # - Confused Hall: High V, Low A → High (1-A) → High Risk
            if h_attn is None or h_block is None:
                raise RuntimeError("v16 requires h_attn and h_block.")

            signals = compute_semantic_authority_v16(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                varentropy_scale=self.config.varentropy_scale,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )

            risk_deception = signals['risk_deception']    # [B, S_resp]
            risk_confusion = signals['risk_confusion']    # [B, S_resp]
            resp_len = risk_deception.shape[1]

            # === AGGREGATION ===
            # Deception (Systemic) → Mean - captures FFN override across sequence
            R_deception = risk_deception.mean(dim=1)  # [B]

            # Confusion (Burst) → Percentile_90 - catches ungrounded uncertainty spikes
            k = max(1, int(risk_confusion.shape[1] * 0.10))
            top_k_vals, _ = torch.topk(risk_confusion, k, dim=1)
            R_confusion = top_k_vals.min(dim=1).values  # [B] - 90th percentile

            # === FUSION: MAX (Logical OR) ===
            # If EITHER detector triggers, flag the sequence
            final_risk = torch.maximum(R_deception, R_confusion)  # [B]

            if return_details:
                return {
                    "uncertainty": final_risk[0].item() if final_risk.numel() == 1 else final_risk,
                    "R_deception": R_deception[0].item() if R_deception.numel() == 1 else R_deception,
                    "R_confusion": R_confusion[0].item() if R_confusion.numel() == 1 else R_confusion,
                    "response_tokens": resp_len,
                }
            return final_risk[0].item() if final_risk.numel() == 1 else final_risk

        elif self.config.version == 15:
            # v15: Coherence-Interaction Model
            # Score = (1 - JSD_seq) × (1 - max(V×D))
            # - Mechanistic Integrity: Mean(1 - JSD) catches deception
            # - Epistemic Coherence: 1 - Max(V×D) catches confusion with shielding
            #
            # Key Innovation - Semantic Shielding:
            # - V × Sigmoid(D) only penalizes when BOTH V and D are high
            # - High V with low D is protected (valid reasoning)
            if h_attn is None or h_block is None:
                raise RuntimeError("v15 requires h_attn and h_block.")

            signals = compute_semantic_authority_v15(
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_threshold,
                dispersion_stiffness=self.config.dispersion_stiffness,
                attention_mask=attention_mask,
            )

            jsd = signals['jsd']           # [B, S_resp]
            confusion = signals['confusion']  # [B, S_resp]
            resp_len = jsd.shape[1]

            # === AGGREGATION ===
            # Mechanistic: Mean(1 - JSD) - systemic override detection
            mech_integrity = (1.0 - jsd).mean(dim=1)  # [B]

            # Epistemic: 1 - Percentile_90(confusion) - burst confusion detection
            # Use topk to find 90th percentile (robust to outliers)
            k = max(1, int(confusion.shape[1] * 0.10))
            top_k_vals, _ = torch.topk(confusion, k, dim=1)
            confusion_p90 = top_k_vals.min(dim=1).values  # [B]
            epist_coherence = 1.0 - confusion_p90  # [B]

            # === FUSION: PRODUCT (Logical AND) ===
            # Both signals must be high for the sequence to pass
            score_seq = mech_integrity * epist_coherence  # [B]
            uncertainty = 1.0 - score_seq

            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "mech_integrity": mech_integrity[0].item() if mech_integrity.numel() == 1 else mech_integrity,
                    "epist_coherence": epist_coherence[0].item() if epist_coherence.numel() == 1 else epist_coherence,
                    "response_tokens": resp_len,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 14:
            # v14: Conservative Min-Fusion (The "Safety Veto")
            # A generation is faithful IFF it is Epistemically Stable AND Mechanistically Grounded.
            #
            # Score_seq = min(S_seq, G_seq)
            # - S_seq: Stability score (catches Confusion - high V, high D)
            # - G_seq: Grounding score (catches Deception - FFN override)
            #
            # This works because:
            # - HaluEval (Confusion): Low S, High G → min = Low → CAUGHT
            # - RAGTruth (Deception): High S, Low G → min = Low → CAUGHT
            # - Facts: High S, High G → min = High → PASS
            if h_attn is None or h_block is None:
                raise RuntimeError("v14 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")

            resp_len = seq_len - response_start

            # === COMPUTE S_seq: STABILITY (Confusion Detector) ===
            # Uses v5's Authority+Trust formula with SlidingWindowMin aggregation
            v5_score = compute_semantic_authority_v5(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_threshold,
                calibration=self._calibration,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
            # v5 already does Mean(A) × Min(T) aggregation internally
            S_seq = v5_score[:, 0]  # [B] - pre-aggregated

            # === COMPUTE G_seq: GROUNDING (Deception Detector) ===
            # Uses v8's JSD formula with Mean aggregation (systemic detection)
            v8_score = compute_semantic_authority_v8(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                attention_mask=attention_mask,
            )
            G_seq = v8_score[:, response_start:].mean(dim=1)  # [B]

            # === FUSION: MIN (Logical AND / Safety Veto) ===
            # If EITHER detector flags the sequence, we flag it.
            # No length heuristics - let the signals speak for themselves.
            final_score = torch.minimum(S_seq, G_seq)  # [B]

            # Uncertainty = 1 - Score (higher uncertainty = more likely hallucination)
            uncertainty = 1.0 - final_score

            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "S_seq": S_seq[0].item() if S_seq.numel() == 1 else S_seq,
                    "G_seq": G_seq[0].item() if G_seq.numel() == 1 else G_seq,
                    "response_tokens": resp_len,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 13:
            # v13.1: Adaptive Regime Switching (Hybrid)
            # Uses the BEST detector for each regime:
            # - Short sequences: v5 (Authority + Trust = Mean(A) × Min(T))
            # - Long sequences: v8 (Grounding = 1 - JSD)
            #
            # Score_seq = (1 - w) × S_v5 + w × G_v8
            # w = sigmoid((Length - τ_len) × α)
            if h_attn is None or h_block is None:
                raise RuntimeError("v13 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")

            resp_len = seq_len - response_start

            # === COMPUTE V5 SCORE (Authority + Trust for short sequences) ===
            # v5 uses: Score = Mean(Authority) × Min(Trust)
            # - Authority: grounding in context from attention
            # - Trust: 1 - (P_v × P_d) with threshold-gated dispersion
            v5_score = compute_semantic_authority_v5(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_threshold,
                calibration=self._calibration,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
            # v5 returns pre-aggregated score broadcasted - take first position
            S_v5 = v5_score[:, 0]  # [B] - already Mean(A) × Min(T)

            # === COMPUTE V8 SCORE (JSD Grounding for long sequences) ===
            # v8 uses: Score = 1 - JSD(P_attn || P_final)
            v8_score = compute_semantic_authority_v8(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                attention_mask=attention_mask,
            )
            # v8 returns per-token scores - aggregate with mean for systemic detection
            G_v8 = v8_score[:, response_start:].mean(dim=1)  # [B]

            # === ADAPTIVE FUSION ===
            # Sigmoid switch based on response length
            # τ=30, α=0.2 → transition zone [20, 40]
            tau_len = self.config.regime_threshold  # default 30
            alpha = self.config.regime_slope  # default 0.2
            w = torch.sigmoid(torch.tensor((float(resp_len) - tau_len) * alpha, device=S_v5.device, dtype=S_v5.dtype))

            # Final score: weighted combination (higher = more trustworthy)
            final_score = (1.0 - w) * S_v5 + w * G_v8  # [B]

            # Uncertainty = 1 - Score (higher uncertainty = more likely hallucination)
            uncertainty = 1.0 - final_score

            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "S_v5": S_v5[0].item() if S_v5.numel() == 1 else S_v5,
                    "G_v8": G_v8[0].item() if G_v8.numel() == 1 else G_v8,
                    "regime_weight": w.item(),
                    "response_tokens": resp_len,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 12:
            # v12: Dual-Stream Risk - Parallel confusion and deception detection
            # Risk_seq = max(R_internal, R_external)
            # R_internal = Percentile_90(tanh(V/τ) × Shield(D))  [Confusion - HaluEval]
            # R_external = Mean(JSD)                              [Deception - RAGTruth]
            if h_attn is None or h_block is None:
                raise RuntimeError("v12 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")

            signals = compute_semantic_authority_v12(
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_threshold,
                attention_mask=attention_mask,
            )

            risk_int = signals['risk_internal'][:, response_start:]  # [B, S_resp]
            risk_ext = signals['risk_external'][:, response_start:]  # [B, S_resp]

            # === AGGREGATION A: INTERNAL (Confusion - Burst Detection) ===
            # Percentile 90 via topk: captures bursts, ignores noise
            # topk gets the k highest values, then min of those is the 90th percentile
            k = max(1, int(risk_int.shape[1] * 0.10))  # Top 10% = 90th percentile
            top_k_vals, _ = torch.topk(risk_int, k, dim=1)  # [B, k]
            R_internal = top_k_vals.min(dim=1).values  # [B] - 90th percentile

            # === AGGREGATION B: EXTERNAL (Deception - Systemic Detection) ===
            # Mean: captures systemic FFN override
            R_external = risk_ext.mean(dim=1)  # [B]

            # === FUSION: MAX RISK ===
            # If either detector flags, we flag (OR logic)
            uncertainty = torch.maximum(R_internal, R_external)  # [B]

            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "R_internal": R_internal[0].item() if R_internal.numel() == 1 else R_internal,
                    "R_external": R_external[0].item() if R_external.numel() == 1 else R_external,
                    "response_tokens": seq_len - response_start,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 11:
            # v11: Information Physics - Unified Energy-Based Detection
            # E_t = JSD(P_attn || P_final) + tanh(V_t)
            # Uncertainty = SoftMax_β(E_t)  [LogSumExp aggregation]
            if h_attn is None or h_block is None:
                raise RuntimeError("v11 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")

            # Compute token-level energy (higher = more likely hallucination)
            energy = compute_semantic_authority_v11(
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_threshold,
                attention_mask=attention_mask,
            )

            # === SoftMax (LogSumExp) Aggregation ===
            # score = (1/β) × log(mean(exp(β × E_t)))
            # Smoothly interpolates between mean (β→0) and max (β→∞)
            energy_resp = energy[:, response_start:]  # [B, S_resp]
            beta = self.config.softmax_beta

            # Stable LogSumExp computation
            # LSE(x) = max(x) + log(mean(exp(x - max(x))))
            scaled = beta * energy_resp  # [B, S_resp]
            max_scaled = scaled.max(dim=1, keepdim=True).values  # [B, 1]
            lse = max_scaled.squeeze(1) + torch.log(torch.mean(torch.exp(scaled - max_scaled), dim=1))  # [B]
            aggregated_energy = lse / beta  # [B], back to original scale

            # Energy is [0, 2], normalize to [0, 1] for uncertainty
            # Divide by 2 to map [0, 2] → [0, 1]
            uncertainty = (aggregated_energy / 2.0).clamp(0.0, 1.0)  # [B]

            # For single-sample return (BS=1 common case)
            if return_details:
                # Also compute component energies for diagnostics
                energy_resp_np = energy_resp[0].detach().cpu().numpy() if energy_resp.numel() > 0 else []
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "aggregated_energy": aggregated_energy[0].item() if aggregated_energy.numel() == 1 else aggregated_energy,
                    "mean_energy": energy_resp.mean().item(),
                    "max_energy": energy_resp.max().item(),
                    "response_tokens": seq_len - response_start,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 10:
            # v10: Orthogonal Signal Fusion - Decoupled aggregation topologies
            # Score_seq = S_seq × G_seq
            # S_seq = SlidingWindowPercentile_10(Stability_t)  [HaluEval burst detection]
            # G_seq = Mean(Grounding_t)                        [RAGTruth systemic detection]
            if h_attn is None or h_block is None:
                raise RuntimeError("v10 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")

            components = compute_semantic_authority_v10(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                varentropy_scale=self.config.varentropy_scale,
                jsd_top_k=self.config.jsd_top_k,
                dispersion_threshold=self.config.dispersion_scale,
                attention_mask=attention_mask,
                calibration=self._calibration,
            )

            # === v10: Orthogonal Signal Fusion with MAX uncertainty ===
            # Two independent detection channels:
            # 1. Stability (semantic shielding) → catches HaluEval confusion (HIGH varentropy)
            # 2. Grounding (pure JSD) → catches RAGTruth deception (FFN override)
            # MAX fusion: whoever is more confident about detecting hallucination wins
            stability_resp = components['stability'][:, response_start:]
            grounding_resp = components['grounding'][:, response_start:]

            # S_seq: SlidingWindowPercentile_10 for stability (catches phrase-level bursts)
            window_size = 5
            if stability_resp.shape[1] >= window_size:
                windows = stability_resp.unfold(1, window_size, 1)  # [B, num_win, W]
                window_scores = windows.mean(dim=2)  # [B, num_win]
                k = max(1, int(window_scores.shape[1] * 0.10))
                bottom_k, _ = torch.topk(window_scores, k, dim=1, largest=False)
                S_seq = bottom_k.max(dim=1).values  # [B]
            else:
                S_seq = stability_resp.min(dim=1).values  # [B]

            # G_seq: Mean for grounding (captures systemic FFN override)
            G_seq = grounding_resp.mean(dim=1)  # [B]

            # MIN of scores: Both signals must pass (AND logic for trust)
            # If EITHER signal fails → low score → high uncertainty
            # HaluEval halls: S_seq LOW, G_seq HIGH → min = LOW → unc HIGH ✓
            # RAGTruth halls: S_seq HIGH, G_seq LOW → min = LOW → unc HIGH ✓
            v10_score = torch.minimum(S_seq, G_seq)  # [B]
            uncertainty = 1.0 - v10_score  # [B]
            unc_stability = 1.0 - S_seq
            unc_grounding = 1.0 - G_seq

            # For single-sample return (BS=1 common case)
            if return_details:
                return {
                    "uncertainty": uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty,
                    "S_seq": S_seq[0].item() if S_seq.numel() == 1 else S_seq,
                    "G_seq": G_seq[0].item() if G_seq.numel() == 1 else G_seq,
                    "unc_stability": unc_stability[0].item() if unc_stability.numel() == 1 else unc_stability,
                    "unc_grounding": unc_grounding[0].item() if unc_grounding.numel() == 1 else unc_grounding,
                    "response_tokens": seq_len - response_start,
                }
            return uncertainty[0].item() if uncertainty.numel() == 1 else uncertainty

        elif self.config.version == 9:
            # v9: Holographic Dual-Stream - Sequence-level MAX uncertainty
            # uncertainty = max(1 - agg(stability), 1 - agg(grounding))
            # Stability = (1 - V_norm) × (1 - D_t) [catches HaluEval confusion]
            # Grounding = 1 - JSD(P_attn || P_final) [catches RAGTruth deception]
            if h_attn is None or h_block is None:
                raise RuntimeError("v9 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")
            components = compute_semantic_authority_v9(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                varentropy_scale=self.config.varentropy_scale,
                jsd_top_k=self.config.jsd_top_k,
                attention_mask=attention_mask,
                return_components=True,
            )
            # Aggregate each component separately using mean
            stability_resp = components['stability'][:, response_start:]
            grounding_resp = components['grounding'][:, response_start:]
            stability_score = stability_resp.mean().item()
            grounding_score = grounding_resp.mean().item()

            # Threshold-gated combination:
            # - Use stability by default (works for HaluEval confusion)
            # - Switch to grounding when JSD is high (grounding < threshold)
            # This prevents grounding from adding false uncertainty to HaluEval facts
            grounding_threshold = 0.55
            if grounding_score < grounding_threshold:
                # High JSD detected - use grounding (RAGTruth deception)
                v9_score = grounding_score
            else:
                # Normal case - use stability (HaluEval confusion)
                v9_score = stability_score

            # Return early for v9 (skip normal aggregation)
            uncertainty = 1.0 - v9_score
            if return_details:
                return {
                    "uncertainty": uncertainty,
                    "stability_score": stability_score,
                    "grounding_score": grounding_score,
                    "response_tokens": seq_len - response_start,
                }
            return uncertainty
        elif self.config.version == 8:
            # v8: Residual Stream Contrast - FFN interference detection
            # Score(t) = (1 - FFN_Interference(t)) × (1 - D_t)
            # FFN_Interference = JSD(P_attn || P_final)
            # Detects "Confident Lies" where FFN overrides context signal
            if h_attn is None or h_block is None:
                raise RuntimeError("v8 requires h_attn and h_block. Ensure ModelAdapter captures attn_outputs and block_outputs.")
            authority = compute_semantic_authority_v8(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                h_attn=h_attn,
                h_block=h_block,
                jsd_top_k=self.config.jsd_top_k,
                attention_mask=attention_mask,
            )
        elif self.config.version == 7:
            # v7: Geometric Manifold Adherence - LID-based detection
            # Score(t) = A(t) × (1 - D_t) × M(t)
            # M(t) = 1 - LID_norm(t) (manifold adherence)
            # Fixes RAGTruth: "Confident lies" have low V but HIGH LID
            if h_block is None:
                raise RuntimeError("v7 requires hidden states. Ensure ModelAdapter is capturing block_outputs.")
            authority = compute_semantic_authority_v7(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                hidden_states=h_block,
                lid_window_size=self.config.lid_window_size,
                calibration=self._calibration,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
        elif self.config.version == 6:
            # v6: Gaussian Complexity Matching - Bidirectional penalty
            # Score_final = Mean(A_t) × Min(T_t)
            # T_t = (1 - D_t) × G_complexity(R_t, σ)
            # G_complexity = exp(-(R_t - 1.0)² / 2σ²)
            # Penalizes both overcomplexity (HaluEval) and oversimplification (RAGTruth)
            authority = compute_semantic_authority_v6(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                complexity_sigma=self.config.complexity_sigma,
                complexity_epsilon=self.config.complexity_epsilon,
                complexity_center=self.config.complexity_center,
                calibration=self._calibration,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
        elif self.config.version == 5:
            # v5: Dual-Path Aggregation - Heterogeneous aggregation
            # Score_final = Mean(A_t) × Min(T_t)
            # - Mean(A_t): Global Grounding (tolerates preambles)
            # - Min(T_t): Local Trust (catches any fabrication)
            # Note: v5 performs aggregation internally and broadcasts the result
            authority = compute_semantic_authority_v5(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_threshold=self.config.dispersion_scale,
                calibration=self._calibration,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
        elif self.config.version == 4:
            # v4: Epistemic Dominance with Context-Aware Semantic Shielding
            # P_faith(t) = A(t) × (1 - P_v × P_d)
            # Dynamic threshold τ = μ_D(prompt) + σ_D(prompt), clamped [0.05, 0.20]
            # Uses min() aggregation to capture hallucination bursts
            authority = compute_semantic_authority_v4(
                attention_weights=attn,
                prompt_length=response_start,
                logits=logits,
                embed_matrix=embed_matrix,
                varentropy_scale=self.config.varentropy_scale,
                dispersion_scale=self.config.dispersion_scale,
                calibration=self._calibration,  # Pass prompt calibration for dynamic threshold
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
        elif self.config.version == 3:
            # v3.4: Cognitive Load - Calibration-free authority
            # Uses absolute varentropy with universal threshold τ
            calibration = {
                'tau': self.config.tau,  # Cognitive load threshold (default 5.0)
            }

            authority = compute_semantic_authority_v3(
                attention_weights=attn,
                prompt_length=response_start,
                h_attn=h_attn,
                h_block=h_block,
                logits=logits,
                embed_matrix=embed_matrix,
                calibration=calibration,
                lambda_struct=self.config.lambda_struct,
                entropy_floor=self.config.entropy_floor,
                attention_mask=attention_mask,
                query_states=q_states,
                key_states=k_states,
            )
        else:
            # v2: Original implementation
            authority = compute_semantic_authority(
                attention_weights=attn,
                prompt_length=response_start,
                h_attn=h_attn,
                h_block=h_block,
                logits=logits,
                embed_matrix=embed_matrix,
                attention_mask=attention_mask,
                varentropy_lambda=self.config.varentropy_lambda,
                calibration=self._calibration,
            )

        # Aggregate token-level scores to sequence-level
        response_authority = authority[:, response_start:]
        response_logits = logits[:, response_start:, :]

        aggregated_authority = self._aggregate_authority(
            response_authority, response_logits
        )

        uncertainty = 1.0 - aggregated_authority

        if return_details:
            log_probs = torch.log_softmax(logits[:, response_start-1:-1, :], dim=-1)
            response_tokens = input_ids[:, response_start:]
            token_log_probs = log_probs.gather(2, response_tokens.unsqueeze(-1)).squeeze(-1)

            return {
                'score': uncertainty,
                'authority': aggregated_authority,
                'model_confidence': token_log_probs.exp().mean().item(),
                'response_start': response_start,
                'sequence_length': input_ids.size(1),
            }

        return uncertainty

    def detect_context_violation(
        self,
        prompt: str,
        response: str,
        threshold: Optional[float] = None,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect context violations with binary classification.

        Args:
            prompt: Source context.
            response: Generated text to evaluate.
            threshold: Decision boundary (default: config.hallucination_threshold).

        Returns:
            Tuple of (is_violation, confidence, details).
        """
        threshold = threshold or self.config.hallucination_threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        details = self.compute_uncertainty(prompt, response, return_details=True)
        is_violation = details['score'] > threshold

        return is_violation, abs(details['score'] - threshold), details

    def cleanup(self):
        """Release resources and remove hooks."""
        self._adapter.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
