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
from .measures import compute_semantic_authority, compute_varentropy
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

        # Gate statistics (full prompt)
        divergence = compute_mlp_divergence(h_attn, h_block)
        gate_values = fused_stability_gate(h_attn, h_block, sensitivity=10.0)
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

        # Compute adaptive thresholds using sigma multiplier
        sigma = self.config.sigma_multiplier
        self._calibration = {
            'gate_mu': gate_mu,
            'gate_sigma': gate_sigma,
            'dispersion_mu': disp_mu,
            'dispersion_sigma': disp_sigma,
            'varentropy_mu': var_mu,
            'varentropy_sigma': var_sigma,
            'adaptive_cpg_gate_threshold': gate_mu + sigma * gate_sigma,
            'adaptive_cpg_dispersion_threshold': disp_mu + sigma * disp_sigma,
            'adaptive_cpg_varentropy_threshold': var_mu + sigma * var_sigma,
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

        if attn is None:
            raise RuntimeError("Attention weights not captured")

        attn = attn.to(self._compute_device)
        h_attn = h_attn.to(self._compute_device) if h_attn is not None else None
        h_block = h_block.to(self._compute_device) if h_block is not None else None
        logits = model_output.logits.to(self._compute_device)
        embed_matrix = self._embed_matrix.to(self._compute_device)

        # Compute authority
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

        # Aggregate: risk-centric 10th percentile weighted by varentropy
        response_authority = authority[:, response_start:]
        response_logits = logits[:, response_start:, :]

        varentropy = compute_varentropy(response_logits)
        probs = torch.softmax(response_logits, dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1)
        v_norm = (varentropy / (entropy + 1e-8)).clamp(0.0, 1.0)

        risk_adjusted = response_authority / (1.0 + v_norm)
        aggregated_authority = torch.quantile(risk_adjusted.flatten(), 0.10).item()

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
