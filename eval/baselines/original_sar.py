"""
Original SAR implementation using Perturbation Analysis.

CRITICAL: This is the O(N) bottleneck that AG-SAR eliminates.

Formula: SAR = Σ_i H(t_i) × R_pert(t_i)
Where R_pert(t_i) = ||embed(sentence) - embed(sentence \ t_i)||

For N tokens, this requires N+1 RoBERTa forward passes:
- 1 pass for full sentence embedding
- N passes with each token removed

This is the computational cost AG-SAR avoids via internal graph analysis.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalSAR:
    """
    SAR implementation using Perturbation Analysis (Eq. 2 from paper).

    COMPLEXITY: O(N) RoBERTa forward passes per sentence
    - N = number of response tokens
    - Each perturbation requires a full RoBERTa embedding

    This is the bottleneck that AG-SAR eliminates by using internal
    attention graph structure instead of external semantic models.

    Example:
        >>> sar = OriginalSAR(gpt2_model, gpt2_tokenizer)
        >>> uncertainty = sar.compute_uncertainty("The capital of France is", "Paris")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        semantic_model_name: str = "roberta-large",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize Original SAR.

        Args:
            model: Language model (GPT-2) for entropy computation
            tokenizer: Tokenizer for language model
            semantic_model_name: RoBERTa model for semantic embeddings
            device: Compute device (auto-detect from model if None)
            dtype: Tensor dtype (auto-detect from model if None)
        """
        from transformers import RobertaModel, RobertaTokenizer

        self.model = model
        self.tokenizer = tokenizer

        # Auto-detect device and dtype from model if not specified
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        if dtype is None:
            self.dtype = next(model.parameters()).dtype
        else:
            self.dtype = dtype

        # Load RoBERTa for semantic embeddings
        print(f"Loading {semantic_model_name} for perturbation analysis...")
        self.semantic_model = RobertaModel.from_pretrained(semantic_model_name)
        self.semantic_tokenizer = RobertaTokenizer.from_pretrained(semantic_model_name)

        # Move to same device and dtype as main model
        self.semantic_model = self.semantic_model.to(self.device).to(self.dtype)
        self.semantic_model.eval()

        # Ensure pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _get_token_entropy(
        self,
        prompt: str,
        response: str
    ) -> torch.Tensor:
        """
        Compute per-token entropy from GPT-2 logits.

        Returns:
            Tensor of shape (response_length,) with entropy per token
        """
        # Tokenize
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        full_tokens = prompt_tokens + response_tokens
        response_start = len(prompt_tokens)

        input_ids = torch.tensor([full_tokens], device=self.device)

        # Forward pass
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, return_dict=True)

        logits = output.logits[0]  # (seq, vocab)

        # Get logits for response tokens
        # logits[t] predicts token[t+1]
        response_logits = logits[response_start-1:-1, :]  # (response_len, vocab)

        # Compute entropy
        probs = F.softmax(response_logits.float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy

    @torch.inference_mode()
    def _get_roberta_embedding(self, text: str) -> torch.Tensor:
        """
        Get RoBERTa [CLS] embedding for text.

        Single forward pass of RoBERTa.
        """
        inputs = self.semantic_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.semantic_model(**inputs)

        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

    @torch.inference_mode()
    def compute_uncertainty(
        self,
        prompt: str,
        response: str,
        return_details: bool = False
    ) -> float:
        """
        Compute SAR uncertainty using perturbation analysis.

        COMPLEXITY: O(N) RoBERTa forward passes where N = response length

        Algorithm:
        1. Get token entropy from GPT-2
        2. Get full sentence embedding from RoBERTa
        3. For each token i: remove it, re-embed, measure distance
        4. Normalize distances as relevance weights
        5. Compute weighted sum: SAR = Σ H(t_i) × R(t_i)

        Args:
            prompt: Input prompt
            response: Generated response
            return_details: If True, return intermediate values

        Returns:
            SAR uncertainty score
        """
        # 1. Get token entropy from GPT-2
        token_entropy = self._get_token_entropy(prompt, response)
        response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
        num_tokens = len(response_tokens)

        if num_tokens == 0:
            return 0.0

        # 2. Get full sentence embedding from RoBERTa
        full_text = prompt + response
        full_embed = self._get_roberta_embedding(full_text)

        # 3. PERTURBATION ANALYSIS: For each token, remove and re-embed
        # THIS IS THE O(N) BOTTLENECK!
        relevance_scores = []

        for i in range(num_tokens):
            # Remove token i
            perturbed_tokens = response_tokens[:i] + response_tokens[i+1:]
            perturbed_response = self.tokenizer.decode(perturbed_tokens)
            perturbed_text = prompt + perturbed_response

            # Get perturbed embedding (another RoBERTa forward pass)
            perturbed_embed = self._get_roberta_embedding(perturbed_text)

            # Compute semantic distance (L2 norm)
            distance = torch.norm(full_embed - perturbed_embed).item()
            relevance_scores.append(distance)

        # 4. Normalize relevance scores
        relevance = torch.tensor(relevance_scores, device=self.device)
        relevance = relevance / (relevance.sum() + 1e-10)

        # 5. Compute SAR: weighted sum of entropy by relevance
        # Ensure matching lengths
        if len(token_entropy) != len(relevance):
            min_len = min(len(token_entropy), len(relevance))
            token_entropy = token_entropy[:min_len]
            relevance = relevance[:min_len]

        sar = (token_entropy * relevance).sum().item()

        if return_details:
            return {
                'sar': sar,
                'token_entropy': token_entropy.cpu().tolist(),
                'relevance': relevance.cpu().tolist(),
                'num_roberta_passes': num_tokens + 1
            }

        return sar

    def compute_confidence(self, prompt: str, response: str) -> float:
        """
        Compute confidence score (inverse of uncertainty).

        Returns:
            Confidence in [0, 1] range
        """
        sar = self.compute_uncertainty(prompt, response)

        # Normalize to [0, 1]
        max_sar = 10.0
        confidence = 1.0 - min(sar / max_sar, 1.0)

        return confidence

    def batch_compute_uncertainty(
        self,
        prompts: list,
        responses: list
    ) -> list:
        """
        Compute uncertainty for multiple prompt-response pairs.

        Note: Each sample requires O(N) RoBERTa passes, so this is SLOW.
        """
        return [
            self.compute_uncertainty(p, r)
            for p, r in zip(prompts, responses)
        ]

    def get_complexity_analysis(self, response_length: int) -> dict:
        """
        Return complexity analysis for given response length.

        Args:
            response_length: Number of tokens in response

        Returns:
            Dict with complexity metrics
        """
        return {
            'roberta_forward_passes': response_length + 1,
            'gpt2_forward_passes': 1,
            'complexity': f'O({response_length}) RoBERTa passes',
            'note': 'This is the bottleneck AG-SAR eliminates with O(1) complexity'
        }
