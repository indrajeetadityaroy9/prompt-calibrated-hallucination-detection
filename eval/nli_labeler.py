"""
NLI-based ground truth labeling for hallucination detection.

Uses DeBERTa-v3-Large as a cross-encoder for Natural Language Inference
to determine if generated responses are factually consistent with reference answers.

CRITICAL (VRAM Thrashing):
    Do NOT interleave Llama-3 generation (16GB) and DeBERTa evaluation (1.5GB)
    on the same GPU. Use "Generate-Then-Grade" pipeline:
    1. Generate ALL responses with Llama-3
    2. Unload Llama-3 (del model; torch.cuda.empty_cache())
    3. Load DeBERTa-Large
    4. Grade ALL responses in batch

Example:
    >>> labeler = NLILabeler()
    >>> is_factual, score = labeler.label_sample(
    ...     reference="Paris is the capital of France.",
    ...     generated="The capital of France is Paris."
    ... )
    >>> print(f"Factual: {is_factual}, Score: {score:.3f}")
"""

from typing import List, Tuple, Optional
import torch


class NLILabeler:
    """
    NLI-based factuality labeler using DeBERTa cross-encoder.

    Uses entailment scoring: if reference entails generated, the response
    is considered factually consistent.

    Attributes:
        model: CrossEncoder model for NLI
        device: Compute device
        threshold: Entailment threshold for factuality (default: 0.5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-large",
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Initialize NLI labeler.

        Args:
            model_name: HuggingFace model ID for cross-encoder
            device: Compute device ("cuda" or "cpu")
            threshold: Entailment threshold for factuality
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for NLI labeling. "
                "Install with: pip install sentence-transformers"
            )

        self.device = device
        self.threshold = threshold
        self.model_name = model_name

        print(f"Loading NLI model: {model_name}...")
        self.model = CrossEncoder(model_name, device=device)
        print(f"  NLI model loaded on {device}")

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Score entailment between premise and hypothesis.

        For factuality checking:
        - Premise = reference answer (ground truth)
        - Hypothesis = generated response

        Args:
            premise: The reference/ground truth text
            hypothesis: The generated text to check

        Returns:
            Entailment probability [0, 1]
        """
        # CrossEncoder returns [contradiction, neutral, entailment] logits
        # We want the entailment score
        scores = self.model.predict([(premise, hypothesis)])

        # For NLI models, output is typically [contradiction, neutral, entailment]
        # or just a single entailment score depending on the model
        if isinstance(scores, (list, tuple)) and len(scores) > 0:
            if hasattr(scores[0], '__len__') and len(scores[0]) == 3:
                # [contradiction, neutral, entailment]
                return float(scores[0][2])  # Entailment score
            else:
                return float(scores[0])
        return float(scores)

    def batch_score_entailment(
        self,
        pairs: List[Tuple[str, str]]
    ) -> List[float]:
        """
        Batch scoring for efficiency.

        Use this AFTER unloading Llama to avoid VRAM thrashing.

        Args:
            pairs: List of (premise, hypothesis) tuples

        Returns:
            List of entailment scores
        """
        if not pairs:
            return []

        scores = self.model.predict(pairs)

        # Handle different output formats
        result = []
        for score in scores:
            if hasattr(score, '__len__') and len(score) == 3:
                result.append(float(score[2]))  # Entailment
            else:
                result.append(float(score))

        return result

    def label_sample(
        self,
        reference: str,
        generated: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Label a single sample as factual or hallucinated.

        Args:
            reference: Ground truth / reference answer
            generated: Model-generated response
            threshold: Override default threshold

        Returns:
            (is_factual, entailment_score)
        """
        threshold = threshold or self.threshold
        score = self.score_entailment(reference, generated)
        is_factual = score > threshold
        return is_factual, score

    def batch_label_samples(
        self,
        references: List[str],
        generated: List[str],
        threshold: Optional[float] = None
    ) -> List[Tuple[bool, float]]:
        """
        Label multiple samples in batch.

        Args:
            references: List of ground truth answers
            generated: List of model-generated responses
            threshold: Override default threshold

        Returns:
            List of (is_factual, entailment_score) tuples
        """
        if len(references) != len(generated):
            raise ValueError("references and generated must have same length")

        threshold = threshold or self.threshold
        pairs = list(zip(references, generated))
        scores = self.batch_score_entailment(pairs)

        return [(score > threshold, score) for score in scores]

    def unload(self) -> None:
        """
        Free VRAM for switching to another model (e.g., Llama).

        Call this before loading a large LLM to avoid VRAM thrashing.
        """
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        print("NLI model unloaded, VRAM freed")

    def __repr__(self) -> str:
        return f"NLILabeler(model={self.model_name}, threshold={self.threshold})"


def generate_then_grade_pipeline(
    model,
    tokenizer,
    prompts: List[str],
    references: List[str],
    nli_labeler: Optional[NLILabeler] = None,
    max_new_tokens: int = 128,
    device: str = "cuda"
) -> List[dict]:
    """
    Generate-Then-Grade pipeline to avoid VRAM thrashing.

    This pipeline:
    1. Generates all responses with the LLM
    2. Unloads the LLM
    3. Loads DeBERTa NLI
    4. Grades all responses
    5. Unloads NLI

    Args:
        model: Language model for generation
        tokenizer: Tokenizer
        prompts: List of prompts
        references: List of reference answers for grading
        nli_labeler: Optional pre-loaded NLI labeler
        max_new_tokens: Max tokens to generate
        device: Compute device

    Returns:
        List of dicts with 'prompt', 'response', 'is_factual', 'nli_score'
    """
    print(f"Phase 1: Generating {len(prompts)} responses...")

    # Phase 1: Generate all responses
    responses = []
    model.eval()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        responses.append(response)

    print(f"  Generated {len(responses)} responses")

    # Phase 2: Unload LLM
    print("Phase 2: Unloading LLM...")
    del model
    torch.cuda.empty_cache()

    # Phase 3: Load NLI and grade
    print("Phase 3: Loading NLI and grading...")
    if nli_labeler is None:
        nli_labeler = NLILabeler(device=device)

    labels = nli_labeler.batch_label_samples(references, responses)

    # Phase 4: Compile results
    results = []
    for prompt, response, ref, (is_factual, score) in zip(
        prompts, responses, references, labels
    ):
        results.append({
            'prompt': prompt,
            'response': response,
            'reference': ref,
            'is_factual': is_factual,
            'nli_score': score
        })

    # Phase 5: Unload NLI
    nli_labeler.unload()

    factual_count = sum(1 for r in results if r['is_factual'])
    print(f"Grading complete: {factual_count}/{len(results)} factual")

    return results
