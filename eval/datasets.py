"""Dataset loaders for evaluation."""

from typing import Iterator, Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import torch


@dataclass
class EvalSample:
    """Single evaluation sample."""
    prompt: str
    response: str
    label: Optional[bool] = None  # True = factual, False = hallucination
    metadata: Optional[Dict[str, Any]] = None


def load_truthfulqa(
    split: str = "validation",
    max_samples: Optional[int] = None
) -> List[EvalSample]:
    """
    Load TruthfulQA dataset for hallucination detection.

    Args:
        split: Dataset split to load
        max_samples: Maximum number of samples (None = all)

    Returns:
        List of EvalSample with prompt, response, and factuality label
    """
    from datasets import load_dataset

    dataset = load_dataset("truthful_qa", "generation", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        # Use best answer as factual response
        prompt = item['question']
        best_answer = item['best_answer']
        incorrect_answers = item.get('incorrect_answers', [])

        # Add factual sample
        samples.append(EvalSample(
            prompt=prompt,
            response=best_answer,
            label=True,
            metadata={'source': 'truthfulqa', 'type': 'factual'}
        ))

        # Add hallucination samples (incorrect answers)
        for inc_ans in incorrect_answers[:2]:  # Limit to 2 per question
            samples.append(EvalSample(
                prompt=prompt,
                response=inc_ans,
                label=False,
                metadata={'source': 'truthfulqa', 'type': 'hallucination'}
            ))

    return samples


def load_wikitext(
    name: str = "wikitext-103-v1",
    split: str = "test",
    max_samples: Optional[int] = None,
    min_length: int = 50,
    max_length: int = 200
) -> List[EvalSample]:
    """
    Load WikiText dataset for mechanistic analysis.

    Args:
        name: WikiText variant
        split: Dataset split
        max_samples: Maximum number of samples
        min_length: Minimum text length (chars)
        max_length: Maximum text length (chars)

    Returns:
        List of EvalSample with text split into prompt/response
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", name, split=split)

    samples = []
    for item in dataset:
        text = item['text'].strip()

        # Skip empty or too short
        if len(text) < min_length:
            continue

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]

        # Split into prompt (first half) and response (second half)
        mid = len(text) // 2
        prompt = text[:mid]
        response = text[mid:]

        samples.append(EvalSample(
            prompt=prompt,
            response=response,
            label=None,  # No label for WikiText
            metadata={'source': 'wikitext', 'length': len(text)}
        ))

        if max_samples and len(samples) >= max_samples:
            break

    return samples


def generate_synthetic_samples(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.7
) -> List[EvalSample]:
    """
    Generate responses from model for evaluation.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompts
        max_new_tokens: Maximum generation length
        temperature: Sampling temperature

    Returns:
        List of EvalSample with generated responses
    """
    samples = []
    device = next(model.parameters()).device

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)

        samples.append(EvalSample(
            prompt=prompt,
            response=response,
            label=None,
            metadata={'source': 'generated', 'temperature': temperature}
        ))

    return samples


class DataLoader:
    """Batch iterator for evaluation samples."""

    def __init__(self, samples: List[EvalSample], batch_size: int = 1, shuffle: bool = False):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[EvalSample]]:
        indices = list(range(len(self.samples)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self.samples[j] for j in batch_indices]

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
