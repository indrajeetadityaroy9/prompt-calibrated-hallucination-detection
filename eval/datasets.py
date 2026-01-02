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
    max_samples: Optional[int] = None,
    offset: int = 50  # Skip first N samples (reserved for calibration)
) -> List[EvalSample]:
    """
    Load TruthfulQA dataset for hallucination detection.

    Args:
        split: Dataset split to load
        max_samples: Maximum number of samples (None = all)
        offset: Skip first N samples (default 50 for calibration hold-out)

    Returns:
        List of EvalSample with prompt, response, and factuality label
    """
    from datasets import load_dataset

    dataset = load_dataset("truthful_qa", "generation", split=split)

    samples = []
    sample_count = 0
    for i, item in enumerate(dataset):
        # Skip calibration samples
        if i < offset:
            continue
        if max_samples and sample_count >= max_samples:
            break
        sample_count += 1

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


def load_triviaqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[EvalSample]:
    """
    Load TriviaQA dataset for hallucination detection sanity check.

    TriviaQA is a reading comprehension dataset with factual Q&A pairs.
    Unlike TruthfulQA (designed to trick models), TriviaQA should yield
    AUROC ~0.75+ since it measures genuine factual knowledge.

    Args:
        split: Dataset split to load ("train" or "validation")
        max_samples: Maximum number of samples (None = all)
        seed: Random seed for deterministic hallucination sample generation

    Returns:
        List of EvalSample with prompt, response, and factuality label
    """
    from datasets import load_dataset
    import random

    # Create seeded RNG for deterministic behavior
    rng = random.Random(seed)

    # Load TriviaQA RC (reading comprehension) task
    dataset = load_dataset("trivia_qa", "rc", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        question = item['question']
        # Get answer aliases (multiple correct spellings)
        answer_dict = item.get('answer', {})
        aliases = answer_dict.get('aliases', [])
        normalized_aliases = answer_dict.get('normalized_aliases', [])
        value = answer_dict.get('value', '')

        # Use primary answer value, fall back to first alias
        correct_answer = value if value else (aliases[0] if aliases else '')
        if not correct_answer:
            continue

        # Add factual sample (correct answer)
        samples.append(EvalSample(
            prompt=f"Question: {question}\nAnswer:",
            response=f" {correct_answer}",
            label=True,
            metadata={
                'source': 'triviaqa',
                'type': 'factual',
                'aliases': aliases,
                'normalized_aliases': normalized_aliases
            }
        ))

        # Generate hallucination samples by using wrong answers from other questions
        # We'll add these in a second pass to avoid data leakage within same item

    # Second pass: create hallucination samples by shuffling answers
    if len(samples) > 1:
        correct_answers = [s.response.strip() for s in samples]
        num_correct = len(correct_answers)

        for i, sample in enumerate(list(samples)):  # Iterate over copy
            # Pick a random wrong answer from another question (deterministic)
            offset = rng.randint(1, num_correct - 1)
            wrong_idx = (i + offset) % num_correct
            wrong_answer = correct_answers[wrong_idx]

            # Avoid accidental matches
            if wrong_answer.lower() != sample.response.strip().lower():
                samples.append(EvalSample(
                    prompt=sample.prompt,
                    response=f" {wrong_answer}",
                    label=False,
                    metadata={
                        'source': 'triviaqa',
                        'type': 'hallucination',
                        'original_correct': sample.response.strip()
                    }
                ))

    return samples


def load_coqa(
    split: str = "validation",
    max_samples: Optional[int] = None,
    seed: int = 42
) -> List[EvalSample]:
    """
    Load CoQA dataset for conversational QA hallucination detection.

    CoQA tests conversational reasoning - the model must attend to context
    (story) rather than just internal knowledge. This is a domain generalization
    test beyond fact retrieval (TriviaQA).

    Args:
        split: Dataset split ("train" or "validation")
        max_samples: Maximum number of samples (None = all)
        seed: Random seed for deterministic hallucination sample generation

    Returns:
        List of EvalSample with prompt, response, and factuality label
    """
    from datasets import load_dataset
    import random

    # Create seeded RNG for deterministic behavior
    rng = random.Random(seed)

    # Load CoQA dataset
    dataset = load_dataset("coqa", split=split)

    samples = []
    all_answers = []  # Collect answers for hallucination generation

    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break

        story = item['story']
        questions = item['questions']
        answers = item['answers']

        # Use first Q&A pair for simplicity (full eval would use all)
        if len(questions) > 0 and len(answers['input_text']) > 0:
            question = questions[0]
            answer = answers['input_text'][0]

            # Truncate story if too long (keep first 500 chars for efficiency)
            story_truncated = story[:500] + "..." if len(story) > 500 else story

            prompt = f"Story: {story_truncated}\n\nQuestion: {question}\nAnswer:"

            # Add factual sample
            samples.append(EvalSample(
                prompt=prompt,
                response=f" {answer}",
                label=True,
                metadata={
                    'source': 'coqa',
                    'type': 'factual',
                    'story_id': i
                }
            ))
            all_answers.append(answer)

    # Second pass: create hallucination samples
    if len(samples) > 1 and len(all_answers) > 1:
        for i, sample in enumerate(list(samples)):
            # Pick a random wrong answer from another story (deterministic)
            offset = rng.randint(1, len(all_answers) - 1)
            wrong_idx = (i + offset) % len(all_answers)
            wrong_answer = all_answers[wrong_idx]

            # Avoid accidental matches
            if wrong_answer.lower() != sample.response.strip().lower():
                samples.append(EvalSample(
                    prompt=sample.prompt,
                    response=f" {wrong_answer}",
                    label=False,
                    metadata={
                        'source': 'coqa',
                        'type': 'hallucination',
                        'original_correct': sample.response.strip()
                    }
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

    def __init__(
        self,
        samples: List[EvalSample],
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int = 42
    ):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[List[EvalSample]]:
        indices = list(range(len(self.samples)))
        if self.shuffle:
            import random
            # Use seeded RNG for deterministic shuffling
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield [self.samples[j] for j in batch_indices]

    def __len__(self) -> int:
        return (len(self.samples) + self.batch_size - 1) // self.batch_size
