"""
Integration test fixtures.

Inherits global fixtures from tests/conftest.py.
Provides model fixtures for end-to-end testing.
"""

import pytest
import torch


@pytest.fixture(scope="module")
def gpt2_model():
    """
    Load GPT-2 model for integration tests.

    Uses module scope to avoid reloading for each test.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,  # Use float32 for CPU tests
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        pytest.skip(f"Could not load GPT-2: {e}")
