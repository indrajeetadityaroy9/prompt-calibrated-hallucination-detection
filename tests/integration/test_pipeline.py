"""Integration tests for the full AG-SAR pipeline."""

import pytest
import torch

# Skip all tests if transformers is not available
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def gpt2_model():
    """Load GPT-2 model for testing."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Use default SDPA - AG-SAR reconstructs attention from Q/K hooks
    # No need for eager attention (which would be slower)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use bfloat16 if available (H100)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    return model, tokenizer


class TestAttentionExtractor:
    """Test attention extraction with real GPT-2 model."""

    def test_extract_qk_shapes(self, gpt2_model):
        """Test that extracted Q/K stacks have correct shapes."""
        from ag_sar.modeling import ModelAdapter

        model, tokenizer = gpt2_model
        device = next(model.parameters()).device

        input_ids = tokenizer.encode("Hello world", return_tensors='pt').to(device)
        seq_len = input_ids.size(1)

        adapter = ModelAdapter(model)
        adapter.register()
        try:
            Q_stack, K_stack, value_norms, output = adapter.extract(input_ids)
        finally:
            adapter.cleanup()

        # Check Q/K stack shapes: (B, L*H, S, D)
        # Default semantic_layers=4, GPT-2 has 12 heads
        num_layers = len(adapter.layers)
        num_heads = model.config.n_head
        head_dim = model.config.n_embd // num_heads

        assert Q_stack.shape[0] == 1, f"Batch mismatch: {Q_stack.shape[0]}"
        assert Q_stack.shape[1] == num_layers * num_heads, f"L*H mismatch: {Q_stack.shape[1]}"
        assert Q_stack.shape[2] == seq_len, f"Seq len mismatch: {Q_stack.shape[2]}"
        assert Q_stack.shape[3] == head_dim, f"Head dim mismatch: {Q_stack.shape[3]}"

        # K_stack should match Q_stack
        assert K_stack.shape == Q_stack.shape

        # Check value norms: dict with (B, H, S) tensors
        assert len(value_norms) > 0
        for layer_idx, norms in value_norms.items():
            batch, heads, seq = norms.shape
            assert batch == 1
            assert heads == model.config.n_head
            assert seq == seq_len


class TestFullPipeline:
    """Test complete AG-SAR pipeline."""

    def test_compute_uncertainty_basic(self, gpt2_model):
        """Test basic uncertainty computation."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        gse = ag_sar.compute_uncertainty(
            prompt="The capital of France is",
            response="Paris"
        )

        assert isinstance(gse, float)
        assert gse >= 0
        ag_sar.cleanup()

    def test_compute_uncertainty_with_details(self, gpt2_model):
        """Test uncertainty computation with details."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        result = ag_sar.compute_uncertainty(
            prompt="What is 2+2?",
            response="4",
            return_details=True
        )

        assert isinstance(result, dict)
        assert 'score' in result
        assert 'metric' in result
        assert 'response_start' in result
        assert 'sequence_length' in result
        ag_sar.cleanup()

    def test_detect_hallucination(self, gpt2_model):
        """Test hallucination detection."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        is_hall, confidence, details = ag_sar.detect_hallucination(
            prompt="The capital of France is",
            response="Paris"
        )

        assert isinstance(is_hall, bool)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
        assert isinstance(details, dict)
        ag_sar.cleanup()

    def test_multiple_calls(self, gpt2_model):
        """Test multiple sequential uncertainty computations."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        # Run multiple computations sequentially
        prompts = [
            "The capital of France is",
            "2 + 2 equals"
        ]
        responses = ["Paris", "4"]

        scores = []
        for prompt, response in zip(prompts, responses):
            score = ag_sar.compute_uncertainty(prompt, response)
            scores.append(score)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0 for s in scores)
        ag_sar.cleanup()


class TestDtypeHandling:
    """Test dtype handling for H100 optimization."""

    def test_bfloat16_no_nan(self, gpt2_model):
        """Test that bfloat16 doesn't produce NaN values."""
        from ag_sar import AGSAR, AGSARConfig

        model, tokenizer = gpt2_model

        # Skip if not on GPU with bfloat16 support
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            pytest.skip("bfloat16 not supported")

        config = AGSARConfig(preferred_dtype=torch.bfloat16)
        ag_sar = AGSAR(model, tokenizer, config=config)

        result = ag_sar.compute_uncertainty(
            prompt="Test prompt for",
            response="numerical stability",
            return_details=True
        )

        # Check for NaN values in returned score
        assert not torch.isnan(torch.tensor(result['score'])), "Score is NaN"
        assert isinstance(result['score'], float), "Score should be a float"
        ag_sar.cleanup()


class TestConfigOptions:
    """Test different configuration options."""

    def test_custom_thresholds(self, gpt2_model):
        """Test custom hallucination threshold."""
        from ag_sar import AGSAR, AGSARConfig

        model, tokenizer = gpt2_model

        config = AGSARConfig(
            hallucination_threshold=0.5
        )
        ag_sar = AGSAR(model, tokenizer, config=config)

        is_hall, conf, _ = ag_sar.detect_hallucination(
            prompt="Test",
            response="response"
        )

        assert isinstance(is_hall, bool)
        ag_sar.cleanup()

    def test_custom_semantic_layers(self, gpt2_model):
        """Test custom number of semantic layers."""
        from ag_sar import AGSAR, AGSARConfig

        model, tokenizer = gpt2_model

        config = AGSARConfig(semantic_layers=2)
        ag_sar = AGSAR(model, tokenizer, config=config)

        gse = ag_sar.compute_uncertainty(
            prompt="Test",
            response="response"
        )

        assert isinstance(gse, float)
        ag_sar.cleanup()
