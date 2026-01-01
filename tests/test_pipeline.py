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

    def test_extract_attention_shapes(self, gpt2_model):
        """Test that extracted attention has correct shapes."""
        from ag_sar.attention_extractor import AttentionExtractor

        model, tokenizer = gpt2_model
        device = next(model.parameters()).device

        input_ids = tokenizer.encode("Hello world", return_tensors='pt').to(device)
        seq_len = input_ids.size(1)

        extractor = AttentionExtractor(model)
        extractor.register_hooks()
        try:
            attn_weights, value_norms, output = extractor.extract(input_ids)
        finally:
            extractor.remove_hooks()

        # Should have attention from all layers
        assert len(attn_weights) > 0

        # Check shapes
        for layer_idx, attn in attn_weights.items():
            batch, heads, s1, s2 = attn.shape
            assert batch == 1
            assert heads == model.config.n_head
            assert s1 == seq_len
            assert s2 == seq_len

        # Check value norms
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
        assert 'gse' in result
        assert 'token_entropy' in result
        assert 'relevance' in result
        assert 'centrality' in result
        assert 'response_start' in result
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

    def test_batch_compute_uncertainty(self, gpt2_model):
        """Test batch uncertainty computation."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        prompts = [
            "The capital of France is",
            "2 + 2 equals"
        ]
        responses = ["Paris", "4"]

        scores = ag_sar.batch_compute_uncertainty(prompts, responses)

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        assert all(s >= 0 for s in scores)
        ag_sar.cleanup()

    def test_token_contributions(self, gpt2_model):
        """Test per-token contribution analysis."""
        from ag_sar import AGSAR

        model, tokenizer = gpt2_model
        ag_sar = AGSAR(model, tokenizer)

        result = ag_sar.get_token_contributions(
            prompt="The capital of France is",
            response="Paris"
        )

        assert 'gse' in result
        assert 'tokens' in result
        assert len(result['tokens']) > 0
        assert all('entropy' in t and 'relevance' in t for t in result['tokens'])
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

        # Check for NaN values
        assert not torch.isnan(torch.tensor(result['gse']))
        assert not torch.any(torch.isnan(result['token_entropy']))
        assert not torch.any(torch.isnan(result['relevance']))
        ag_sar.cleanup()


class TestConfigOptions:
    """Test different configuration options."""

    def test_custom_thresholds(self, gpt2_model):
        """Test custom entropy thresholds."""
        from ag_sar import AGSAR, AGSARConfig

        model, tokenizer = gpt2_model

        config = AGSARConfig(
            entropy_threshold_low=0.2,
            entropy_threshold_high=0.9,
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
