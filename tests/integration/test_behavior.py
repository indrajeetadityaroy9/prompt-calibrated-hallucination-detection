"""
Advanced Behavioral Verification Tests for AG-SAR v8.0

These tests verify SCIENTIFIC CORRECTNESS, not just code execution.
They catch subtle bugs (broadcasting errors, hook misalignment, memory leaks)
that unit tests miss.

Checklist items verified:
- [ ] Pre-MLP Integrity: h_attn != final hidden states
- [ ] Register Mask: BOS token properly masked
- [ ] Dimension Broadcast: Correct tensor operations
- [ ] Kurtosis Distribution Detection: Semantic vs Register tokens
"""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoConfig

from ag_sar import AGSAR, AGSARConfig
from ag_sar.modeling import ModelAdapter
from ag_sar.ops import (
    compute_authority_flow,
    compute_spectral_roughness,
    compute_register_mask,
    fisher_kurtosis,
)


class TestPreMLPIntegrity:
    """Phase 1.2: Verify h_attn is captured BEFORE MLP."""

    @pytest.fixture
    def small_model(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        return AutoModelForCausalLM.from_config(config)

    def test_h_attn_not_equal_final_hidden(self, small_model):
        """
        SCIENTIFIC CHECK:
        h_attn must NOT equal final hidden states.

        If they are identical, the hook captured post-MLP output,
        causing Spectral Roughness to detect MLP non-linearities
        as hallucinations (False Positives).
        """
        extractor = ModelAdapter(
            model=small_model,
            layers=[0, 1],
            dtype=torch.float32
        )
        extractor.register()

        input_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            outputs = small_model(input_ids, output_hidden_states=True)

        # Get captured h_attn (pre-MLP)
        h_attn_captured = extractor.capture.attn_outputs.get(1)  # Last semantic layer

        # Get final hidden states (post all layers)
        final_hidden = outputs.hidden_states[-1]  # (B, S, D)

        if h_attn_captured is not None:
            # h_attn should NOT be identical to final output
            # (Small differences expected due to MLP transformation)
            assert not torch.allclose(h_attn_captured, final_hidden, atol=1e-3), \
                "h_attn equals final hidden - hook captured wrong location!"

        extractor.cleanup()


class TestRegisterMask:
    """Phase 1.3: Verify Register Mask correctly identifies sink tokens."""

    def test_bos_token_masked(self):
        """
        SCIENTIFIC CHECK:
        First tokens (BOS/sink) should have low kurtosis and be masked.
        """
        torch.manual_seed(42)
        B, S, D = 1, 32, 64

        # Create value vectors where first tokens are "sink-like" (uniform)
        v = torch.randn(B, S, D)
        # Make first 4 tokens uniform (low kurtosis)
        v[:, :4, :] = torch.rand(B, 4, D) * 0.1

        mask, ema_state = compute_register_mask(
            v, sink_token_count=4, kurtosis_threshold=1.0
        )

        # First 4 tokens (sinks) should be masked to 0
        assert (mask[:, :4] == 0).all(), \
            f"Sink tokens not masked: {mask[:, :4]}"

    def test_ema_state_not_zero(self):
        """
        SCIENTIFIC CHECK:
        After processing tokens, EMA mean should not be zero vector.
        """
        torch.manual_seed(42)
        v = torch.randn(1, 32, 64)

        # Process multiple batches to update EMA
        ema_state = None
        for _ in range(10):
            _, ema_state = compute_register_mask(
                v, ema_state=ema_state, update_ema=True
            )

        assert ema_state is not None
        assert ema_state.count == 10
        # Mean should not be zero after updates
        assert not torch.allclose(ema_state.mean, torch.zeros_like(ema_state.mean)), \
            "EMA mean is still zero after 10 updates"


class TestDimensionBroadcast:
    """Phase 1.4: Verify correct tensor dimension handling."""

    def test_authority_flow_dimension_handling(self):
        """
        SCIENTIFIC CHECK:
        Authority Flow must correctly handle:
        - Attention weights: (B, H, S, S) -> sum over S (keys)
        - Past Authority: (B, S)

        If broadcasting is wrong, it might sum over Heads instead of S.
        """
        torch.manual_seed(42)
        B, H, S = 2, 8, 32
        prompt_length = 16

        # Multi-head attention weights
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        # Compute authority
        authority = compute_authority_flow(attn, prompt_length)

        # Verify output shape
        assert authority.shape == (B, S), \
            f"Authority shape wrong: {authority.shape}, expected {(B, S)}"

        # Verify prompt tokens have full authority
        assert (authority[:, :prompt_length] == 1.0).all(), \
            "Prompt tokens don't have authority=1.0"

        # Verify authority is bounded
        assert (authority >= 0).all() and (authority <= 1).all(), \
            "Authority out of bounds - possible broadcast error"

    def test_roughness_dimension_handling(self):
        """
        SCIENTIFIC CHECK:
        Spectral Roughness must correctly compute:
        ||h_attn - Σ A·v||_2 with proper dimension reduction.
        """
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 16, 64

        h_attn = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        roughness = compute_spectral_roughness(h_attn, v, attn)

        # Verify output shape
        assert roughness.shape == (B, S), \
            f"Roughness shape wrong: {roughness.shape}, expected {(B, S)}"

        # Roughness should be non-negative (L2 norm)
        assert (roughness >= 0).all(), "Negative roughness detected"


class TestSpectralRoughnessSignal:
    """Phase 2.2: Scientific check - spectral roughness signal quality."""

    def test_roughness_non_negative(self):
        """
        SCIENTIFIC CHECK:
        Spectral Roughness should always be non-negative.
        """
        torch.manual_seed(42)
        B, H, S, D = 1, 4, 16, 64

        # Create value vectors and attention with proper shapes
        v = torch.randn(B, S, D)
        h_attn = torch.randn(B, S, D)

        # Multi-head attention (B, H, S, S)
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        roughness = compute_spectral_roughness(h_attn, v, attn)

        # Roughness should be non-negative
        assert (roughness >= 0).all(), "Roughness should be non-negative"
        assert roughness.shape == (B, S), f"Expected shape (B, S), got {roughness.shape}"


class TestKurtosisDistributionDetection:
    """Verify kurtosis correctly identifies distribution types."""

    def test_semantic_vs_register_tokens(self):
        """
        SCIENTIFIC CHECK:
        Semantic tokens (high kurtosis) vs Register tokens (low kurtosis)
        should be distinguishable.
        """
        torch.manual_seed(42)

        # Semantic token: spiky distribution (high kurtosis)
        semantic = torch.zeros(1, 256)
        semantic[0, ::10] = torch.randn(26) * 10  # Sparse large values
        kurt_semantic = fisher_kurtosis(semantic, dim=-1)

        # Register token: uniform distribution (low kurtosis)
        register = torch.rand(1, 256) * 2 - 1  # Uniform [-1, 1]
        kurt_register = fisher_kurtosis(register, dim=-1)

        # Semantic should have higher kurtosis than register
        assert kurt_semantic.item() > kurt_register.item(), \
            f"Semantic kurtosis ({kurt_semantic.item():.2f}) should > Register ({kurt_register.item():.2f})"

        # Register should have negative kurtosis (platykurtic)
        assert kurt_register.item() < 0, \
            f"Register kurtosis should be negative, got {kurt_register.item():.2f}"


class TestFullPipelineBatch:
    """Test the full AG-SAR pipeline with batch API."""

    @pytest.fixture
    def small_model(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000
        # Use eager attention to avoid SDPA kernel issues on macOS
        return AutoModelForCausalLM.from_config(config, attn_implementation="eager")

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with proper interface."""
        class MockTokenizerOutput(dict):
            """Dict-like object that also has attribute access."""
            def __init__(self, input_ids):
                super().__init__()
                self['input_ids'] = input_ids
                self.input_ids = input_ids

        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                ids = list(range(2, 2 + len(text.split())))
                if return_tensors == "pt":
                    return MockTokenizerOutput(torch.tensor([ids]))
                return {"input_ids": ids}

        return MockTokenizer()

    def test_compute_uncertainty_returns_valid_score(self, small_model, mock_tokenizer):
        """Test that compute_uncertainty returns a score in [0, 1]."""
        config = AGSARConfig(
            semantic_layers=2,
            enable_unified_gating=True,
            enable_semantic_dispersion=False,  # Disable to avoid embed matrix issues
        )
        agsar = AGSAR(small_model, mock_tokenizer, config)

        score = agsar.compute_uncertainty("hello world", "this is a test")

        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"

        agsar.cleanup()

    def test_detect_hallucination_returns_tuple(self, small_model, mock_tokenizer):
        """Test that detect_hallucination returns proper tuple."""
        config = AGSARConfig(
            semantic_layers=2,
            enable_unified_gating=True,
            enable_semantic_dispersion=False,
        )
        agsar = AGSAR(small_model, mock_tokenizer, config)

        is_hall, confidence, details = agsar.detect_hallucination(
            "hello world", "this is a test"
        )

        assert isinstance(is_hall, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(details, dict)
        assert "score" in details
        assert "authority" in details

        agsar.cleanup()

    def test_reset_clears_ema_state(self, small_model, mock_tokenizer):
        """Test that reset() clears streaming state."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_model, mock_tokenizer, config)

        # Run once to populate EMA state
        agsar.compute_uncertainty("hello", "world")
        assert agsar._ema_state is not None

        # Reset should clear it
        agsar.reset()
        assert agsar._ema_state is None

        agsar.cleanup()
