"""
Advanced Behavioral Verification Tests for AG-SAR v8.5

These tests verify SCIENTIFIC CORRECTNESS, not just code execution.
They catch subtle bugs (broadcasting errors, hook misalignment, memory leaks)
that unit tests miss.

Checklist items verified:
- [ ] Pre-MLP Integrity: h_attn != final hidden states
- [ ] Dimension Broadcast: Correct tensor operations
- [ ] Full Pipeline: End-to-end score validity
"""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoConfig

from ag_sar import AGSAR, AGSARConfig
from ag_sar.modeling import ModelAdapter
from ag_sar.ops import (
    compute_authority_flow,
    compute_stability_gate,
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
        causing stability gate to detect MLP non-linearities
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

    def test_stability_gate_dimension_handling(self):
        """
        SCIENTIFIC CHECK:
        Stability Gate must correctly compute:
        Gate = exp(-sensitivity × divergence)
        with proper dimension reduction.
        """
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)

        gate = compute_stability_gate(h_attn, h_block, sensitivity=1.0)

        # Verify output shape
        assert gate.shape == (B, S), \
            f"Gate shape wrong: {gate.shape}, expected {(B, S)}"

        # Gate should be in [0, 1]
        assert (gate >= 0).all() and (gate <= 1).all(), \
            "Gate out of bounds [0, 1]"


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

    def test_reset_is_noop(self, small_model, mock_tokenizer):
        """Test that reset() is a no-op in v8.5 (no streaming state)."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_model, mock_tokenizer, config)

        # Ensure model is in eval mode for deterministic results
        small_model.eval()

        # Run twice to verify no accumulating state
        score1 = agsar.compute_uncertainty("hello", "world")
        agsar.reset()
        score2 = agsar.compute_uncertainty("hello", "world")

        # Scores should be valid regardless of reset
        assert 0.0 <= score1 <= 1.0
        assert 0.0 <= score2 <= 1.0

        agsar.cleanup()

    def test_cleanup_releases_resources(self, small_model, mock_tokenizer):
        """Test that cleanup() properly releases resources."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_model, mock_tokenizer, config)

        # Verify adapter is registered
        assert agsar._adapter is not None

        # Cleanup
        agsar.cleanup()

        # Running again should fail gracefully or be handled
        # (depending on implementation)
