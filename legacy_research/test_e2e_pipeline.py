"""
QA Tests: End-to-End Pipeline Verification for AG-SAR v3.1

Purpose: Verify the recursive Authority Flow works over a real generation sequence.

Checklist items verified:
- [ ] Prompt vs. Gen: Correct handling of prompt processing vs token generation
- [ ] State Evolution: Authority history and EMA stats update correctly
- [ ] Score Bounds: Hallucination scores are in valid range
"""

import torch
import pytest
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoConfig, AutoModelForCausalLM

from ag_sar import AGSAR, AGSARConfig


class TestFullGenerationFlow:
    """Test full generation flow with real model."""

    @pytest.fixture
    def small_gpt2(self):
        """Create a small GPT-2 model for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000

        model = AutoModelForCausalLM.from_config(config)
        return model

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer-like object."""
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def encode(self, text, add_special_tokens=True):
                # Simple mock: return fixed-length tokens
                return list(range(2, 2 + len(text.split())))

            def decode(self, token_ids):
                return " ".join([f"token_{i}" for i in token_ids])

        return MockTokenizer()

    def test_agsar_initialization(self, small_gpt2, mock_tokenizer):
        """Test AGSAR initializes correctly."""
        config = AGSARConfig(
            semantic_layers=2,
            enable_register_filter=True,
            enable_authority_flow=True,
            enable_spectral_roughness=True,
        )

        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Verify initialization
        assert agsar.model is small_gpt2
        assert agsar.config is config
        assert agsar._authority_history is None  # Not initialized until process_prompt
        assert agsar._is_prompt_phase is True

        agsar.cleanup()

    def test_reset_clears_state(self, small_gpt2, mock_tokenizer):
        """Test reset() clears all state."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Set some dummy state
        agsar._authority_history = torch.ones(1, 10)
        agsar._prompt_length = 5
        agsar._is_prompt_phase = False

        # Reset
        agsar.reset()

        # Verify state is cleared
        assert agsar._authority_history is None
        assert agsar._prompt_length == 0
        assert agsar._is_prompt_phase is True
        assert agsar._ema_state is None

        agsar.cleanup()

    def test_process_prompt_initializes_state(self, small_gpt2, mock_tokenizer):
        """Test process_prompt initializes authority and state."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Create prompt input
        prompt_ids = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            result = agsar.process_prompt(prompt_ids)

        # Verify state initialization
        assert agsar._prompt_length == 8
        assert agsar._is_prompt_phase is False
        assert agsar._authority_history is not None
        assert agsar._authority_history.shape == (1, 8)

        # Prompt tokens should have authority = 1.0
        assert (agsar._authority_history == 1.0).all()

        # Result should contain expected keys
        assert 'prompt_length' in result
        assert 'authority' in result
        assert 'logits' in result

        agsar.cleanup()

    def test_process_step_requires_prompt_first(self, small_gpt2, mock_tokenizer):
        """Test process_step raises error if called before process_prompt."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        input_ids = torch.randint(0, 100, (1, 10))

        with pytest.raises(RuntimeError, match="Must call process_prompt"):
            agsar.process_step(input_ids)

        agsar.cleanup()

    def test_process_step_returns_valid_score(self, small_gpt2, mock_tokenizer):
        """Test process_step returns hallucination score in valid range."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Process prompt
        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        # Process one generation step
        full_ids = torch.randint(0, 100, (1, 9))  # prompt + 1 new token

        with torch.no_grad():
            score = agsar.process_step(full_ids)

        # Score should be a float in [0, 1]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

        agsar.cleanup()

    def test_process_step_with_details(self, small_gpt2, mock_tokenizer):
        """Test process_step with return_details=True."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        full_ids = torch.randint(0, 100, (1, 9))

        with torch.no_grad():
            result = agsar.process_step(full_ids, return_details=True)

        # Verify result structure
        assert isinstance(result, dict)
        assert 'hallucination_score' in result
        assert 'authority' in result
        assert 'roughness' in result
        assert 'token_position' in result
        assert 'prompt_length' in result

        # Verify values
        assert 0.0 <= result['hallucination_score'] <= 1.0
        assert result['token_position'] == 8  # Last token index
        assert result['prompt_length'] == 8

        agsar.cleanup()

    def test_multiple_generation_steps(self, small_gpt2, mock_tokenizer):
        """Test multiple sequential generation steps."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Process prompt
        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        # Generate 5 tokens
        scores = []
        current_ids = prompt_ids.clone()

        for i in range(5):
            # Append a new token
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                score = agsar.process_step(current_ids)

            scores.append(score)

            # Verify authority history grows
            assert agsar._authority_history.shape[1] == 8 + i + 1

        # All scores should be valid
        assert len(scores) == 5
        for score in scores:
            assert 0.0 <= score <= 1.0

        agsar.cleanup()

    def test_authority_history_evolution(self, small_gpt2, mock_tokenizer):
        """Test authority history evolves correctly."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        # Prompt tokens should have authority = 1.0
        prompt_authority = agsar._authority_history[:, :8].clone()
        assert (prompt_authority == 1.0).all()

        # Generate a few tokens
        current_ids = prompt_ids.clone()
        for _ in range(3):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                agsar.process_step(current_ids)

        # Prompt authority should still be 1.0
        assert (agsar._authority_history[:, :8] == 1.0).all()

        # Generated tokens should have authority in [0, 1]
        gen_authority = agsar._authority_history[:, 8:]
        assert (gen_authority >= 0).all()
        assert (gen_authority <= 1).all()

        agsar.cleanup()


class TestV31UncertaintyMetric:
    """Test v3.1 uncertainty metric through compute_uncertainty."""

    @pytest.fixture
    def small_gpt2(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000
        return AutoModelForCausalLM.from_config(config)

    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def encode(self, text, add_special_tokens=True):
                return list(range(2, 2 + len(text.split())))

            def decode(self, token_ids):
                return " ".join([f"token_{i}" for i in token_ids])

        return MockTokenizer()

    def test_v31_metric_via_config(self, small_gpt2, mock_tokenizer):
        """Test v3.1 metric is used when configured."""
        config = AGSARConfig(
            semantic_layers=2,
            uncertainty_metric="v31",
        )

        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # compute_uncertainty should use v3.1 pipeline
        with torch.no_grad():
            score = agsar.compute_uncertainty("test prompt", "test response")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        agsar.cleanup()

    def test_authority_metric_via_config(self, small_gpt2, mock_tokenizer):
        """Test authority metric is used when configured."""
        config = AGSARConfig(
            semantic_layers=2,
            uncertainty_metric="authority",
        )

        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        with torch.no_grad():
            score = agsar.compute_uncertainty("test prompt", "test response")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

        agsar.cleanup()

    def test_v31_with_details(self, small_gpt2, mock_tokenizer):
        """Test v3.1 metric returns correct details."""
        config = AGSARConfig(
            semantic_layers=2,
            uncertainty_metric="v31",
        )

        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        with torch.no_grad():
            result = agsar.compute_uncertainty(
                "test prompt",
                "test response",
                return_details=True
            )

        assert isinstance(result, dict)
        assert 'uncertainty' in result
        assert 'metric' in result
        assert result['metric'] == 'v3.1_authority'
        assert 'authority' in result
        assert 'roughness' in result

        agsar.cleanup()


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def small_gpt2(self):
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000
        return AutoModelForCausalLM.from_config(config)

    @pytest.fixture
    def mock_tokenizer(self):
        class MockTokenizer:
            def __init__(self):
                self.pad_token = None
                self.eos_token = "<eos>"
                self.pad_token_id = 0
                self.eos_token_id = 1

            def encode(self, text, add_special_tokens=True):
                if not text.strip():
                    return [2]  # Return single token for empty
                return list(range(2, 2 + len(text.split())))

            def decode(self, token_ids):
                return " ".join([f"token_{i}" for i in token_ids])

        return MockTokenizer()

    def test_empty_response(self, small_gpt2, mock_tokenizer):
        """Test handling of empty response."""
        config = AGSARConfig(semantic_layers=2, uncertainty_metric="v31")
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        with torch.no_grad():
            score = agsar.compute_uncertainty("test prompt", "")

        # Should return 0.0 for empty response
        assert score == 0.0

        agsar.cleanup()

    def test_very_short_prompt(self, small_gpt2, mock_tokenizer):
        """Test with very short prompt (< sink_token_count)."""
        config = AGSARConfig(semantic_layers=2, sink_token_count=4)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Prompt with only 2 tokens
        prompt_ids = torch.randint(0, 100, (1, 2))

        with torch.no_grad():
            result = agsar.process_prompt(prompt_ids)

        # Should handle gracefully
        assert result['prompt_length'] == 2
        assert agsar._authority_history.shape[1] == 2

        agsar.cleanup()

    def test_long_sequence(self, small_gpt2, mock_tokenizer):
        """Test with longer sequence."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        # Longer prompt
        prompt_ids = torch.randint(0, 100, (1, 64))

        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        # Generate 32 tokens
        current_ids = prompt_ids.clone()
        for _ in range(32):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                score = agsar.process_step(current_ids)

            assert 0.0 <= score <= 1.0

        assert agsar._authority_history.shape[1] == 96

        agsar.cleanup()

    def test_batch_size_one(self, small_gpt2, mock_tokenizer):
        """Verify batch size 1 works correctly."""
        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(small_gpt2, mock_tokenizer, config)

        prompt_ids = torch.randint(0, 100, (1, 8))

        with torch.no_grad():
            agsar.process_prompt(prompt_ids)

        current_ids = torch.cat([prompt_ids, torch.randint(0, 100, (1, 1))], dim=-1)

        with torch.no_grad():
            score = agsar.process_step(current_ids)

        assert isinstance(score, float)

        agsar.cleanup()
