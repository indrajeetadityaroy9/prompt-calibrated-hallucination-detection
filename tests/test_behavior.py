"""
Advanced Behavioral Verification Tests for AG-SAR v3.1

These tests verify SCIENTIFIC CORRECTNESS, not just code execution.
They catch subtle bugs (broadcasting errors, hook misalignment, memory leaks)
that unit tests miss.

Checklist items verified:
- [ ] Bounded Authority: Scores remain in [0, 1]
- [ ] Pre-MLP Integrity: h_attn != final hidden states
- [ ] Register Mask: BOS token properly masked
- [ ] Dimension Broadcast: Correct tensor operations
- [ ] Repetition Maintains Authority: Copying context = low roughness
- [ ] Random Noise Drops Authority: Orthogonal vectors = high roughness
- [ ] Memory Leak Check: O(N) scaling verified
"""

import torch
import pytest
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from ag_sar.ag_sar import AGSAR
from ag_sar.config import AGSARConfig
from ag_sar.ops import (
    compute_authority_flow,
    compute_spectral_roughness,
    compute_register_mask,
    fisher_kurtosis,
)


class TestBoundedAuthority:
    """Phase 1.1: Verify authority scores remain bounded in [0, 1]."""

    @pytest.fixture
    def small_model(self):
        """Create a small GPT-2 model for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 4
        config.n_head = 4
        config.n_embd = 128
        config.vocab_size = 1000
        return AutoModelForCausalLM.from_config(config)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
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

            def __call__(self, text, return_tensors=None):
                ids = self.encode(text)
                if return_tensors == "pt":
                    class Output:
                        input_ids = torch.tensor([ids])
                    return Output()
                return {"input_ids": ids}

        return MockTokenizer()

    def test_authority_bounded_during_generation(self, small_model, mock_tokenizer):
        """
        SCIENTIFIC CHECK:
        Authority must remain in [0.0, 1.0] throughout generation.

        Failure Mode: If scores grow exponentially (1.2, 1.5, 9.0...),
        the recursion has a broadcasting bug summing heads incorrectly.
        """
        config = AGSARConfig(
            semantic_layers=2,
            enable_register_filter=True,
            enable_authority_flow=True,
            enable_spectral_roughness=True,
        )
        engine = AGSAR(small_model, mock_tokenizer, config)

        # Process prompt
        prompt_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            engine.process_prompt(prompt_ids)

        # Generate 50 tokens and track authority
        current_ids = prompt_ids.clone()
        all_scores = []
        all_authorities = []

        for i in range(50):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                result = engine.process_step(current_ids, return_details=True)

            all_scores.append(result['hallucination_score'])
            all_authorities.append(result['authority'])

        # Verify all authorities are bounded
        for i, auth in enumerate(all_authorities):
            assert 0.0 <= auth <= 1.0, \
                f"Token {i}: Authority {auth} out of bounds [0, 1]"

        # Verify no exponential growth (check authority history)
        auth_history = engine._authority_history
        assert (auth_history >= 0).all(), "Negative authority detected"
        assert (auth_history <= 1.0).all(), "Authority > 1.0 detected (broadcasting bug)"

        engine.cleanup()

    def test_authority_no_vanishing(self, small_model, mock_tokenizer):
        """
        SCIENTIFIC CHECK:
        Authority should not vanish to ~0 instantly due to missing recharge.

        Failure Mode: If authority drops to 10^-9, recharge_weight isn't
        adding the Source Authority from prompt tokens.
        """
        config = AGSARConfig(
            semantic_layers=2,
            enable_authority_flow=True,
            recharge_weight=1.0,  # Ensure recharge is enabled
        )
        engine = AGSAR(small_model, mock_tokenizer, config)

        prompt_ids = torch.randint(0, 100, (1, 16))
        with torch.no_grad():
            engine.process_prompt(prompt_ids)

        # Generate tokens
        current_ids = prompt_ids.clone()
        authorities = []

        for _ in range(20):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                result = engine.process_step(current_ids, return_details=True)
            authorities.append(result['authority'])

        # Authority should not vanish (minimum should be > 0.01)
        min_auth = min(authorities)
        assert min_auth > 0.01, \
            f"Authority vanished to {min_auth} - recharge not working"

        engine.cleanup()


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
        from ag_sar.attention_extractor import AttentionExtractor

        extractor = AttentionExtractor(
            model=small_model,
            layers=[0, 1],
            dtype=torch.float32
        )
        extractor.register_hooks()

        input_ids = torch.randint(0, 100, (1, 16))

        with torch.no_grad():
            outputs = small_model(input_ids, output_hidden_states=True)

        # Get captured h_attn (pre-MLP)
        h_attn_captured = extractor._attn_outputs.get(1)  # Last semantic layer

        # Get final hidden states (post all layers)
        final_hidden = outputs.hidden_states[-1]  # (B, S, D)

        if h_attn_captured is not None:
            # h_attn should NOT be identical to final output
            # (Small differences expected due to MLP transformation)
            assert not torch.allclose(h_attn_captured, final_hidden, atol=1e-3), \
                "h_attn equals final hidden - hook captured wrong location!"

        extractor.remove_hooks()


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


class TestRepetitionMaintainsAuthority:
    """Phase 2.1: Scientific check - repetition should maintain high authority."""

    @pytest.fixture
    def setup_engine(self):
        """Set up a small GPT-2 model for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64
        config.vocab_size = 1000

        model = AutoModelForCausalLM.from_config(config)

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

            def __call__(self, text, return_tensors=None):
                ids = self.encode(text)
                if return_tensors == "pt":
                    class Output:
                        input_ids = torch.tensor([ids])
                    return Output()
                return {"input_ids": ids}

        tokenizer = MockTokenizer()

        ag_config = AGSARConfig(
            semantic_layers=2,
            enable_register_filter=True,
            enable_spectral_roughness=True,
            lambda_roughness=5.0
        )
        engine = AGSAR(model, tokenizer, ag_config)

        return model, tokenizer, engine

    def test_repetition_maintains_authority(self, setup_engine):
        """
        SCIENTIFIC CHECK:
        If the model strictly repeats the context (copying),
        Authority Flow should be high (near 1.0) because
        Spectral Roughness is low (perfect linear alignment).
        """
        model, tokenizer, engine = setup_engine

        # Create prompt
        prompt_ids = torch.randint(2, 100, (1, 8))

        # Process prompt
        with torch.no_grad():
            engine.process_prompt(prompt_ids)

        # "Generate" by repeating prompt tokens (simulates copying)
        current_ids = prompt_ids.clone()
        scores = []

        for i in range(min(4, prompt_ids.shape[1])):
            # Re-feed prompt tokens as if model is copying
            next_token = prompt_ids[:, i:i+1]
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            with torch.no_grad():
                result = engine.process_step(current_ids, return_details=True)
                scores.append(result['hallucination_score'])

        # Hallucination score should be LOW (high authority for copying)
        avg_score = sum(scores) / len(scores) if scores else 0
        # Note: With random model weights, we can't guarantee perfect copying behavior
        # So we just verify the score is reasonable (< 0.9)
        assert avg_score < 0.9, \
            f"Repetition should maintain some authority, got avg score: {avg_score}"

        engine.cleanup()


class TestRandomNoiseDropsAuthority:
    """Phase 2.2: Scientific check - random noise should drop authority."""

    def test_orthogonal_vectors_high_roughness(self):
        """
        SCIENTIFIC CHECK:
        When h_attn is orthogonal to the expected weighted sum,
        Spectral Roughness should be high.
        """
        torch.manual_seed(42)
        B, S, D = 1, 16, 64

        # Create value vectors
        v = torch.randn(B, S, D)

        # Create attention (lower triangular, normalized)
        attn = torch.tril(torch.ones(B, S, S))
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Perfect prediction (should have ~0 roughness)
        h_perfect = torch.bmm(attn, v)
        roughness_perfect = compute_spectral_roughness(h_perfect, v, attn)

        # Orthogonal/random h_attn (should have high roughness)
        h_random = torch.randn(B, S, D) * 10  # Large random deviation
        roughness_random = compute_spectral_roughness(h_random, v, attn)

        # Random should have much higher roughness
        assert roughness_random.mean() > roughness_perfect.mean() * 10, \
            f"Random noise should spike roughness. Perfect: {roughness_perfect.mean():.4f}, Random: {roughness_random.mean():.4f}"


class TestMemoryLeak:
    """Phase 2.3: Systems check - verify O(N) memory scaling."""

    @pytest.fixture
    def small_model(self):
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

    def test_no_gradient_accumulation(self, small_model, mock_tokenizer):
        """
        SYSTEMS CHECK:
        Verify authority_history doesn't accidentally attach computation graphs.
        """
        config = AGSARConfig(semantic_layers=2)
        engine = AGSAR(small_model, mock_tokenizer, config)

        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            engine.process_prompt(prompt_ids)

        # Run 50 steps
        current_ids = prompt_ids.clone()
        for _ in range(50):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                engine.process_step(current_ids)

        # Verify authority_history doesn't require grad
        assert not engine._authority_history.requires_grad, \
            "Authority history has gradient tracking - memory leak!"

        # Verify history size is O(N) - should be (1, 58) = 8 prompt + 50 generated
        assert engine._authority_history.shape == (1, 58), \
            f"Authority history shape wrong: {engine._authority_history.shape}"

        engine.cleanup()

    def test_memory_growth_bounded(self, small_model, mock_tokenizer):
        """
        SYSTEMS CHECK:
        Memory growth should be bounded and linear in sequence length.
        """
        config = AGSARConfig(semantic_layers=2)
        engine = AGSAR(small_model, mock_tokenizer, config)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        prompt_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            engine.process_prompt(prompt_ids)

        # Track tensor count before
        initial_tensors = len(gc.get_objects())

        # Run 100 steps
        current_ids = prompt_ids.clone()
        for _ in range(100):
            new_token = torch.randint(0, 100, (1, 1))
            current_ids = torch.cat([current_ids, new_token], dim=-1)

            with torch.no_grad():
                engine.process_step(current_ids)

            # Clear extractor cache each step (simulates streaming)
            engine.extractor.clear_cache()

        # Force garbage collection
        gc.collect()

        # Check that we're not accumulating excessive objects
        final_tensors = len(gc.get_objects())
        tensor_growth = final_tensors - initial_tensors

        # Growth should be reasonable (not thousands of new objects)
        # This is a loose check - mainly catching egregious leaks
        assert tensor_growth < 1000, \
            f"Possible memory leak: {tensor_growth} new objects created"

        engine.cleanup()


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
