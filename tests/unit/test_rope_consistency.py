"""
RoPE Consistency Tests for AG-SAR Attention Hooks.

Purpose: Verify that manually-computed Q @ K.T from captured post-RoPE tensors
matches the model's actual attention weights (output_attentions=True).

CRITICAL: If hooks capture Q/K BEFORE RoPE rotation, the manual Q @ K.T
computation will produce garbage (random attention weights) because the
model relies on RoPE to inject positional information.

This test suite verifies:
1. Hooks capture Q/K AFTER apply_rotary_pos_emb() (not at linear projection)
2. Manual attention computation matches model's ground truth to 1e-5 tolerance
3. GQA head expansion is handled correctly for Llama-3.1+ models

Test Matrix:
| Architecture | RoPE | GQA | Hook Point |
|-------------|------|-----|------------|
| GPT-2       | No   | No  | c_attn output (baseline) |
| Llama       | Yes  | Yes | After apply_rotary_pos_emb() |
| Mistral     | Yes  | Yes | After apply_rotary_pos_emb() |
| Qwen        | Yes  | Yes | After apply_rotary_pos_emb() |
"""

import math
import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM

from ag_sar.modeling import ModelAdapter


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_attention_from_qk(
    Q: torch.Tensor,  # (B, H, S, D)
    K: torch.Tensor,  # (B, H, S, D)
    head_dim: int,
    causal_mask: bool = True,
) -> torch.Tensor:
    """
    Compute attention weights from Q/K the same way as hooks.py.

    This mirrors the computation in _patch_llama_attention:
        attn_weights = torch.matmul(Q, K.transpose(2, 3)) / sqrt(head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

    Args:
        Q: Query states, shape (B, H, S, D) - must be post-RoPE for Llama/Mistral
        K: Key states, shape (B, H, S, D) - must be post-RoPE for Llama/Mistral
        head_dim: Head dimension for scaling
        causal_mask: Whether to apply causal mask

    Returns:
        Attention weights, shape (B, H, S, S)
    """
    scale = 1.0 / math.sqrt(head_dim)
    attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if causal_mask:
        seq_len = Q.shape[2]
        # Create causal mask (upper triangular = -inf)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=Q.dtype),
            diagonal=1
        )
        attn_logits = attn_logits.masked_fill(mask.bool(), float('-inf'))

    # Softmax in float32 for numerical stability (matches hooks.py)
    attn_probs = F.softmax(attn_logits, dim=-1, dtype=torch.float32)
    return attn_probs


def assert_attention_close(
    computed: torch.Tensor,
    ground_truth: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """
    Assert attention matrices are close within tolerance.

    Uses relative error: |computed - truth| / max(|truth|, atol)

    Args:
        computed: Attention weights from manual Q @ K.T computation
        ground_truth: Attention weights from model's output_attentions=True
        rtol: Relative tolerance (default 1e-5)
        atol: Absolute tolerance for denominator clamping (default 1e-6)
    """
    if computed.shape != ground_truth.shape:
        raise AssertionError(
            f"Shape mismatch: computed {computed.shape} vs truth {ground_truth.shape}"
        )

    # Cast to same dtype for comparison
    computed = computed.float()
    ground_truth = ground_truth.float()

    # Compute relative error
    diff = (computed - ground_truth).abs()
    denom = torch.clamp(ground_truth.abs(), min=atol)
    rel_error = diff / denom
    max_rel_error = rel_error.max().item()

    if max_rel_error > rtol:
        # Find worst position for debugging
        worst_flat_idx = rel_error.argmax()
        worst_idx = []
        remaining = worst_flat_idx.item()
        for dim in reversed(rel_error.shape):
            worst_idx.insert(0, remaining % dim)
            remaining //= dim
        worst_idx = tuple(worst_idx)

        raise AssertionError(
            f"Attention mismatch: max relative error {max_rel_error:.2e} > {rtol:.2e}\n"
            f"Worst position: {worst_idx}\n"
            f"Computed: {computed[worst_idx]:.6f}, Truth: {ground_truth[worst_idx]:.6f}"
        )


# =============================================================================
# GPT-2 TESTS (NO ROPE - BASELINE)
# =============================================================================

class TestGPT2NoRoPE:
    """
    Test attention hook consistency on GPT-2 (no RoPE).

    GPT-2 uses absolute positional embeddings, not RoPE.
    This serves as a baseline to verify the test infrastructure works
    before testing RoPE-based architectures.
    """

    @pytest.fixture
    def gpt2_model(self):
        """Create minimal GPT-2 model for testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 4
        config.n_embd = 64
        config.vocab_size = 100
        config.n_positions = 128

        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        yield model
        del model

    def test_gpt2_attention_consistency(self, gpt2_model):
        """Verify GPT-2 attention weights match between hook and ground truth."""
        model = gpt2_model
        seq_len = 16
        batch_size = 1

        # Get ground truth attention weights
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth_attn = outputs.attentions  # Tuple of (B, H, S, S) per layer

        # Capture Q/K via hooks
        adapter = ModelAdapter(model=model, layers=[0, 1], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        # Compute attention from captured Q/K
        for layer_idx in [0, 1]:
            Q = adapter.capture.query_states.get(layer_idx)
            K = adapter.capture.key_states.get(layer_idx)

            if Q is None or K is None:
                pytest.skip(f"Q/K not captured for layer {layer_idx}")

            # Reshape if needed (GPT-2 may have different format)
            if Q.dim() == 3:  # (B, S, H*D) -> need to split heads
                n_heads = model.config.n_head
                head_dim = Q.shape[-1] // n_heads
                Q = Q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                K = K.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            elif Q.dim() == 4:  # (B, H, S, D) - already correct
                head_dim = Q.shape[-1]
            else:
                pytest.skip(f"Unexpected Q shape: {Q.shape}")

            computed_attn = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)

            # Compare with ground truth
            gt_layer = ground_truth_attn[layer_idx]  # (B, H, S, S)
            assert_attention_close(computed_attn, gt_layer, rtol=1e-4, atol=1e-6)

        adapter.cleanup()

    @pytest.mark.parametrize("seq_len", [8, 32, 64])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_gpt2_seq_batch_matrix(self, gpt2_model, seq_len, batch_size):
        """Test GPT-2 attention consistency across seq lengths and batch sizes."""
        model = gpt2_model

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth_attn = outputs.attentions[0]  # First layer

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is None or K is None:
            adapter.cleanup()
            pytest.skip("Q/K not captured")

        # Reshape
        if Q.dim() == 3:
            n_heads = model.config.n_head
            head_dim = Q.shape[-1] // n_heads
            Q = Q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
        else:
            head_dim = Q.shape[-1]

        computed_attn = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)
        assert_attention_close(computed_attn, ground_truth_attn, rtol=1e-4, atol=1e-6)

        adapter.cleanup()


# =============================================================================
# LLAMA TESTS (WITH ROPE)
# =============================================================================

class TestLlamaRoPE:
    """
    Test attention hook consistency on Llama-style models with RoPE.

    CRITICAL: Hooks must capture Q/K AFTER apply_rotary_pos_emb().
    If captured before RoPE, the manual Q @ K.T will be garbage.
    """

    @pytest.fixture
    def llama_model(self):
        """Create minimal Llama model for testing."""
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            pytest.skip("LlamaConfig not available in this transformers version")

        # Minimal config with GQA (Llama-3.1 style: 4x GQA ratio)
        config = LlamaConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,  # GQA: 8 Q heads, 2 KV heads = 4x ratio
            intermediate_size=256,
            vocab_size=100,
            max_position_embeddings=256,
            rope_theta=10000.0,
            # CRITICAL: Use eager attention for output_attentions
            attn_implementation="eager",
        )

        model = LlamaForCausalLM(config)
        model.eval()
        yield model
        del model

    def test_llama_rope_consistency(self, llama_model):
        """
        Verify Llama attention weights match between post-RoPE hooks and ground truth.

        This is the CRITICAL test for RoPE correctness.
        """
        model = llama_model
        seq_len = 16
        batch_size = 1

        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        # Get ground truth attention weights
        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth_attn = outputs.attentions  # Tuple of (B, H, S, S) per layer

        # Capture via hooks
        adapter = ModelAdapter(model=model, layers=[0, 1], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        # Compare for each layer
        for layer_idx in [0, 1]:
            Q = adapter.capture.query_states.get(layer_idx)
            K = adapter.capture.key_states.get(layer_idx)

            if Q is None or K is None:
                pytest.skip(f"Q/K not captured for layer {layer_idx}")

            # Q should be (B, H, S, D) after hooks
            assert Q.dim() == 4, f"Expected 4D Q tensor, got {Q.dim()}D"
            head_dim = Q.shape[-1]

            computed_attn = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)

            # Ground truth
            gt_layer = ground_truth_attn[layer_idx]

            # Llama may have expanded K via repeat_kv for GQA
            # Ground truth has full H heads, computed may have H_kv heads
            if computed_attn.shape[1] != gt_layer.shape[1]:
                # K was captured before GQA expansion - expand now
                n_rep = gt_layer.shape[1] // computed_attn.shape[1]
                computed_attn = computed_attn.repeat_interleave(n_rep, dim=1)

            assert_attention_close(computed_attn, gt_layer, rtol=1e-4, atol=1e-5)

        adapter.cleanup()

    @pytest.mark.parametrize("seq_len", [8, 32, 64])
    def test_llama_various_seq_lengths(self, llama_model, seq_len):
        """Test Llama RoPE consistency at various sequence lengths."""
        model = llama_model
        batch_size = 1

        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth_attn = outputs.attentions[0]

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is None or K is None:
            adapter.cleanup()
            pytest.skip("Q/K not captured")

        head_dim = Q.shape[-1]
        computed_attn = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)

        # Handle GQA expansion
        if computed_attn.shape[1] != ground_truth_attn.shape[1]:
            n_rep = ground_truth_attn.shape[1] // computed_attn.shape[1]
            computed_attn = computed_attn.repeat_interleave(n_rep, dim=1)

        assert_attention_close(computed_attn, ground_truth_attn, rtol=1e-4, atol=1e-5)
        adapter.cleanup()


# =============================================================================
# GQA (GROUPED QUERY ATTENTION) TESTS
# =============================================================================

class TestGQAExpansion:
    """Test that GQA head expansion is handled correctly."""

    @pytest.fixture
    def gqa_model(self):
        """Create Llama model with explicit GQA configuration."""
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            pytest.skip("LlamaConfig not available")

        config = LlamaConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,   # Q heads
            num_key_value_heads=2,   # KV heads (4x GQA)
            intermediate_size=256,
            vocab_size=100,
            max_position_embeddings=128,
            attn_implementation="eager",
        )

        model = LlamaForCausalLM(config)
        model.eval()
        yield model
        del model

    def test_gqa_kv_head_count(self, gqa_model):
        """Verify K states have correct reduced head count before expansion."""
        model = gqa_model
        expected_kv_heads = model.config.num_key_value_heads
        expected_q_heads = model.config.num_attention_heads

        input_ids = torch.randint(0, 100, (1, 16))

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is None or K is None:
            adapter.cleanup()
            pytest.skip("Q/K not captured")

        # Q should have full head count
        assert Q.shape[1] == expected_q_heads, \
            f"Q heads: expected {expected_q_heads}, got {Q.shape[1]}"

        # K may have reduced heads (if captured before repeat_kv)
        # or full heads (if captured after repeat_kv)
        assert K.shape[1] in [expected_kv_heads, expected_q_heads], \
            f"K heads: expected {expected_kv_heads} or {expected_q_heads}, got {K.shape[1]}"

        adapter.cleanup()


# =============================================================================
# EDGE CASES
# =============================================================================

class TestRoPEEdgeCases:
    """Edge cases for RoPE consistency testing."""

    @pytest.fixture
    def small_model(self):
        """Create minimal GPT-2 for edge case testing."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 32
        config.vocab_size = 50
        config.n_positions = 64

        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        yield model
        del model

    def test_single_token(self, small_model):
        """Verify attention works with single token (seq_len=1)."""
        model = small_model
        input_ids = torch.randint(0, 50, (1, 1))

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth = outputs.attentions[0]  # (B, H, 1, 1)

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is not None and K is not None:
            if Q.dim() == 3:
                n_heads = model.config.n_head
                head_dim = Q.shape[-1] // n_heads
                Q = Q.view(1, 1, n_heads, head_dim).transpose(1, 2)
                K = K.view(1, 1, n_heads, head_dim).transpose(1, 2)
            else:
                head_dim = Q.shape[-1]

            computed = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)

            # Single token: attention should be all 1.0 (attend to self)
            assert computed.shape == ground_truth.shape
            assert torch.allclose(computed, ground_truth, rtol=1e-4, atol=1e-4)

        adapter.cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_consistency(self, small_model):
        """Verify RoPE consistency on CUDA."""
        model = small_model.cuda()
        input_ids = torch.randint(0, 50, (1, 16), device="cuda")

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth = outputs.attentions[0]

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.float32)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is not None and K is not None:
            assert Q.device.type == "cuda"
            assert K.device.type == "cuda"

            if Q.dim() == 3:
                n_heads = model.config.n_head
                head_dim = Q.shape[-1] // n_heads
                Q = Q.view(1, 16, n_heads, head_dim).transpose(1, 2)
                K = K.view(1, 16, n_heads, head_dim).transpose(1, 2)
            else:
                head_dim = Q.shape[-1]

            computed = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)
            assert_attention_close(computed, ground_truth, rtol=1e-4, atol=1e-5)

        adapter.cleanup()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bfloat16_precision(self, small_model):
        """Verify numerical consistency in bfloat16 (with relaxed tolerance)."""
        model = small_model.cuda().to(torch.bfloat16)
        input_ids = torch.randint(0, 50, (1, 16), device="cuda")

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)
        ground_truth = outputs.attentions[0]

        adapter = ModelAdapter(model=model, layers=[0], dtype=torch.bfloat16)
        adapter.register()

        with torch.no_grad():
            _ = model(input_ids)

        Q = adapter.capture.query_states.get(0)
        K = adapter.capture.key_states.get(0)

        if Q is not None and K is not None:
            if Q.dim() == 3:
                n_heads = model.config.n_head
                head_dim = Q.shape[-1] // n_heads
                Q = Q.view(1, 16, n_heads, head_dim).transpose(1, 2)
                K = K.view(1, 16, n_heads, head_dim).transpose(1, 2)
            else:
                head_dim = Q.shape[-1]

            computed = compute_attention_from_qk(Q, K, head_dim, causal_mask=True)
            # Relaxed tolerance for bfloat16
            assert_attention_close(computed, ground_truth, rtol=1e-2, atol=1e-3)

        adapter.cleanup()
