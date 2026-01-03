"""
Unit tests for AG-SAR v3.1 operations.

Tests the four v3.1 mechanisms:
1. Register Filter (fisher_kurtosis, compute_register_mask, EMAState)
2. Authority Flow (compute_authority_flow, compute_authority_flow_vectorized)
3. Spectral Roughness (compute_spectral_roughness)
4. SnapKV Eviction (compute_snapkv_eviction, compress_kv_cache)
"""

import pytest
import torch

from ag_sar.ops import (
    fisher_kurtosis,
    welford_update,
    compute_register_mask,
    compute_spectral_roughness,
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_snapkv_eviction,
    compress_kv_cache,
    EMAState,
)


class TestFisherKurtosis:
    """Tests for fisher_kurtosis function."""

    def test_normal_distribution_kurtosis(self):
        """Normal distribution should have kurtosis ≈ 0."""
        torch.manual_seed(42)
        # Large sample for stability
        x = torch.randn(1000, 256)
        kurt = fisher_kurtosis(x, dim=-1)
        # Fisher kurtosis of normal ≈ 0 (excess kurtosis)
        assert kurt.abs().mean() < 0.5, f"Expected ~0, got {kurt.mean():.4f}"

    def test_uniform_distribution_kurtosis(self):
        """Uniform distribution should have negative kurtosis (platykurtic)."""
        torch.manual_seed(42)
        x = torch.rand(100, 256) * 2 - 1  # Uniform [-1, 1]
        kurt = fisher_kurtosis(x, dim=-1)
        # Uniform has kurtosis = -1.2
        assert kurt.mean() < 0, f"Expected negative kurtosis, got {kurt.mean():.4f}"

    def test_spiky_distribution_kurtosis(self):
        """Spiky distribution should have positive kurtosis (leptokurtic)."""
        torch.manual_seed(42)
        # Create spiky distribution (mostly zeros with occasional large values)
        x = torch.zeros(100, 256)
        x[:, ::10] = torch.randn(100, 26) * 10  # Sparse large values
        kurt = fisher_kurtosis(x, dim=-1)
        # Spiky distributions have high positive kurtosis
        assert kurt.mean() > 0, f"Expected positive kurtosis, got {kurt.mean():.4f}"

    def test_output_shape(self):
        """Output should reduce the specified dimension."""
        x = torch.randn(4, 8, 128)
        kurt = fisher_kurtosis(x, dim=-1)
        assert kurt.shape == (4, 8)

        kurt = fisher_kurtosis(x, dim=1)
        assert kurt.shape == (4, 128)


class TestWelfordUpdate:
    """Tests for welford_update function."""

    def test_mean_converges(self):
        """EMA mean should converge to sample mean."""
        torch.manual_seed(42)
        samples = torch.randn(100) + 5.0  # Mean ≈ 5.0

        running_mean = torch.tensor(0.0)
        running_var = torch.tensor(1.0)

        for s in samples:
            running_mean, running_var = welford_update(
                s, running_mean, running_var, decay=0.9
            )

        # Should be close to true mean
        assert abs(running_mean.item() - 5.0) < 1.0

    def test_variance_positive(self):
        """Variance should always be positive."""
        running_mean = torch.tensor(0.0)
        running_var = torch.tensor(1.0)

        for _ in range(50):
            val = torch.randn(1).squeeze()
            running_mean, running_var = welford_update(
                val, running_mean, running_var, decay=0.99
            )
            assert running_var >= 0


class TestEMAState:
    """Tests for EMAState dataclass."""

    def test_initialization(self):
        """Test EMAState.initialize factory method."""
        state = EMAState.initialize(
            num_layers=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            init_mean=1.0,
            init_var=2.0,
        )
        assert state.mean.shape == (4,)
        assert state.var.shape == (4,)
        assert state.count == 0
        assert (state.mean == 1.0).all()
        assert (state.var == 2.0).all()


class TestComputeRegisterMask:
    """Tests for compute_register_mask function."""

    def test_sink_tokens_masked(self):
        """First sink_token_count tokens should have mask = 0."""
        torch.manual_seed(42)
        v = torch.randn(2, 16, 64)
        mask, _ = compute_register_mask(v, sink_token_count=4)

        assert mask.shape == (2, 16)
        # First 4 tokens should be 0
        assert (mask[:, :4] == 0).all()

    def test_mask_in_valid_range(self):
        """Mask values should be in [0, 1]."""
        torch.manual_seed(42)
        v = torch.randn(2, 32, 128)
        mask, _ = compute_register_mask(v)

        assert (mask >= 0).all()
        assert (mask <= 1).all()

    def test_ema_state_updated(self):
        """EMA state should be updated when update_ema=True."""
        torch.manual_seed(42)
        v = torch.randn(2, 16, 64)

        _, state1 = compute_register_mask(v, ema_state=None)
        assert state1.count == 1

        v2 = torch.randn(2, 16, 64)
        _, state2 = compute_register_mask(v2, ema_state=state1, update_ema=True)
        assert state2.count == 2

    def test_ema_state_not_updated(self):
        """EMA state should not change when update_ema=False."""
        torch.manual_seed(42)
        v = torch.randn(2, 16, 64)
        _, state1 = compute_register_mask(v, ema_state=None)

        v2 = torch.randn(2, 16, 64)
        mask, state2 = compute_register_mask(v2, ema_state=state1, update_ema=False)
        # State should be same object (not updated)
        assert state1 is state2


class TestComputeSpectralRoughness:
    """Tests for compute_spectral_roughness function."""

    def test_perfect_alignment_zero_roughness(self):
        """When h_attn = weighted sum of values, roughness should be ~0."""
        torch.manual_seed(42)
        B, S, D = 2, 8, 32

        # Create attention weights (lower triangular, normalized)
        attn = torch.tril(torch.ones(B, S, S))
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Create value vectors
        v = torch.randn(B, S, D)

        # h_attn = perfect weighted sum
        h_attn = torch.bmm(attn, v)

        roughness = compute_spectral_roughness(h_attn, v, attn)

        # Should be very close to 0
        assert roughness.abs().max() < 1e-5

    def test_output_shape(self):
        """Output should be (B, S)."""
        B, S, D, H = 2, 16, 64, 4
        h_attn = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        roughness = compute_spectral_roughness(h_attn, v, attn)
        assert roughness.shape == (B, S)

    def test_handles_head_dimension(self):
        """Should work with (B, H, S, S) attention."""
        B, H, S, D = 2, 4, 8, 32
        h_attn = torch.randn(B, S, D)
        v = torch.randn(B, S, D)
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        roughness = compute_spectral_roughness(h_attn, v, attn)
        assert roughness.shape == (B, S)


class TestComputeAuthorityFlow:
    """Tests for compute_authority_flow function."""

    def test_prompt_tokens_have_full_authority(self):
        """Prompt tokens should have authority = 1.0."""
        B, S = 2, 16
        prompt_length = 8
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        authority = compute_authority_flow(attn, prompt_length)

        assert authority.shape == (B, S)
        # Prompt tokens should be 1.0
        assert (authority[:, :prompt_length] == 1.0).all()

    def test_authority_in_valid_range(self):
        """Authority should be in [0, 1]."""
        B, S = 2, 32
        prompt_length = 10
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        authority = compute_authority_flow(attn, prompt_length)

        assert (authority >= 0).all()
        assert (authority <= 1).all()

    def test_register_mask_reduces_authority(self):
        """Register mask should reduce authority for masked tokens."""
        B, S = 2, 16
        prompt_length = 8
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        # No mask
        auth_no_mask = compute_authority_flow(attn, prompt_length, register_mask=None)

        # Mask that zeros out some tokens
        mask = torch.ones(B, S)
        mask[:, 10:] = 0.0
        auth_with_mask = compute_authority_flow(attn, prompt_length, register_mask=mask)

        # Masked tokens should have lower or equal authority
        assert (auth_with_mask[:, 10:] <= auth_no_mask[:, 10:]).all()

    def test_previous_authority_used(self):
        """Should use previous_authority when provided."""
        B, S = 2, 16
        prompt_length = 8
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        prev_auth = torch.ones(B, S) * 0.5
        prev_auth[:, :prompt_length] = 1.0

        authority = compute_authority_flow(
            attn, prompt_length, previous_authority=prev_auth
        )

        # Generated tokens should use previous authority
        assert authority.shape == (B, S)


class TestComputeAuthorityFlowVectorized:
    """Tests for compute_authority_flow_vectorized function."""

    def test_prompt_tokens_have_full_authority(self):
        """Prompt tokens should have authority = 1.0."""
        B, S = 2, 16
        prompt_length = 8
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        authority = compute_authority_flow_vectorized(attn, prompt_length)

        assert (authority[:, :prompt_length] == 1.0).all()

    def test_authority_in_valid_range(self):
        """Authority should be in [0, 1]."""
        B, S = 2, 32
        prompt_length = 10
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        authority = compute_authority_flow_vectorized(attn, prompt_length)

        assert (authority >= 0).all()
        assert (authority <= 1).all()


class TestComputeSnapKVEviction:
    """Tests for compute_snapkv_eviction function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        B, H, S = 2, 4, 64
        budget = 16
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)
        authority = torch.rand(B, S)

        indices = compute_snapkv_eviction(
            attn, authority, budget=budget, observation_window=8
        )

        # Should return (B, H, budget) indices
        assert indices.shape[0] == B
        assert indices.shape[1] == H

    def test_sink_tokens_always_kept(self):
        """Sink tokens should always be included in keep_indices."""
        B, H, S = 2, 4, 64
        budget = 20
        sink_count = 4
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)
        authority = torch.rand(B, S)

        indices = compute_snapkv_eviction(
            attn, authority, budget=budget, sink_token_count=sink_count
        )

        # First sink_count indices should be 0, 1, 2, 3
        for b in range(B):
            for h in range(H):
                kept = set(indices[b, h].tolist())
                for i in range(sink_count):
                    assert i in kept, f"Sink token {i} not kept"

    def test_indices_sorted(self):
        """Indices should be sorted for contiguous memory access."""
        B, H, S = 2, 4, 64
        budget = 16
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)
        authority = torch.rand(B, S)

        indices = compute_snapkv_eviction(attn, authority, budget=budget)

        # Check sorted
        for b in range(B):
            for h in range(H):
                sorted_indices, _ = indices[b, h].sort()
                assert (indices[b, h] == sorted_indices).all()


class TestCompressKVCache:
    """Tests for compress_kv_cache function."""

    def test_output_shape(self):
        """Compressed KV should have reduced sequence length."""
        B, H, S, D = 2, 4, 64, 32
        budget = 16
        obs_window = 8

        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        indices = torch.randint(0, S - obs_window, (B, H, budget))
        indices, _ = indices.sort(dim=-1)  # Must be sorted

        k_comp, v_comp = compress_kv_cache(k, v, indices, observation_window=obs_window)

        # New length = budget + observation_window
        expected_len = budget + obs_window
        assert k_comp.shape == (B, H, expected_len, D)
        assert v_comp.shape == (B, H, expected_len, D)

    def test_observation_window_preserved(self):
        """Last observation_window tokens should be unchanged."""
        B, H, S, D = 2, 4, 64, 32
        budget = 16
        obs_window = 8

        k = torch.randn(B, H, S, D)
        v = torch.randn(B, H, S, D)
        indices = torch.randint(0, S - obs_window, (B, H, budget))
        indices, _ = indices.sort(dim=-1)

        k_comp, v_comp = compress_kv_cache(k, v, indices, observation_window=obs_window)

        # Last obs_window tokens should match original
        assert torch.allclose(k_comp[:, :, -obs_window:, :], k[:, :, -obs_window:, :])
        assert torch.allclose(v_comp[:, :, -obs_window:, :], v[:, :, -obs_window:, :])


class TestIntegration:
    """Integration tests for v3.1 pipeline."""

    def test_full_v31_pipeline(self):
        """Test complete v3.1 pipeline flow."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 32, 64
        prompt_length = 16

        # Simulate model outputs
        v = torch.randn(B, S, D)  # Value vectors
        h_attn = torch.randn(B, S, D)  # Attention output
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)  # Attention weights

        # Step 1: Register Filter
        mask, ema_state = compute_register_mask(v, sink_token_count=4)
        assert mask.shape == (B, S)
        assert ema_state is not None

        # Step 2: Authority Flow
        authority = compute_authority_flow(attn, prompt_length, register_mask=mask)
        assert authority.shape == (B, S)
        assert (authority[:, :prompt_length] == 1.0).all()

        # Step 3: Spectral Roughness
        roughness = compute_spectral_roughness(h_attn, v, attn)
        assert roughness.shape == (B, S)

        # Step 4: Final Authority
        lambda_roughness = 10.0
        authority_final = authority / (1.0 + lambda_roughness * roughness)
        authority_final = authority_final.clamp(0.0, 1.0)
        assert (authority_final >= 0).all()
        assert (authority_final <= 1).all()

        # Step 5: SnapKV Eviction (optional)
        indices = compute_snapkv_eviction(
            attn, authority_final, budget=8, observation_window=4
        )
        assert indices.shape[0] == B


class TestGQAAlignment:
    """Tests for GQA (Grouped Query Attention) alignment functions."""

    def test_align_gqa_heads_basic(self):
        """Test basic GQA head expansion for Llama-3.1-8B config."""
        from ag_sar.ops import align_gqa_heads

        B, n_kv_heads, S, head_dim = 2, 8, 64, 128
        n_q_heads = 32  # Llama-3.1-8B: 32 Q heads, 8 KV heads

        v_states = torch.randn(B, n_kv_heads, S, head_dim)
        v_aligned = align_gqa_heads(v_states, n_q_heads)

        # Should expand to (B, 32, S, head_dim)
        assert v_aligned.shape == (B, n_q_heads, S, head_dim)

    def test_align_gqa_heads_repetition_pattern(self):
        """Verify each KV head is repeated correctly."""
        from ag_sar.ops import align_gqa_heads

        B, n_kv_heads, S, head_dim = 1, 8, 16, 64
        n_q_heads = 32

        # Create distinct values per KV head
        v_states = torch.arange(n_kv_heads).float().view(1, n_kv_heads, 1, 1)
        v_states = v_states.expand(B, n_kv_heads, S, head_dim)

        v_aligned = align_gqa_heads(v_states, n_q_heads)

        # Each KV head should be repeated 4 times (32/8 = 4)
        n_rep = n_q_heads // n_kv_heads
        for kv_idx in range(n_kv_heads):
            for rep in range(n_rep):
                q_idx = kv_idx * n_rep + rep
                assert torch.allclose(
                    v_aligned[0, q_idx, 0, 0],
                    torch.tensor(float(kv_idx))
                ), f"Q head {q_idx} should match KV head {kv_idx}"

    def test_align_gqa_heads_mha_passthrough(self):
        """MHA (n_kv_heads == n_q_heads) should return unchanged."""
        from ag_sar.ops import align_gqa_heads

        B, H, S, head_dim = 2, 32, 64, 128
        v_states = torch.randn(B, H, S, head_dim)

        v_aligned = align_gqa_heads(v_states, n_q_heads=32)

        assert v_aligned.shape == v_states.shape
        assert torch.allclose(v_aligned, v_states)

    def test_align_gqa_heads_3d_passthrough(self):
        """3D input (B, S, D) should be returned as-is."""
        from ag_sar.ops import align_gqa_heads

        B, S, D = 2, 64, 4096
        v_states = torch.randn(B, S, D)

        v_aligned = align_gqa_heads(v_states, n_q_heads=32)

        assert v_aligned.shape == v_states.shape
        assert torch.allclose(v_aligned, v_states)


class TestSpectralRoughnessGQA:
    """Tests for GQA-compatible spectral roughness."""

    def test_roughness_gqa_perfect_alignment(self):
        """When h_attn = A @ v_aligned, roughness should be ~0."""
        from ag_sar.ops import compute_spectral_roughness_gqa, align_gqa_heads

        B, n_q_heads, n_kv_heads, S, head_dim = 2, 32, 8, 16, 64

        # Create GQA-style tensors
        v_states = torch.randn(B, n_kv_heads, S, head_dim)
        v_aligned = align_gqa_heads(v_states, n_q_heads)

        # Create attention weights (lower triangular, normalized)
        attn = torch.tril(torch.ones(B, n_q_heads, S, S))
        attn = attn / attn.sum(dim=-1, keepdim=True)

        # Perfect prediction: h_attn = A @ v_aligned
        h_attn = torch.matmul(attn, v_aligned)

        roughness = compute_spectral_roughness_gqa(h_attn, v_states, attn, n_q_heads)

        # Should be very close to 0
        assert roughness.abs().max() < 1e-5

    def test_roughness_gqa_output_shape(self):
        """Output should be (B, S)."""
        from ag_sar.ops import compute_spectral_roughness_gqa

        B, n_q_heads, n_kv_heads, S, head_dim = 2, 32, 8, 64, 128

        h_attn = torch.randn(B, n_q_heads, S, head_dim)
        v_states = torch.randn(B, n_kv_heads, S, head_dim)
        attn = torch.softmax(torch.randn(B, n_q_heads, S, S), dim=-1)

        roughness = compute_spectral_roughness_gqa(h_attn, v_states, attn, n_q_heads)

        assert roughness.shape == (B, S)

    def test_roughness_gqa_fallback_to_standard(self):
        """3D input should fall back to standard compute_spectral_roughness."""
        from ag_sar.ops import compute_spectral_roughness_gqa

        B, S, D = 2, 32, 256

        h_attn = torch.randn(B, S, D)
        v_states = torch.randn(B, S, D)
        attn = torch.softmax(torch.randn(B, S, S), dim=-1)

        roughness = compute_spectral_roughness_gqa(h_attn, v_states, attn)

        assert roughness.shape == (B, S)

    def test_roughness_gqa_llama31_8b_config(self):
        """Test with actual Llama-3.1-8B dimensions."""
        from ag_sar.ops import compute_spectral_roughness_gqa

        # Llama-3.1-8B: 32 Q heads, 8 KV heads, head_dim=128
        B, n_q_heads, n_kv_heads, S, head_dim = 1, 32, 8, 128, 128

        h_attn = torch.randn(B, n_q_heads, S, head_dim)
        v_states = torch.randn(B, n_kv_heads, S, head_dim)
        attn = torch.softmax(torch.randn(B, n_q_heads, S, S), dim=-1)

        roughness = compute_spectral_roughness_gqa(h_attn, v_states, attn, n_q_heads)

        assert roughness.shape == (B, S)
        assert (roughness >= 0).all()  # L2 norm is non-negative
