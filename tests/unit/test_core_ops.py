"""
Unit tests for AG-SAR core operations (v8.0 Gold Master).

Tests the core mechanisms:
1. Authority Flow (compute_authority_flow, compute_authority_flow_vectorized)
2. MLP Divergence (compute_mlp_divergence) - The discriminative metric
3. GQA Alignment (align_gqa_heads)
"""

import pytest
import torch

from ag_sar.ops import (
    compute_mlp_divergence,
    compute_authority_flow,
    compute_authority_flow_vectorized,
    compute_stability_gate,
    align_gqa_heads,
)


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


class TestIntegration:
    """Integration tests for v8.0 pipeline."""

    def test_full_v80_pipeline(self):
        """Test complete v8.0 pipeline flow (Authority Flow + Stability Gate)."""
        torch.manual_seed(42)
        B, H, S, D = 2, 4, 32, 64
        prompt_length = 16

        # Simulate model outputs
        h_attn = torch.randn(B, S, D)  # Attention output
        h_block = torch.randn(B, S, D)  # Block output
        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)  # Attention weights

        # Step 1: Authority Flow
        authority = compute_authority_flow(attn, prompt_length)
        assert authority.shape == (B, S)
        assert (authority[:, :prompt_length] == 1.0).all()

        # Step 2: Stability Gate
        gate = compute_stability_gate(h_attn, h_block, sensitivity=1.0)
        assert gate.shape == (B, S)
        assert (gate >= 0).all()
        assert (gate <= 1).all()

        # Step 3: Final Authority (gated)
        final_authority = gate * authority
        assert final_authority.shape == (B, S)
        assert (final_authority >= 0).all()
        assert (final_authority <= 1).all()


class TestGQAAlignment:
    """Tests for GQA (Grouped Query Attention) alignment functions."""

    def test_align_gqa_heads_basic(self):
        """Test basic GQA head expansion for Llama-3.1-8B config."""
        B, n_kv_heads, S, head_dim = 2, 8, 64, 128
        n_q_heads = 32  # Llama-3.1-8B: 32 Q heads, 8 KV heads

        v_states = torch.randn(B, n_kv_heads, S, head_dim)
        v_aligned = align_gqa_heads(v_states, n_q_heads)

        # Should expand to (B, 32, S, head_dim)
        assert v_aligned.shape == (B, n_q_heads, S, head_dim)

    def test_align_gqa_heads_repetition_pattern(self):
        """Verify each KV head is repeated correctly."""
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
        B, H, S, head_dim = 2, 32, 64, 128
        v_states = torch.randn(B, H, S, head_dim)

        v_aligned = align_gqa_heads(v_states, n_q_heads=32)

        assert v_aligned.shape == v_states.shape
        assert torch.allclose(v_aligned, v_states)

    def test_align_gqa_heads_3d_passthrough(self):
        """3D input (B, S, D) should be returned as-is."""
        B, S, D = 2, 64, 4096
        v_states = torch.randn(B, S, D)

        v_aligned = align_gqa_heads(v_states, n_q_heads=32)

        assert v_aligned.shape == v_states.shape
        assert torch.allclose(v_aligned, v_states)


class TestMLPDivergence:
    """
    Tests for v3.2 MLP Divergence metric.

    MLP Divergence is the primary discriminative metric for hallucination detection.
    It measures: δ(t) = 1 - CosineSim(h_attn, h_block)

    Hypothesis: When a model hallucinates, the MLP layer overrides the attention
    layer's signal with parametric memory, causing divergence.
    """

    def test_identical_vectors_zero_divergence(self):
        """When h_attn == h_block, divergence should be 0."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        h_block = h_attn.clone()

        divergence = compute_mlp_divergence(h_attn, h_block)

        assert torch.allclose(divergence, torch.zeros(B, S), atol=1e-5), \
            f"Identical vectors should have zero divergence, got max={divergence.max():.6f}"

    def test_orthogonal_vectors_high_divergence(self):
        """When h_attn and h_block are orthogonal, divergence should be ~1."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        # Create orthogonal vectors using QR decomposition
        h_attn = torch.randn(B, S, D)
        noise = torch.randn(B, S, D)

        # Gram-Schmidt: make noise orthogonal to h_attn
        h_attn_norm = h_attn / h_attn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = (noise * h_attn_norm).sum(dim=-1, keepdim=True) * h_attn_norm
        h_block = noise - proj  # orthogonal component

        divergence = compute_mlp_divergence(h_attn, h_block)

        # Orthogonal vectors have cosine similarity = 0, so divergence = 1
        assert (divergence.abs() - 1.0).abs().mean() < 0.1, \
            f"Orthogonal vectors should have divergence ~1, got mean={divergence.mean():.4f}"

    def test_opposite_vectors_max_divergence(self):
        """When h_attn and h_block are opposite, divergence should be ~2."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        h_block = -h_attn  # Opposite direction

        divergence = compute_mlp_divergence(h_attn, h_block)

        # Opposite vectors have cosine similarity = -1, so divergence = 2
        assert torch.allclose(divergence, torch.full((B, S), 2.0), atol=0.01), \
            f"Opposite vectors should have divergence ~2, got mean={divergence.mean():.4f}"

    def test_divergence_bounded(self):
        """Divergence should always be in [0, 2]."""
        torch.manual_seed(42)
        B, S, D = 4, 32, 128

        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)

        divergence = compute_mlp_divergence(h_attn, h_block)

        assert torch.all(divergence >= 0.0), "Divergence cannot be negative"
        assert torch.all(divergence <= 2.0), "Divergence cannot exceed 2"

    def test_divergence_detects_mlp_shift(self):
        """MLP Divergence should detect when MLP shifts the representation."""
        torch.manual_seed(42)
        B, S, D = 1, 16, 64

        # Simulate grounded: MLP refines but doesn't contradict
        h_attn_grounded = torch.randn(B, S, D)
        h_block_grounded = h_attn_grounded + 0.1 * torch.randn(B, S, D)

        # Simulate hallucination: MLP overrides attention signal
        h_attn_halluc = torch.randn(B, S, D)
        h_block_halluc = torch.randn(B, S, D)  # Unrelated direction

        div_grounded = compute_mlp_divergence(h_attn_grounded, h_block_grounded)
        div_halluc = compute_mlp_divergence(h_attn_halluc, h_block_halluc)

        # Hallucination should have higher divergence
        assert div_halluc.mean() > div_grounded.mean(), \
            f"Hallucination divergence ({div_halluc.mean():.4f}) should exceed grounded ({div_grounded.mean():.4f})"

    def test_output_shape(self):
        """Output should be (B, S)."""
        B, S, D = 2, 32, 128
        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)

        divergence = compute_mlp_divergence(h_attn, h_block)

        assert divergence.shape == (B, S)

    def test_attention_mask_applied(self):
        """Attention mask should zero out padded positions."""
        B, S, D = 2, 16, 64
        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)

        # Mask: first 8 tokens valid, last 8 padded
        mask = torch.zeros(B, S)
        mask[:, :8] = 1.0

        divergence = compute_mlp_divergence(h_attn, h_block, attention_mask=mask)

        # Padded positions should be zero
        assert torch.allclose(divergence[:, 8:], torch.zeros(B, S - 8)), \
            "Padded positions should have zero divergence"


class TestStabilityGate:
    """Tests for compute_stability_gate function."""

    def test_identical_vectors_high_gate(self):
        """When h_attn == h_block, gate should be ~1.0 (stable)."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        h_block = h_attn.clone()

        gate = compute_stability_gate(h_attn, h_block, sensitivity=1.0)

        assert torch.allclose(gate, torch.ones(B, S), atol=1e-4), \
            f"Identical vectors should have gate ~1.0, got mean={gate.mean():.4f}"

    def test_orthogonal_vectors_low_gate(self):
        """When h_attn and h_block are orthogonal, gate should be ~exp(-1)."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        noise = torch.randn(B, S, D)

        # Gram-Schmidt: make noise orthogonal to h_attn
        h_attn_norm = h_attn / h_attn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        proj = (noise * h_attn_norm).sum(dim=-1, keepdim=True) * h_attn_norm
        h_block = noise - proj

        gate = compute_stability_gate(h_attn, h_block, sensitivity=1.0)

        # Gate = exp(-sensitivity * divergence) = exp(-1 * 1) = 0.368
        expected = torch.exp(torch.tensor(-1.0))
        assert (gate.mean() - expected).abs() < 0.1, \
            f"Orthogonal vectors should have gate ~{expected:.3f}, got {gate.mean():.4f}"

    def test_gate_in_valid_range(self):
        """Gate should always be in [0, 1]."""
        torch.manual_seed(42)
        B, S, D = 4, 32, 128

        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)

        gate = compute_stability_gate(h_attn, h_block, sensitivity=1.0)

        assert torch.all(gate >= 0.0), "Gate cannot be negative"
        assert torch.all(gate <= 1.0), "Gate cannot exceed 1"

    def test_sensitivity_affects_gate(self):
        """Higher sensitivity should make gate more sensitive to divergence."""
        torch.manual_seed(42)
        B, S, D = 2, 16, 64

        h_attn = torch.randn(B, S, D)
        h_block = torch.randn(B, S, D)  # Random (divergent)

        gate_low = compute_stability_gate(h_attn, h_block, sensitivity=0.1)
        gate_high = compute_stability_gate(h_attn, h_block, sensitivity=10.0)

        # Higher sensitivity should result in lower gate values (more sensitive)
        assert gate_high.mean() < gate_low.mean(), \
            f"Higher sensitivity should give lower gate: {gate_high.mean():.4f} vs {gate_low.mean():.4f}"


