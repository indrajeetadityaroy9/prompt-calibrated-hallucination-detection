"""
QA Tests: Mathematical Kernel Verification for AG-SAR v8.0 Gold Master.

Purpose: Verify Authority Flow and supporting kernels against known ground truths.

Checklist items verified:
- [ ] Authority Flow: Prompt tokens have full authority, authority bounded [0,1]
"""

import torch
import pytest
from ag_sar.ops import (
    compute_authority_flow,
)


class TestAuthorityRecharge:
    """Verify Authority Flow with Prompt Recharge."""

    def test_prompt_tokens_have_full_authority(self):
        """Prompt tokens should always have authority = 1.0."""
        B, S = 2, 16
        prompt_length = 8

        attn = torch.softmax(torch.randn(B, S, S), dim=-1)
        authority = compute_authority_flow(attn, prompt_length)

        assert (authority[:, :prompt_length] == 1.0).all(), \
            "All prompt tokens should have authority = 1.0"

    def test_attending_only_to_prompt_gives_high_authority(self):
        """If generated token attends ONLY to prompt, authority ≈ 1.0."""
        B, S = 1, 16
        prompt_length = 8

        # Create attention that only attends to prompt
        attn = torch.zeros(B, S, S)
        # Generated tokens attend uniformly to prompt only
        for t in range(prompt_length, S):
            attn[:, t, :prompt_length] = 1.0 / prompt_length

        # Make it valid (lower triangular + prompt self-attention)
        for t in range(prompt_length):
            attn[:, t, :t+1] = 1.0 / (t + 1)

        authority = compute_authority_flow(attn, prompt_length)

        # Generated tokens should have authority close to 1.0
        gen_authority = authority[:, prompt_length:]
        assert (gen_authority > 0.95).all(), \
            f"Attending only to prompt should give authority ~1.0, got min={gen_authority.min().item():.4f}"

    def test_attending_only_to_generated_decays_authority(self):
        """If generated token only attends to other generated tokens with partial attention, authority decays."""
        B, S = 1, 32
        prompt_length = 8

        # Create attention where generated tokens partially attend to earlier tokens
        # This simulates a scenario where information is "diluted" through the chain
        attn = torch.zeros(B, S, S)

        # Prompt tokens self-attend
        for t in range(prompt_length):
            attn[:, t, :t+1] = 1.0 / (t + 1)

        # First generated token attends 50% to prompt, 50% to itself (sink behavior)
        # This creates an initial authority < 1.0
        attn[:, prompt_length, :prompt_length] = 0.5 / prompt_length
        attn[:, prompt_length, prompt_length] = 0.5  # Self-attention (no authority from self)

        # Subsequent generated tokens also split attention - partial to previous, partial to self
        for t in range(prompt_length + 1, S):
            # 50% to previous generated tokens
            prev_count = t - prompt_length
            if prev_count > 0:
                attn[:, t, prompt_length:t] = 0.5 / prev_count
            # 50% self-attention (contributes nothing since own authority is 0 at computation time)
            attn[:, t, t] = 0.5

        authority = compute_authority_flow(attn, prompt_length)

        # First generated token should have authority ~0.5 (50% from prompt, 50% from nothing)
        first_gen_auth = authority[:, prompt_length].item()
        assert first_gen_auth < 0.9, f"First gen should have reduced authority, got {first_gen_auth:.4f}"

        # Authority should decay as we go further from prompt
        later_authority = authority[:, -5:].mean()
        early_authority = authority[:, prompt_length:prompt_length+5].mean()

        assert later_authority < early_authority, \
            f"Authority should decay: early={early_authority:.4f}, later={later_authority:.4f}"

    def test_authority_bounded_zero_one(self):
        """Authority should always be in [0, 1]."""
        torch.manual_seed(42)
        B, S = 4, 64
        prompt_length = 16

        attn = torch.softmax(torch.randn(B, S, S), dim=-1)
        authority = compute_authority_flow(attn, prompt_length)

        assert (authority >= 0).all(), "Authority should be >= 0"
        assert (authority <= 1).all(), "Authority should be <= 1"


class TestDeviceConsistency:
    """Verify operations work on different devices."""

    def test_cpu_operations(self):
        """All operations should work on CPU."""
        torch.manual_seed(42)
        device = torch.device("cpu")

        attn = torch.softmax(torch.randn(2, 16, 16, device=device), dim=-1)

        # Authority flow should work without error
        authority = compute_authority_flow(attn, prompt_length=8)

        assert authority.device == device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_operations(self):
        """All operations should work on CUDA."""
        torch.manual_seed(42)
        device = torch.device("cuda")

        attn = torch.softmax(torch.randn(2, 16, 16, device=device), dim=-1)

        # Authority flow should work without error
        authority = compute_authority_flow(attn, prompt_length=8)

        assert authority.device.type == "cuda"


class TestBatchAndMultiHead:
    """Verify operations handle batch size > 1 and multi-head attention."""

    def test_batch_size_handling(self):
        """Operations should work with various batch sizes."""
        torch.manual_seed(42)

        for B in [1, 2, 4, 8]:
            S = 16
            attn = torch.softmax(torch.randn(B, S, S), dim=-1)

            authority = compute_authority_flow(attn, prompt_length=8)

            assert authority.shape == (B, S), f"Batch {B}: authority shape mismatch"

    def test_multi_head_attention(self):
        """Operations should handle multi-head attention (B, H, S, S)."""
        torch.manual_seed(42)
        B, H, S = 2, 8, 16

        attn = torch.softmax(torch.randn(B, H, S, S), dim=-1)

        authority = compute_authority_flow(attn, prompt_length=8)

        assert authority.shape == (B, S), f"Multi-head authority shape: {authority.shape}"


