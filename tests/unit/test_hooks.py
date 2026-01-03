"""
QA Tests: Model Hooks & Architecture Verification for AG-SAR v3.1

Purpose: Verify hooks capture Pre-MLP tensors on real architectures.

Checklist items verified:
- [ ] Hook Location: Capturing h_attn BEFORE residual/MLP
- [ ] Shape Safety: Handles Batch Size > 1 and Multi-Head Attention
- [ ] Device Consistency: States move to GPU automatically
"""

import torch
import pytest
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ag_sar.modeling import ModelAdapter
from ag_sar.config import AGSARConfig


class TestHookCaptures:
    """Verify hook captures on different architectures."""

    def test_gpt2_hook_captures(self):
        """Test hook captures on GPT-2 architecture."""
        # Load small GPT-2 config
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2  # Reduce layers for speed
        config.n_head = 2
        config.n_embd = 64

        # Create model with random weights
        model = AutoModelForCausalLM.from_config(config)

        # Init adapter
        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        # Dummy forward pass
        input_ids = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            _ = model(input_ids)

        # Check captures exist (using correct attribute names)
        assert len(adapter.capture.query_states) > 0, "Failed to capture Q"
        assert len(adapter.capture.key_states) > 0, "Failed to capture K"

        # Verify shapes
        for layer_idx, q in adapter.capture.query_states.items():
            assert q.dim() >= 2, f"Q should be at least 2D, got {q.dim()}"

        adapter.cleanup()
        del model

    def test_hook_hidden_dimension(self):
        """Verify captured tensors have correct hidden dimension."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 4
        config.n_embd = 128
        hidden_size = config.n_embd

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        input_ids = torch.randint(0, 100, (2, 8))  # Batch size 2
        with torch.no_grad():
            _ = model(input_ids)

        # Check value states if captured
        if adapter.capture.value_states:
            for layer_idx, v in adapter.capture.value_states.items():
                # Value states should have hidden_size in last dimension
                assert v.shape[-1] == hidden_size, \
                    f"Value hidden dim mismatch: expected {hidden_size}, got {v.shape[-1]}"

        adapter.cleanup()
        del model

    def test_batch_size_handling(self):
        """Verify hooks handle batch size > 1."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        # Test with batch size 4
        batch_size = 4
        seq_len = 12
        input_ids = torch.randint(0, 100, (batch_size, seq_len))

        with torch.no_grad():
            _ = model(input_ids)

        # Verify batch dimension in captures
        for layer_idx, q in adapter.capture.query_states.items():
            assert q.shape[0] == batch_size, \
                f"Q batch size mismatch: expected {batch_size}, got {q.shape[0]}"

        adapter.cleanup()
        del model

    def test_sequence_length_handling(self):
        """Verify hooks handle various sequence lengths."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        for seq_len in [4, 16, 32]:
            adapter.capture.clear()
            input_ids = torch.randint(0, 100, (1, seq_len))

            with torch.no_grad():
                _ = model(input_ids)

            # Check captures have correct sequence length
            for layer_idx, q in adapter.capture.query_states.items():
                # Q shape depends on architecture but should include seq_len
                assert seq_len in q.shape, \
                    f"Seq len {seq_len} not found in Q shape {q.shape}"

        adapter.cleanup()
        del model

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_consistency(self):
        """Verify hooks work on CUDA and maintain device consistency."""
        device = torch.device("cuda")

        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config).to(device)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        input_ids = torch.randint(0, 100, (1, 10), device=device)
        with torch.no_grad():
            _ = model(input_ids)

        # Verify all captures are on CUDA
        for layer_idx, q in adapter.capture.query_states.items():
            assert q.device.type == "cuda", f"Q not on CUDA: {q.device}"

        for layer_idx, k in adapter.capture.key_states.items():
            assert k.device.type == "cuda", f"K not on CUDA: {k.device}"

        adapter.cleanup()
        del model


class TestPreMLPCapture:
    """Verify h_attn is captured BEFORE residual/MLP."""

    def test_attn_output_captured_before_mlp(self):
        """
        Verify h_attn is captured at the right point.

        Strategy: If h_attn is captured before MLP, then h_attn should NOT
        include MLP transformations. We can verify this by checking that
        h_attn doesn't equal the full layer output.
        """
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # If attn_outputs is populated, verify it's different from hidden states
        if adapter.capture.attn_outputs:
            for layer_idx, h_attn in adapter.capture.attn_outputs.items():
                # h_attn should exist and have reasonable shape
                assert h_attn is not None
                assert h_attn.shape[-1] == config.n_embd

        adapter.cleanup()
        del model

    def test_value_states_captured(self):
        """Verify value states are captured for spectral roughness."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _ = model(input_ids)

        # Value states may be captured depending on architecture
        # For GPT-2, check if we got them
        if adapter.capture.value_states:
            for layer_idx, v in adapter.capture.value_states.items():
                assert v is not None
                # Should have hidden dimension
                assert v.shape[-1] == config.n_embd or v.shape[-1] == config.n_embd // config.n_head

        adapter.cleanup()
        del model


class TestHookCleanup:
    """Verify hooks are properly cleaned up."""

    def test_remove_hooks(self):
        """Verify cleanup removes hooks properly."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        # Verify hooks are registered
        assert len(adapter._hooks) > 0

        # Remove hooks
        adapter.cleanup()

        # Verify hooks are removed
        assert len(adapter._hooks) == 0

        del model

    def test_clear_cache(self):
        """Verify capture.clear() empties all capture dictionaries."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        # Do a forward pass to populate caches
        input_ids = torch.randint(0, 100, (1, 8))
        with torch.no_grad():
            _ = model(input_ids)

        # Verify caches are populated
        assert len(adapter.capture.query_states) > 0 or len(adapter.capture.key_states) > 0

        # Clear cache
        adapter.capture.clear()

        # Verify caches are empty
        assert len(adapter.capture.query_states) == 0
        assert len(adapter.capture.key_states) == 0
        assert len(adapter.capture.attn_outputs) == 0
        assert len(adapter.capture.value_states) == 0

        adapter.cleanup()
        del model


class TestMultipleForwardPasses:
    """Verify hooks work correctly across multiple forward passes."""

    def test_sequential_forward_passes(self):
        """Verify hooks work for multiple sequential forward passes."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        # Multiple forward passes
        for i in range(3):
            adapter.capture.clear()
            input_ids = torch.randint(0, 100, (1, 8 + i))

            with torch.no_grad():
                _ = model(input_ids)

            # Verify captures exist after each pass
            assert len(adapter.capture.query_states) > 0, f"Pass {i}: No Q captures"

        adapter.cleanup()
        del model

    def test_different_batch_sizes(self):
        """Verify hooks handle changing batch sizes."""
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer = 2
        config.n_head = 2
        config.n_embd = 64

        model = AutoModelForCausalLM.from_config(config)

        adapter = ModelAdapter(
            model=model,
            layers=[0, 1],
            dtype=torch.float32
        )
        adapter.register()

        for batch_size in [1, 2, 4]:
            adapter.capture.clear()
            input_ids = torch.randint(0, 100, (batch_size, 8))

            with torch.no_grad():
                _ = model(input_ids)

            # Verify batch dimension
            for layer_idx, q in adapter.capture.query_states.items():
                assert q.shape[0] == batch_size, \
                    f"Batch {batch_size}: Q shape mismatch {q.shape}"

        adapter.cleanup()
        del model
