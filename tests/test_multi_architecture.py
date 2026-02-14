"""
Multi-Architecture Hook Verification Tests.

Tests that DSG hooks work correctly across:
- LLaMA 3.1 8B
- Mistral 7B
- Qwen2 7B
- Gemma-1.1 2B

Each test verifies:
1. Model structure compatibility (layers, norm, lm_head)
2. Hook installation and capture
"""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def get_model_structure(model):
    """Extract standardized model structure info."""
    structure = {}

    # Check for standard transformer structure
    # ForCausalLM -> model -> layers, norm, embed_tokens
    if hasattr(model, 'model'):
        inner = model.model
        structure['has_inner_model'] = True
        structure['layers_attr'] = 'layers' if hasattr(inner, 'layers') else None
        structure['norm_attr'] = 'norm' if hasattr(inner, 'norm') else None
        structure['embed_attr'] = 'embed_tokens' if hasattr(inner, 'embed_tokens') else None

        if structure['layers_attr']:
            layers = getattr(inner, structure['layers_attr'])
            structure['num_layers'] = len(layers)

            # Check layer structure
            if len(layers) > 0:
                layer = layers[0]
                structure['has_post_attention_layernorm'] = hasattr(layer, 'post_attention_layernorm')
                structure['has_input_layernorm'] = hasattr(layer, 'input_layernorm')
                structure['has_mlp'] = hasattr(layer, 'mlp')
                structure['has_self_attn'] = hasattr(layer, 'self_attn')
    else:
        structure['has_inner_model'] = False

    # Check lm_head
    structure['has_lm_head'] = hasattr(model, 'lm_head')

    return structure


def verify_structure_compatibility(model, model_name: str) -> bool:
    """Verify model structure is compatible with AG-SAR hooks."""
    structure = get_model_structure(model)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Required structure
    required = [
        ('has_inner_model', True),
        ('layers_attr', 'layers'),
        ('norm_attr', 'norm'),
        ('has_lm_head', True),
        ('has_post_attention_layernorm', True),
    ]

    all_pass = True
    for key, expected in required:
        actual = structure.get(key)
        status = "✓" if actual == expected else "✗"
        if actual != expected:
            all_pass = False
        print(f"  {status} {key}: {actual} (expected: {expected})")

    print(f"  Num layers: {structure.get('num_layers', 'N/A')}")

    return all_pass


def check_hook_installation(model, model_name: str) -> bool:
    """Test that hooks can be installed and capture hidden states."""
    from ag_sar.hooks import EphemeralHiddenBuffer, LayerHooks

    print(f"\nTesting hook installation on {model_name}...")

    try:
        # Create buffer and hook
        buffer = EphemeralHiddenBuffer()
        layer_idx = len(model.model.layers) - 1  # Last layer
        layer = model.model.layers[layer_idx]

        hook = LayerHooks(layer_idx, buffer)
        hook.install(layer)

        # Run a forward pass
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Simple input
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)

        with torch.no_grad():
            _ = model(input_ids, use_cache=False)

        # Check buffer was populated
        states = buffer.get_states()

        if layer_idx not in states:
            print(f"  ✗ Layer {layer_idx} not in captured states")
            hook.remove()
            return False

        layer_state = states[layer_idx]

        # Verify all three capture points
        checks = [
            ('h_resid_attn', layer_state.h_resid_attn),
            ('h_mlp_in', layer_state.h_mlp_in),
            ('h_resid_mlp', layer_state.h_resid_mlp),
        ]

        all_pass = True
        for name, tensor in checks:
            if tensor is None:
                print(f"  ✗ {name} is None")
                all_pass = False
            elif tensor.shape[-1] != model.config.hidden_size:
                print(f"  ✗ {name} shape mismatch: {tensor.shape}")
                all_pass = False
            else:
                print(f"  ✓ {name}: {tensor.shape}, dtype={tensor.dtype}")

        # Cleanup
        hook.remove()
        buffer.clear()

        return all_pass

    except Exception as e:
        print(f"  ✗ Hook installation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_checks(model_id: str, token: str = None):
    """Run all tests on a single model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os

    print(f"\n{'#'*70}")
    print(f"# Testing: {model_id}")
    print(f"{'#'*70}")

    # Get token
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            token_file = os.path.expanduser("~/.huggingface/token")
            if os.path.exists(token_file):
                with open(token_file) as f:
                    token = f.read().strip()

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=True)

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token,
            trust_remote_code=True,
        )

        # Add pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        results = {}

        # Test 1: Structure compatibility
        results['structure'] = verify_structure_compatibility(model, model_id)

        # Test 2: Hook installation
        results['hooks'] = check_hook_installation(model, model_id)

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY: {model_id}")
        print(f"{'='*60}")
        all_pass = True
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {test_name}")
            if not passed:
                all_pass = False

        # Cleanup
        del model
        torch.cuda.empty_cache()

        return all_pass

    except Exception as e:
        print(f"Failed to load model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests on all supported models."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-architecture hook verification")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to test (default: all)")
    parser.add_argument("--token", type=str, default=None,
                       help="HuggingFace token for gated models")
    args = parser.parse_args()

    # Models to test
    all_models = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen2": "Qwen/Qwen2-7B-Instruct",
        "gemma": "google/gemma-1.1-2b-it",
    }

    if args.models:
        models_to_test = {k: v for k, v in all_models.items() if k in args.models}
    else:
        models_to_test = all_models

    results = {}
    for name, model_id in models_to_test.items():
        try:
            results[name] = run_model_checks(model_id, args.token)
        except Exception as e:
            print(f"Failed to test {name}: {e}")
            results[name] = False

    # Final summary
    print(f"\n{'#'*70}")
    print("# FINAL SUMMARY")
    print(f"{'#'*70}")

    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
