#!/usr/bin/env python3
"""
AG-SAR Smoke Test - Verify installation and basic functionality.

Usage:
    python smoke_test.py
    python smoke_test.py --skip-inference
"""

import argparse
import sys


def check_imports():
    """Check that all core imports work."""
    print("[1/5] Checking core imports...")
    errors = []

    try:
        from ag_sar import AGSAR, AGSARConfig
        print("  [OK] ag_sar.AGSAR, ag_sar.AGSARConfig")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar core: {e}")

    try:
        from ag_sar.ops import compute_authority_flow_vectorized, _TRITON_AVAILABLE
        backend = "Triton" if _TRITON_AVAILABLE else "PyTorch"
        print(f"  [OK] ag_sar.ops (backend: {backend})")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar.ops: {e}")

    return errors


def check_experiments_imports():
    """Check experiments framework imports."""
    print("\n[2/5] Checking experiments framework...")
    errors = []

    try:
        from experiments.evaluation import BenchmarkEngine
        print("  [OK] experiments.evaluation.BenchmarkEngine")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.evaluation: {e}")

    try:
        from experiments.data import (
            HaluEvalDataset,
            RAGTruthDataset,
            TruthfulQADataset,
            WikiTextDataset,
            FAVADataset,
        )
        print("  [OK] experiments.data (5 dataset loaders)")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.data: {e}")

    try:
        from experiments.methods.base import UncertaintyMethod
        from experiments.methods.agsar_wrapper import AGSARMethod
        print("  [OK] experiments.methods")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.methods: {e}")

    try:
        from experiments.configs.schema import ExperimentConfig
        print("  [OK] experiments.configs.schema")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.configs.schema: {e}")

    return errors


def check_config_validation():
    """Check that AGSARConfig validates parameters correctly."""
    print("\n[3/5] Checking config validation...")
    errors = []

    from ag_sar import AGSARConfig

    try:
        config = AGSARConfig()
        print("  [OK] Default config creates successfully")
    except Exception as e:
        errors.append(f"  [FAIL] Default config: {e}")
        return errors

    # Test invalid parameters
    invalid_tests = [
        {"semantic_layers": 0, "field": "semantic_layers"},
        {"varentropy_lambda": -1, "field": "varentropy_lambda"},
        {"hallucination_threshold": 1.5, "field": "hallucination_threshold"},
        {"calibration_window": 0, "field": "calibration_window"},
    ]

    for test in invalid_tests:
        field = test.pop("field")
        try:
            AGSARConfig(**test)
            errors.append(f"  [FAIL] Should reject invalid {field}")
        except ValueError:
            print(f"  [OK] Correctly rejects invalid {field}")

    return errors


def check_dependencies():
    """Check key dependencies are available."""
    print("\n[4/5] Checking dependencies...")
    errors = []

    try:
        import torch
        cuda_status = f"CUDA {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_status += f" ({torch.cuda.get_device_name(0)})"
        print(f"  [OK] PyTorch {torch.__version__} ({cuda_status})")
    except ImportError as e:
        errors.append(f"  [FAIL] torch: {e}")
        return errors

    try:
        import transformers
        print(f"  [OK] transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"  [FAIL] transformers: {e}")

    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"  [FAIL] numpy: {e}")

    try:
        import yaml
        print("  [OK] PyYAML")
    except ImportError as e:
        errors.append(f"  [FAIL] PyYAML: {e}")

    return errors


def check_inference(skip: bool = False):
    """Run a quick inference test with GPT-2."""
    print("\n[5/5] Checking inference...")

    if skip:
        print("  [SKIP] Inference test skipped")
        return []

    errors = []

    try:
        import torch
        if not torch.cuda.is_available():
            print("  [SKIP] No CUDA available")
            return []
    except ImportError:
        return ["  [FAIL] torch not available"]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ag_sar import AGSAR, AGSARConfig

        print("  Loading GPT-2...")

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager",
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AGSARConfig(semantic_layers=2)
        agsar = AGSAR(model, tokenizer, config)

        prompt = "The capital of France is"
        response = " Paris."

        print(f"  Prompt: '{prompt}'")
        print(f"  Response: '{response}'")

        result = agsar.compute_uncertainty(prompt, response, return_details=True)

        print(f"  [OK] Uncertainty score: {result['score']:.4f}")

        agsar.cleanup()
        del model
        torch.cuda.empty_cache()

    except Exception as e:
        errors.append(f"  [FAIL] Inference test: {e}")
        import traceback
        traceback.print_exc()

    return errors


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Smoke Test")
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("AG-SAR Smoke Test")
    print("=" * 60)

    all_errors = []
    all_errors.extend(check_imports())
    all_errors.extend(check_experiments_imports())
    all_errors.extend(check_config_validation())
    all_errors.extend(check_dependencies())
    all_errors.extend(check_inference(skip=args.skip_inference))

    print("\n" + "=" * 60)
    if all_errors:
        print(f"SMOKE TEST FAILED ({len(all_errors)} errors)")
        print("=" * 60)
        for error in all_errors:
            print(error)
        return 1
    else:
        print("SMOKE TEST PASSED")
        print("=" * 60)
        return 0


if __name__ == "__main__":
    sys.exit(main())
