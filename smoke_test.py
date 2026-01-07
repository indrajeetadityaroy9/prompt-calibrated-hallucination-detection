#!/usr/bin/env python3
"""
AG-SAR Smoke Test - Verify installation and basic functionality.

This script validates that AG-SAR is properly installed and can:
1. Import all required modules
2. Load data modules
3. Create AGSARConfig with validation
4. Run basic inference (if GPU available)

Usage:
    python smoke_test.py
    python smoke_test.py --skip-inference  # Skip GPU inference test
"""

import argparse
import sys
from pathlib import Path


def check_imports():
    """Check that all core imports work."""
    print("[1/5] Checking core imports...")

    errors = []

    # Core AG-SAR
    try:
        from ag_sar import AGSAR, AGSARConfig
        print("  [OK] ag_sar.AGSAR, ag_sar.AGSARConfig")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar core: {e}")

    # Utilities
    try:
        from ag_sar import enable_h100_optimizations, get_optimal_dtype
        print("  [OK] ag_sar.enable_h100_optimizations, ag_sar.get_optimal_dtype")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar utilities: {e}")

    # Presets
    try:
        from ag_sar.presets import load_preset, get_available_presets
        presets = get_available_presets()
        print(f"  [OK] ag_sar.presets ({len(presets)} presets available)")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar.presets: {e}")

    # Measures
    try:
        from ag_sar.measures import compute_authority_score, compute_semantic_dispersion
        print("  [OK] ag_sar.measures")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar.measures: {e}")

    # Ops
    try:
        from ag_sar.ops import compute_authority_flow, _TRITON_AVAILABLE
        backend = "Triton" if _TRITON_AVAILABLE else "PyTorch"
        print(f"  [OK] ag_sar.ops (backend: {backend})")
    except ImportError as e:
        errors.append(f"  [FAIL] ag_sar.ops: {e}")

    return errors


def check_experiments_imports():
    """Check experiments framework imports."""
    print("\n[2/5] Checking experiments framework...")

    errors = []

    # Evaluation engine
    try:
        from experiments.evaluation import BenchmarkEngine
        print("  [OK] experiments.evaluation.BenchmarkEngine")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.evaluation: {e}")

    # Data modules
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

    # Methods
    try:
        from experiments.methods.base import UncertaintyMethod, MethodResult
        from experiments.methods.agsar_wrapper import AGSARMethod
        print("  [OK] experiments.methods")
    except ImportError as e:
        errors.append(f"  [FAIL] experiments.methods: {e}")

    # Config schema
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

    # Valid config
    try:
        config = AGSARConfig()
        print("  [OK] Default config creates successfully")
    except Exception as e:
        errors.append(f"  [FAIL] Default config: {e}")
        return errors

    # Invalid parameter tests
    invalid_tests = [
        {"residual_weight": 1.5, "error": "residual_weight"},
        {"dispersion_k": 0, "error": "dispersion_k"},
        {"parametric_weight": -0.1, "error": "parametric_weight"},
        {"nucleus_top_p": 0, "error": "nucleus_top_p"},
        {"stability_sensitivity": 0, "error": "stability_sensitivity"},
    ]

    for test in invalid_tests:
        error_field = test.pop("error")
        try:
            AGSARConfig(**test)
            errors.append(f"  [FAIL] Should reject invalid {error_field}")
        except ValueError:
            print(f"  [OK] Correctly rejects invalid {error_field}")

    return errors


def check_dependencies():
    """Check key dependencies are available."""
    print("\n[4/5] Checking dependencies...")

    errors = []

    # PyTorch
    try:
        import torch
        cuda_status = f"CUDA {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_status += f" ({torch.cuda.get_device_name(0)})"
        print(f"  [OK] PyTorch {torch.__version__} ({cuda_status})")
    except ImportError as e:
        errors.append(f"  [FAIL] torch: {e}")
        return errors  # Can't continue without torch

    # Transformers
    try:
        import transformers
        version = transformers.__version__
        major, minor = int(version.split(".")[0]), int(version.split(".")[1])
        if major == 4 and 40 <= minor < 45:
            print(f"  [OK] transformers {version} (compatible)")
        else:
            print(f"  [WARN] transformers {version} (may be incompatible, need 4.40-4.44)")
    except ImportError as e:
        errors.append(f"  [FAIL] transformers: {e}")

    # NumPy
    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError as e:
        errors.append(f"  [FAIL] numpy: {e}")

    # YAML
    try:
        import yaml
        print(f"  [OK] PyYAML")
    except ImportError as e:
        errors.append(f"  [FAIL] PyYAML: {e}")

    # Datasets (optional, for experiments)
    try:
        import datasets
        print(f"  [OK] datasets {datasets.__version__}")
    except ImportError:
        print("  [SKIP] datasets (optional, needed for experiments)")

    # Sentence Transformers (optional, for baselines)
    try:
        import sentence_transformers
        print(f"  [OK] sentence-transformers")
    except ImportError:
        print("  [SKIP] sentence-transformers (optional, for SelfCheck/SemanticEntropy)")

    return errors


def check_inference(skip: bool = False):
    """Run a quick inference test with GPT-2."""
    print("\n[5/5] Checking inference...")

    if skip:
        print("  [SKIP] Inference test skipped (--skip-inference)")
        return []

    errors = []

    try:
        import torch
        if not torch.cuda.is_available():
            print("  [SKIP] No CUDA available, skipping GPU inference test")
            return []
    except ImportError:
        return ["  [FAIL] torch not available"]

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ag_sar import AGSAR, AGSARConfig

        print("  Loading GPT-2 (small, ~500MB)...")

        # Load tiny model for quick test
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create AG-SAR engine
        config = AGSARConfig(
            semantic_layers=2,  # Use fewer layers for speed
            power_iteration_steps=2,
        )
        agsar = AGSAR(model, tokenizer, config)

        # Test inference
        prompt = "The capital of France is"
        response = " Paris."

        print(f"  Prompt: '{prompt}'")
        print(f"  Response: '{response}'")

        result = agsar.compute_uncertainty(prompt, response, return_details=True)

        print(f"  [OK] Uncertainty score: {result['score']:.4f}")
        print(f"  [OK] Latency: {result.get('latency_ms', 0):.1f}ms")

        # Cleanup
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
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip GPU inference test",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("AG-SAR Smoke Test")
    print("=" * 60)

    all_errors = []

    # Run all checks
    all_errors.extend(check_imports())
    all_errors.extend(check_experiments_imports())
    all_errors.extend(check_config_validation())
    all_errors.extend(check_dependencies())
    all_errors.extend(check_inference(skip=args.skip_inference))

    # Summary
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
        print("AG-SAR is correctly installed and functional.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
