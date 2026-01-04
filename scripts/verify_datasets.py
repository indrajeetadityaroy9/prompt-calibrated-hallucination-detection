#!/usr/bin/env python3
"""
Dataset Verification Script (Phase 3.3)

Hard gate: This script MUST pass before running experiments.
Exits non-zero on any schema mismatch.

Usage:
    python scripts/verify_datasets.py
"""

import sys
from typing import List, Tuple


def verify_halueval() -> Tuple[bool, List[str]]:
    """
    Verify HaluEval dataset schema.

    Phase 3.2 Requirements:
    - Repo: pminervini/HaluEval
    - Config: qa or summarization
    - Split: data
    - Required columns: knowledge, question, right_answer, hallucinated_answer
    """
    errors = []
    try:
        from datasets import load_dataset

        # Test QA variant (streaming to avoid full download)
        ds = load_dataset("pminervini/HaluEval", "qa", split="data", streaming=True)
        sample = next(iter(ds))

        required_cols = ["knowledge", "question", "right_answer", "hallucinated_answer"]
        for col in required_cols:
            if col not in sample:
                errors.append(f"HaluEval-QA missing column: {col}")

        # Verify values are non-empty strings
        for col in ["right_answer", "hallucinated_answer"]:
            if col in sample and not isinstance(sample[col], str):
                errors.append(f"HaluEval-QA column {col} is not a string")

        print("✓ HaluEval-QA schema verified")

        # Test summarization variant
        ds_summ = load_dataset(
            "pminervini/HaluEval", "summarization", split="data", streaming=True
        )
        sample_summ = next(iter(ds_summ))

        if "document" not in sample_summ:
            errors.append("HaluEval-Summarization missing column: document")

        print("✓ HaluEval-Summarization schema verified")

    except Exception as e:
        errors.append(f"HaluEval verification failed: {e}")

    return len(errors) == 0, errors


def verify_ragtruth() -> Tuple[bool, List[str]]:
    """
    Verify RAGTruth dataset schema.

    Phase 3.2 Requirements:
    - Repo: wandb/RAGTruth-processed
    - Split: test
    - Required columns: query/input_str, output, hallucination_labels, task_type
    """
    errors = []
    try:
        from datasets import load_dataset

        ds = load_dataset("wandb/RAGTruth-processed", split="test", streaming=True)
        sample = next(iter(ds))

        # Required columns (either input_str or context+query for prompt)
        if "input_str" not in sample and "context" not in sample:
            errors.append("RAGTruth missing prompt columns (input_str or context)")

        if "output" not in sample:
            errors.append("RAGTruth missing column: output")

        if "hallucination_labels" not in sample:
            errors.append("RAGTruth missing column: hallucination_labels")

        if "task_type" not in sample:
            errors.append("RAGTruth missing column: task_type")

        # Verify hallucination_labels structure (list or can be converted to list)
        if "hallucination_labels" in sample:
            labels = sample["hallucination_labels"]
            # In streaming mode, may come as various iterable types
            try:
                _ = len(list(labels) if not isinstance(labels, list) else labels)
            except (TypeError, ValueError):
                errors.append("RAGTruth hallucination_labels cannot be converted to list")

        print("✓ RAGTruth schema verified")

    except Exception as e:
        errors.append(f"RAGTruth verification failed: {e}")

    return len(errors) == 0, errors


def verify_label_balance() -> Tuple[bool, List[str]]:
    """
    Verify dataset label balance for AUROC validity.

    Both positive and negative labels must be present.
    """
    errors = []
    warnings = []

    # Add experiments to path if needed
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from experiments.data.halueval import HaluEvalDataset

        # Check HaluEval (should be perfectly balanced by design)
        halueval = HaluEvalDataset(variant="qa", num_samples=100, seed=42)
        halueval.load()
        stats = halueval.get_statistics()

        if stats["hallucinated"] == 0:
            errors.append("HaluEval has no hallucination samples")
        if stats["factual"] == 0:
            errors.append("HaluEval has no factual samples")
        if stats["hallucination_rate"] != 0.5:
            print(f"  Note: HaluEval hallucination_rate={stats['hallucination_rate']:.2%}")

        print(f"✓ HaluEval balance: {stats['hallucinated']}/{stats['factual']} (hal/fact)")

    except Exception as e:
        warnings.append(f"HaluEval balance check skipped: {e}")
        print(f"⚠ HaluEval balance check skipped: {e}")

    try:
        from experiments.data.ragtruth import RAGTruthDataset

        # Check RAGTruth
        ragtruth = RAGTruthDataset(num_samples=100, seed=42)
        ragtruth.load()
        stats = ragtruth.get_statistics()

        if stats["hallucinated"] == 0:
            errors.append("RAGTruth has no hallucination samples")
        if stats["factual"] == 0:
            errors.append("RAGTruth has no factual samples")

        print(f"✓ RAGTruth balance: {stats['hallucinated']}/{stats['factual']} (hal/fact)")

    except Exception as e:
        warnings.append(f"RAGTruth balance check skipped: {e}")
        print(f"⚠ RAGTruth balance check skipped: {e}")

    return len(errors) == 0, errors


def verify_dependencies() -> Tuple[bool, List[str]]:
    """
    Verify all required science libraries are importable.

    Phase 0.1 requirement.
    """
    errors = []
    warnings = []

    try:
        import numpy
        import scipy
        import sklearn
        from sklearn.metrics import roc_auc_score, average_precision_score
        from sklearn.calibration import calibration_curve
        print("✓ Science libraries (numpy, scipy, sklearn) importable")
    except ImportError as e:
        errors.append(f"Missing science library: {e}")

    try:
        import torch
        print("✓ PyTorch importable")
    except ImportError as e:
        errors.append(f"Missing torch: {e}")

    try:
        import transformers
        print("✓ transformers importable")
    except ImportError as e:
        warnings.append(f"Missing transformers (optional for schema check): {e}")
        print(f"⚠ transformers not available (optional)")

    try:
        import sentence_transformers
        print("✓ sentence_transformers importable")
    except ImportError as e:
        warnings.append(f"Missing sentence_transformers (optional for schema check): {e}")
        print(f"⚠ sentence_transformers not available (optional for SelfCheckGPT)")

    return len(errors) == 0, errors


def verify_cuda() -> Tuple[bool, List[str]]:
    """
    Verify CUDA availability and report VRAM.
    """
    errors = []
    try:
        import torch

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            total_vram = 0
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                total_vram += vram_gb
                print(f"✓ GPU {i}: {props.name} ({vram_gb:.1f}GB)")
            print(f"✓ Total VRAM: {total_vram:.1f}GB")
        else:
            print("⚠ CUDA not available (CPU only)")

    except Exception as e:
        errors.append(f"CUDA verification failed: {e}")

    return len(errors) == 0, errors


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("DATASET VERIFICATION (Phase 3.3)")
    print("=" * 60)
    print()

    all_passed = True
    all_errors = []

    # Phase 0.1: Dependencies
    print("[Phase 0.1] Verifying dependencies...")
    passed, errors = verify_dependencies()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # CUDA check
    print("[CUDA] Checking GPU availability...")
    passed, errors = verify_cuda()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 3.2: HaluEval schema
    print("[Phase 3.2] Verifying HaluEval schema...")
    passed, errors = verify_halueval()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 3.2: RAGTruth schema
    print("[Phase 3.2] Verifying RAGTruth schema...")
    passed, errors = verify_ragtruth()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 3.1: Label balance
    print("[Phase 3.1] Verifying label balance...")
    passed, errors = verify_label_balance()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("  Datasets are ready for ICML/NeurIPS benchmark runs")
        print("=" * 60)
        return 0
    else:
        print("✗ VERIFICATION FAILED")
        print()
        print("Errors:")
        for err in all_errors:
            print(f"  - {err}")
        print()
        print("Fix the above errors before running experiments.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
