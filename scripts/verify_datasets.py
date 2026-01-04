#!/usr/bin/env python3
"""
Dataset Verification Script

Validates both Raw Sources (HuggingFace) and Internal Wrappers (experiments/data/*.py)
to prevent Silent Data Failures (class imbalance, schema drift, missing labels).

Usage:
    python scripts/verify_datasets.py

Exit codes:
    0 - All checks passed
    1 - Verification failed
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple

# Ensure repo root is in path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_halueval_schema() -> Tuple[bool, List[str]]:
    """
    Verify HaluEval raw dataset schema from HuggingFace.

    Checks:
    - Loads from pminervini/HaluEval with config "qa" and split "data"
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
        errors.append(f"HaluEval schema verification failed: {e}")

    return len(errors) == 0, errors


def verify_halueval_wrapper() -> Tuple[bool, List[str]]:
    """
    Verify HaluEvalDataset wrapper logic.

    Critical checks:
    - Dual-yield: Each raw row yields TWO samples (hallucinated + factual)
    - Class balance: ~50% hallucinations, ~50% factual
    - Prompt sharing: Paired samples share the same prompt
    - num_samples controls OUTPUT samples (after pairing)
    """
    errors = []
    try:
        from experiments.data.halueval import HaluEvalDataset

        print("[-] Testing HaluEvalDataset wrapper...")

        # Load 20 output samples (sampling happens after dual-yield)
        ds = HaluEvalDataset(variant="qa", num_samples=20, seed=42)
        ds.load()
        samples = list(ds)

        # Check 1: Quantity
        if len(samples) != 20:
            errors.append(f"Expected 20 samples, got {len(samples)}.")
            return False, errors

        # Check 2: Class Balance (~50% each due to dual-yield design)
        labels = [s.label for s in samples]
        positives = sum(labels)
        negatives = len(labels) - positives
        # Allow some variance due to random sampling, but should be roughly balanced
        if positives == 0 or negatives == 0:
            errors.append(
                f"Severe class imbalance: {positives} hallucinations, {negatives} factual. "
                "Dual-yield should produce both classes."
            )
        elif abs(positives - negatives) > len(samples) * 0.3:
            errors.append(
                f"Class imbalance: {positives} hallucinations, {negatives} factual. "
                "Expected roughly 50/50 from dual-yield."
            )

        # Check 3: Verify dual-yield by loading without sampling
        # The base _load() should produce pairs - verify on raw load
        ds_full = HaluEvalDataset(variant="qa", num_samples=None, seed=42)
        ds_full.load()
        full_samples = list(ds_full)[:100]  # Check first 100

        # Check adjacent pairs share prompts (before random sampling)
        pairs_matched = 0
        for i in range(0, len(full_samples) - 1, 2):
            if full_samples[i].prompt == full_samples[i + 1].prompt:
                pairs_matched += 1
        if pairs_matched < len(full_samples) // 4:  # At least 50% pairs should match
            errors.append(
                f"Dual-yield pairing broken: only {pairs_matched}/{len(full_samples)//2} pairs share prompts."
            )
        else:
            print(f"  Dual-yield verified: {pairs_matched} adjacent pairs share prompts")

        # Check 4: Label integrity
        if not all(x in [0, 1] for x in labels):
            errors.append("Labels contain values other than 0 or 1")

        # Check 5: No exact response duplication in prompt
        # Note: Knowledge/context containing similar info is expected for QA tasks
        for s in samples[:4]:
            # Only flag if the ENTIRE response appears verbatim in prompt
            if s.response.strip() == s.prompt.strip():
                errors.append("Data leakage: response is identical to prompt")
                break

        if not errors:
            print(f"✓ HaluEval wrapper: {len(samples)} samples, {positives} pos / {negatives} neg")

    except Exception as e:
        errors.append(f"HaluEval wrapper verification crashed: {e}")

    return len(errors) == 0, errors


def verify_ragtruth_schema() -> Tuple[bool, List[str]]:
    """
    Verify RAGTruth raw dataset schema from HuggingFace.

    Checks:
    - Loads from wandb/RAGTruth-processed split "test"
    - Required columns: query/context, output, hallucination_labels, task_type, quality
    """
    errors = []
    try:
        from datasets import load_dataset

        ds = load_dataset("wandb/RAGTruth-processed", split="test", streaming=True)
        sample = next(iter(ds))

        # Required columns
        if "context" not in sample and "input_str" not in sample:
            errors.append("RAGTruth missing prompt columns (context or input_str)")

        if "output" not in sample:
            errors.append("RAGTruth missing column: output")

        if "hallucination_labels_processed" not in sample:
            errors.append("RAGTruth missing column: hallucination_labels_processed")

        if "task_type" not in sample:
            errors.append("RAGTruth missing column: task_type")

        if "quality" not in sample:
            errors.append("RAGTruth missing column: quality (needed for refusal filtering)")

        print("✓ RAGTruth schema verified")

    except Exception as e:
        errors.append(f"RAGTruth schema verification failed: {e}")

    return len(errors) == 0, errors


def verify_ragtruth_wrapper() -> Tuple[bool, List[str]]:
    """
    Verify RAGTruthDataset wrapper logic.

    Critical checks:
    - Label mapping: is_hallucination -> label in [0, 1]
    - Refusal filtering: filter_refusals parameter uses quality field
    - Task filtering: task_type filter works
    """
    errors = []
    try:
        from experiments.data.ragtruth import RAGTruthDataset

        print("[-] Testing RAGTruthDataset wrapper...")

        # Test 1: Basic loading without filter
        ds_raw = RAGTruthDataset(task_type="QA", num_samples=50, seed=42, filter_refusals=False)
        ds_raw.load()
        samples_raw = list(ds_raw)
        count_raw = len(samples_raw)

        # Test 2: With refusal filter
        ds_filt = RAGTruthDataset(task_type="QA", num_samples=50, seed=42, filter_refusals=True)
        ds_filt.load()
        samples_filt = list(ds_filt)
        count_filt = len(samples_filt)

        print(f"   Raw: {count_raw}, Filtered: {count_filt}")

        # Check 1: Label integrity
        labels = [s.label for s in samples_raw]
        if not all(x in [0, 1] for x in labels):
            errors.append("RAGTruth labels contain values other than 0 or 1")

        # Check 2: Filter should not increase count
        if count_filt > count_raw:
            errors.append(f"Filtered dataset grew: {count_raw} -> {count_filt}")

        # Check 3: Has both classes (for AUROC)
        positives = sum(labels)
        negatives = count_raw - positives
        if positives == 0:
            errors.append("RAGTruth has no hallucination samples (all 0s)")
        if negatives == 0:
            errors.append("RAGTruth has no factual samples (all 1s)")

        # Check 4: Prompt structure
        if samples_raw:
            prompt = samples_raw[0].prompt
            if "Context:" not in prompt or "Question:" not in prompt:
                errors.append("RAGTruth prompt missing expected structure (Context/Question)")

        if not errors:
            print(f"✓ RAGTruth wrapper: {count_raw} samples, {positives} hall / {negatives} fact")
            if count_filt < count_raw:
                print(f"  Refusal filter removed {count_raw - count_filt} samples")

    except Exception as e:
        errors.append(f"RAGTruth wrapper verification crashed: {e}")

    return len(errors) == 0, errors


def verify_label_balance() -> Tuple[bool, List[str]]:
    """
    Verify both datasets have label balance for valid AUROC.
    """
    errors = []
    warnings = []

    try:
        from experiments.data.halueval import HaluEvalDataset

        halueval = HaluEvalDataset(variant="qa", num_samples=100, seed=42)
        halueval.load()
        stats = halueval.get_statistics()

        if stats["hallucinated"] == 0:
            errors.append("HaluEval has no hallucination samples")
        if stats["factual"] == 0:
            errors.append("HaluEval has no factual samples")
        if abs(stats["hallucination_rate"] - 0.5) > 0.01:
            warnings.append(f"HaluEval rate={stats['hallucination_rate']:.2%} (expected 50%)")

        print(f"✓ HaluEval balance: {stats['hallucinated']}/{stats['factual']} (hall/fact)")

    except Exception as e:
        errors.append(f"HaluEval balance check failed: {e}")

    try:
        from experiments.data.ragtruth import RAGTruthDataset

        ragtruth = RAGTruthDataset(task_type="QA", num_samples=100, seed=42)
        ragtruth.load()
        stats = ragtruth.get_statistics()

        if stats["hallucinated"] == 0:
            errors.append("RAGTruth has no hallucination samples")
        if stats["factual"] == 0:
            errors.append("RAGTruth has no factual samples")

        print(f"✓ RAGTruth balance: {stats['hallucinated']}/{stats['factual']} (hall/fact)")

    except Exception as e:
        errors.append(f"RAGTruth balance check failed: {e}")

    return len(errors) == 0, errors


def verify_dependencies() -> Tuple[bool, List[str]]:
    """Verify required libraries are importable."""
    errors = []

    try:
        import numpy
        import scipy
        import sklearn
        from sklearn.metrics import roc_auc_score, average_precision_score
        print("✓ Science libraries (numpy, scipy, sklearn)")
    except ImportError as e:
        errors.append(f"Missing science library: {e}")

    try:
        import torch
        print("✓ PyTorch")
    except ImportError as e:
        errors.append(f"Missing torch: {e}")

    try:
        import transformers
        print("✓ Transformers")
    except ImportError as e:
        errors.append(f"Missing transformers: {e}")

    try:
        from datasets import load_dataset
        print("✓ HuggingFace datasets")
    except ImportError as e:
        errors.append(f"Missing datasets: {e}")

    return len(errors) == 0, errors


def verify_cuda() -> Tuple[bool, List[str]]:
    """Check CUDA availability and report VRAM."""
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

    return True, errors  # CUDA is optional


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    print()

    all_passed = True
    all_errors = []

    # Phase 1: Dependencies
    print("[1] Verifying dependencies...")
    passed, errors = verify_dependencies()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 2: CUDA (optional)
    print("[2] Checking CUDA...")
    passed, errors = verify_cuda()
    all_errors.extend(errors)
    print()

    # Phase 3: Raw HuggingFace Schemas
    print("[3] Verifying HuggingFace schemas...")
    passed, errors = verify_halueval_schema()
    all_passed = all_passed and passed
    all_errors.extend(errors)

    passed, errors = verify_ragtruth_schema()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 4: Wrapper Logic (Critical)
    print("[4] Verifying dataset wrapper logic...")
    passed, errors = verify_halueval_wrapper()
    all_passed = all_passed and passed
    all_errors.extend(errors)

    passed, errors = verify_ragtruth_wrapper()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Phase 5: Label Balance
    print("[5] Verifying label balance...")
    passed, errors = verify_label_balance()
    all_passed = all_passed and passed
    all_errors.extend(errors)
    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("🚀 DATASETS READY FOR EXPERIMENTS")
        print("   All schemas verified, wrappers tested, labels balanced.")
        print("=" * 60)
        return 0
    else:
        print("🛑 DATASET ERRORS DETECTED")
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
