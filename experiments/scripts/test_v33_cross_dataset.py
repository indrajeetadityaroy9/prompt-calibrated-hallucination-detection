#!/usr/bin/env python3
"""
Test v3.3 Relativistic Epiplexity on HaluEval and RAGTruth for cross-dataset AUROC.

Expected: AUROC > 0.7 on both datasets with the same parameters.
Theory: E^λ crushes hallucination scores via complexity ratio.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from experiments.preflight import check_installation
check_installation()

from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig


def test_dataset(agsar, dataset, dataset_name):
    """Evaluate AG-SAR v3.3 on a dataset and return AUROC."""
    uncertainties = []
    labels = []

    for sample in tqdm(dataset, desc=f"Testing {dataset_name}"):
        try:
            # v3.3 REQUIRES calibration for complexity ratio
            agsar.calibrate_on_prompt(sample.prompt)
            uncertainty = agsar.compute_uncertainty(sample.prompt, sample.response)
            uncertainties.append(uncertainty)
            labels.append(sample.label)
        except Exception as e:
            print(f"  Error on sample: {e}")
            continue

    if len(uncertainties) == 0:
        return None, 0

    auroc = roc_auc_score(labels, uncertainties)
    return auroc, len(uncertainties)


def main():
    print("=" * 70)
    print("AG-SAR v3.3 Relativistic Epiplexity - Cross-Dataset Validation")
    print("=" * 70)

    # Load model
    model_name = "meta-llama/Llama-3.1-8B"
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize AG-SAR v3.3
    config = AGSARConfig()
    print(f"\nConfig: version={config.version}, lambda_struct={config.lambda_struct}")
    print(f"        entropy_floor={config.entropy_floor}, aggregation={config.aggregation_method}")
    print(f"\nMaster Equation: Score = A × (1-D) × E^{config.lambda_struct}")
    print(f"Epiplexity: E = Clamp(V_response / μ_prompt, 0, 2)")

    agsar = AGSAR(model, tokenizer, config)

    # Test on HaluEval QA
    print("\n" + "=" * 70)
    print("Dataset 1: HaluEval QA")
    print("=" * 70)

    from experiments.data import HaluEvalDataset
    halueval = HaluEvalDataset(num_samples=100, seed=42, variant="qa")
    halueval.load()

    halueval_auroc, halueval_n = test_dataset(agsar, halueval, "HaluEval QA")
    print(f"\nHaluEval QA AUROC: {halueval_auroc:.4f} (n={halueval_n})")

    # Test on RAGTruth QA
    print("\n" + "=" * 70)
    print("Dataset 2: RAGTruth QA")
    print("=" * 70)

    from experiments.data import RAGTruthDataset
    ragtruth = RAGTruthDataset(num_samples=100, seed=42, task_type="QA")
    ragtruth.load()

    ragtruth_auroc, ragtruth_n = test_dataset(agsar, ragtruth, "RAGTruth QA")
    print(f"\nRAGTruth QA AUROC: {ragtruth_auroc:.4f} (n={ragtruth_n})")

    # Summary
    print("\n" + "=" * 70)
    print("v3.3 RELATIVISTIC EPIPLEXITY RESULTS")
    print("=" * 70)
    print(f"\n  HaluEval QA:   AUROC = {halueval_auroc:.4f}")
    print(f"  RAGTruth QA:   AUROC = {ragtruth_auroc:.4f}")

    target_auroc = 0.7
    halueval_pass = halueval_auroc >= target_auroc
    ragtruth_pass = ragtruth_auroc >= target_auroc

    print(f"\n  Target: AUROC >= {target_auroc}")
    print(f"  HaluEval:  {'PASS' if halueval_pass else 'FAIL'}")
    print(f"  RAGTruth:  {'PASS' if ragtruth_pass else 'FAIL'}")

    if halueval_pass and ragtruth_pass:
        print("\n  v3.3 Relativistic Epiplexity VALIDATED for cross-dataset generalization!")
    else:
        print("\n  INVESTIGATION NEEDED: v3.3 did not meet target on all datasets")

    # Cleanup
    agsar.cleanup()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
