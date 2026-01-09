#!/usr/bin/env python3
"""
Verify v4 AUROC with Safe Harbor fix.
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ag_sar import AGSAR, AGSARConfig


def main():
    print("Loading model...")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    # Initialize v4 with Semantic Shielding
    config = AGSARConfig(
        version=4,
        aggregation_method="min",
        varentropy_scale=3.0,
        dispersion_scale=0.15,  # Lenient shield, breaks on high D only
    )
    agsar = AGSAR(model, tokenizer, config)

    print("Loading HaluEval QA dataset...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    num_samples = 100
    uncertainties = []
    labels = []  # 1 = hallucination, 0 = fact

    print(f"\nEvaluating {num_samples} samples...")
    for i in range(num_samples):
        sample = dataset[i]
        question = sample["question"]
        answer = sample["answer"]
        is_hall = sample["hallucination"] == "yes"

        prompt = f"Question: {question}\nAnswer:"

        try:
            # Calibrate on prompt for dynamic dispersion threshold
            agsar.calibrate_on_prompt(prompt)

            uncertainty = agsar.compute_uncertainty(prompt, answer)
            uncertainties.append(uncertainty)
            labels.append(1 if is_hall else 0)

            label_str = "HALL" if is_hall else "FACT"
            print(f"[{i+1:3d}/{num_samples}] {label_str:4s} | Uncertainty={uncertainty:.4f}")
        except Exception as e:
            print(f"[{i+1:3d}/{num_samples}] ERROR: {e}")
            continue

    # Compute AUROC
    auroc = roc_auc_score(labels, uncertainties)

    print("\n" + "=" * 60)
    print(f"AG-SAR v4 (Safe Harbor) AUROC: {auroc:.4f}")
    print("=" * 60)

    # Statistics by class
    facts = [u for u, l in zip(uncertainties, labels) if l == 0]
    halls = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"\nUncertainty Statistics:")
    print(f"  Hallucinations: {np.mean(halls):.4f} ± {np.std(halls):.4f} (n={len(halls)})")
    print(f"  Facts:          {np.mean(facts):.4f} ± {np.std(facts):.4f} (n={len(facts)})")
    print(f"  Gap:            {np.mean(halls) - np.mean(facts):.4f}")


if __name__ == "__main__":
    main()
