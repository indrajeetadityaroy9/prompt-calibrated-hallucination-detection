#!/usr/bin/env python3
"""
AG-SAR v19 Faithfulness Validation - Extrinsic Faithfulness Stress Tests

Tests v19 on datasets that isolate Context Adherence from World Knowledge:

1. FaithEval-counterfactual: Context contradicts world knowledge
   - Tests JSD (Deception) stream: Can model suppress parametric memory?
   - Expected: High JSD when model overrides context

2. FaithEval-unanswerable: Context doesn't contain the answer
   - Tests V×D (Confusion) stream: Does model fabricate or abstain?
   - Expected: High Varentropy when model fabricates

3. RAGBench: Pre-labeled adherence scores
   - Tests both streams: Full pipeline validation
   - Expected: Uncertainty correlates with non-adherence

References:
- FaithEval: https://huggingface.co/datasets/Salesforce/FaithEval-counterfactual-v1.0
- RAGBench: https://huggingface.co/datasets/rungalileo/ragbench
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ag_sar import AGSAR, AGSARConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_faitheval_counterfactual(agsar, tokenizer, model, num_samples=100):
    """
    FaithEval Counterfactual: Context contains statements that contradict world knowledge.

    The Test: Can the model suppress its parametric memory to follow the context?

    Hypothesis:
    - Model Failure: Model says factual answer (ignores counterfactual context)
    - AG-SAR v19 Signal: FFN fights attention to force parametric answer → JSD spikes

    This validates the Mechanistic Grounding (JSD/Deception) stream.
    """
    print(f"\n{'=' * 70}")
    print("FaithEval Counterfactual: Parametric Override Detection")
    print("=" * 70)
    print("Hypothesis: JSD should spike when model overrides counterfactual context")
    print("")

    dataset = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test")

    uncertainties = []
    labels = []
    jsd_values = []
    vd_values = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Build prompt with counterfactual context
        context = sample['context']
        question = sample['question']
        choices = sample['choices']
        correct_key = sample['answerKey']  # The context-faithful answer

        # Format choices
        choice_text = "\n".join([f"{l}: {t}" for l, t in zip(choices['label'], choices['text'])])

        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {question}

Choices:
{choice_text}

Answer with the letter of the correct choice:"""

        # Generate model response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Determine if model followed context (faithful) or overrode it (hallucination)
        response_letter = response.strip().upper()[:1]
        is_faithful = response_letter == correct_key

        # For this test: Faithful = 0 (good), Override = 1 (hallucination)
        label = 0 if is_faithful else 1

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, response, return_details=True)

            uncertainty = details.get('uncertainty', 0)
            r_dec = details.get('R_deception', 0)
            r_conf = details.get('R_confusion', 0)

            uncertainties.append(uncertainty)
            labels.append(label)
            jsd_values.append(r_dec)
            vd_values.append(r_conf)

            if i < 5:
                print(f"  Sample {i}: faithful={is_faithful}, pred={response_letter}, correct={correct_key}")
                print(f"    JSD={r_dec:.4f}, V×D={r_conf:.4f}, unc={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print("  Error: Not enough samples of both classes")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)

    # Separate by label for analysis
    faithful_jsd = [j for j, l in zip(jsd_values, labels) if l == 0]
    override_jsd = [j for j, l in zip(jsd_values, labels) if l == 1]
    faithful_vd = [v for v, l in zip(vd_values, labels) if l == 0]
    override_vd = [v for v, l in zip(vd_values, labels) if l == 1]

    print(f"\n  Results:")
    print(f"  Samples: {len(uncertainties)} ({sum(labels)} overrides, {len(labels) - sum(labels)} faithful)")
    print(f"  AUROC: {auroc:.4f}")

    print(f"\n  Signal Analysis (JSD should be higher for overrides):")
    print(f"    Faithful JSD: {np.mean(faithful_jsd):.4f} +/- {np.std(faithful_jsd):.4f}")
    print(f"    Override JSD: {np.mean(override_jsd):.4f} +/- {np.std(override_jsd):.4f}")
    print(f"    JSD Gap: {np.mean(override_jsd) - np.mean(faithful_jsd):.4f}")

    print(f"\n  Signal Analysis (V×D):")
    print(f"    Faithful V×D: {np.mean(faithful_vd):.4f} +/- {np.std(faithful_vd):.4f}")
    print(f"    Override V×D: {np.mean(override_vd):.4f} +/- {np.std(override_vd):.4f}")

    # Which stream discriminates better?
    auroc_jsd = roc_auc_score(labels, jsd_values)
    auroc_vd = roc_auc_score(labels, vd_values)
    print(f"\n  Stream AUROC:")
    print(f"    JSD alone: {auroc_jsd:.4f}")
    print(f"    V×D alone: {auroc_vd:.4f}")
    print(f"    Dominant stream: {'JSD (Deception)' if auroc_jsd > auroc_vd else 'V×D (Confusion)'}")

    return auroc, {
        'auroc_jsd': auroc_jsd,
        'auroc_vd': auroc_vd,
        'jsd_gap': np.mean(override_jsd) - np.mean(faithful_jsd),
    }


def test_faitheval_unanswerable(agsar, tokenizer, model, num_samples=100):
    """
    FaithEval Unanswerable: Context doesn't contain the answer to the question.

    The Test: Does the model say "I don't know" (Faithful) or fabricate (Hallucination)?

    Hypothesis:
    - Model Failure: Model fabricates an answer
    - AG-SAR v19 Signal: Model is epistemically unstable → Varentropy spikes

    This validates the Epistemic Stability (V×D/Confusion) stream.
    """
    print(f"\n{'=' * 70}")
    print("FaithEval Unanswerable: Fabrication Detection")
    print("=" * 70)
    print("Hypothesis: V×D should spike when model fabricates from unanswerable context")
    print("")

    dataset = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test")

    uncertainties = []
    labels = []
    jsd_values = []
    vd_values = []

    # Keywords that indicate abstention
    abstention_keywords = ['unanswerable', 'unknown', 'not available', 'no answer',
                          'cannot be determined', 'not mentioned', 'not provided',
                          "don't know", "doesn't say", 'unclear', 'not stated']

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        context = sample['context']
        question = sample['question']

        prompt = f"""Based only on the following context, answer the question. If the answer cannot be determined from the context, say "The answer is not available in the context."

Context: {context}

Question: {question}

Answer:"""

        # Generate model response
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Determine if model abstained (faithful) or fabricated (hallucination)
        response_lower = response.lower()
        is_abstention = any(kw in response_lower for kw in abstention_keywords)

        # Faithful abstention = 0, Fabrication = 1
        label = 0 if is_abstention else 1

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, response, return_details=True)

            uncertainty = details.get('uncertainty', 0)
            r_dec = details.get('R_deception', 0)
            r_conf = details.get('R_confusion', 0)

            uncertainties.append(uncertainty)
            labels.append(label)
            jsd_values.append(r_dec)
            vd_values.append(r_conf)

            if i < 5:
                print(f"  Sample {i}: abstained={is_abstention}")
                print(f"    Response: {response[:80]}...")
                print(f"    JSD={r_dec:.4f}, V×D={r_conf:.4f}, unc={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class found ({sum(labels)} fabrications, {len(labels) - sum(labels)} abstentions)")
        if len(labels) > 0:
            print(f"  Mean uncertainty: {np.mean(uncertainties):.4f}")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)

    # Separate by label
    abstain_jsd = [j for j, l in zip(jsd_values, labels) if l == 0]
    fabricate_jsd = [j for j, l in zip(jsd_values, labels) if l == 1]
    abstain_vd = [v for v, l in zip(vd_values, labels) if l == 0]
    fabricate_vd = [v for v, l in zip(vd_values, labels) if l == 1]

    print(f"\n  Results:")
    print(f"  Samples: {len(uncertainties)} ({sum(labels)} fabrications, {len(labels) - sum(labels)} abstentions)")
    print(f"  AUROC: {auroc:.4f}")

    if abstain_vd and fabricate_vd:
        print(f"\n  Signal Analysis (V×D should be higher for fabrications):")
        print(f"    Abstain V×D: {np.mean(abstain_vd):.4f} +/- {np.std(abstain_vd):.4f}")
        print(f"    Fabricate V×D: {np.mean(fabricate_vd):.4f} +/- {np.std(fabricate_vd):.4f}")
        print(f"    V×D Gap: {np.mean(fabricate_vd) - np.mean(abstain_vd):.4f}")

        print(f"\n  Signal Analysis (JSD):")
        print(f"    Abstain JSD: {np.mean(abstain_jsd):.4f} +/- {np.std(abstain_jsd):.4f}")
        print(f"    Fabricate JSD: {np.mean(fabricate_jsd):.4f} +/- {np.std(fabricate_jsd):.4f}")

        # Which stream discriminates better?
        auroc_jsd = roc_auc_score(labels, jsd_values)
        auroc_vd = roc_auc_score(labels, vd_values)
        print(f"\n  Stream AUROC:")
        print(f"    JSD alone: {auroc_jsd:.4f}")
        print(f"    V×D alone: {auroc_vd:.4f}")
        print(f"    Dominant stream: {'JSD (Deception)' if auroc_jsd > auroc_vd else 'V×D (Confusion)'}")

        return auroc, {'auroc_jsd': auroc_jsd, 'auroc_vd': auroc_vd}

    return auroc, {}


def test_ragbench(agsar, num_samples=100):
    """
    RAGBench: Pre-labeled adherence scores for RAG responses.

    Uses existing responses with ground-truth adherence labels.
    Tests full v19 pipeline on realistic RAG outputs.
    """
    print(f"\n{'=' * 70}")
    print("RAGBench: RAG Adherence Detection")
    print("=" * 70)
    print("Testing on pre-labeled RAG responses with adherence scores")
    print("")

    # Use covidqa config
    dataset = load_dataset("rungalileo/ragbench", "covidqa", split="test")

    uncertainties = []
    labels = []
    jsd_values = []
    vd_values = []

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Build prompt from documents
        docs = sample['documents']
        context = "\n\n".join(docs[:3])  # Use first 3 docs
        question = sample['question']
        response = sample['response']

        # Ground truth: adherence_score (boolean)
        is_adherent = sample['adherence_score']
        label = 0 if is_adherent else 1  # Non-adherent = hallucination = 1

        prompt = f"""Based on the following documents, answer the question.

Documents:
{context[:2000]}

Question: {question}

Answer:"""

        try:
            agsar.calibrate_on_prompt(prompt)
            details = agsar.compute_uncertainty(prompt, " " + response, return_details=True)

            uncertainty = details.get('uncertainty', 0)
            r_dec = details.get('R_deception', 0)
            r_conf = details.get('R_confusion', 0)

            uncertainties.append(uncertainty)
            labels.append(label)
            jsd_values.append(r_dec)
            vd_values.append(r_conf)

            if i < 3:
                print(f"  Sample {i}: adherent={is_adherent}")
                print(f"    JSD={r_dec:.4f}, V×D={r_conf:.4f}, unc={uncertainty:.4f}")

        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            continue

    if len(set(labels)) < 2:
        print(f"  Warning: Only one class found")
        return 0.5, {}

    auroc = roc_auc_score(labels, uncertainties)

    # Analysis
    adherent_unc = [u for u, l in zip(uncertainties, labels) if l == 0]
    nonadherent_unc = [u for u, l in zip(uncertainties, labels) if l == 1]

    print(f"\n  Results:")
    print(f"  Samples: {len(uncertainties)} ({sum(labels)} non-adherent, {len(labels) - sum(labels)} adherent)")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Adherent uncertainty: {np.mean(adherent_unc):.4f} +/- {np.std(adherent_unc):.4f}")
    print(f"  Non-adherent uncertainty: {np.mean(nonadherent_unc):.4f} +/- {np.std(nonadherent_unc):.4f}")
    print(f"  Gap: {np.mean(nonadherent_unc) - np.mean(adherent_unc):.4f}")

    # Stream analysis
    auroc_jsd = roc_auc_score(labels, jsd_values)
    auroc_vd = roc_auc_score(labels, vd_values)
    print(f"\n  Stream AUROC:")
    print(f"    JSD alone: {auroc_jsd:.4f}")
    print(f"    V×D alone: {auroc_vd:.4f}")

    return auroc, {'auroc_jsd': auroc_jsd, 'auroc_vd': auroc_vd}


def main():
    print("=" * 70)
    print("AG-SAR v19 Faithfulness Validation")
    print("Extrinsic Faithfulness Stress Tests")
    print("=" * 70)

    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )

    config = AGSARConfig(version=19)
    agsar = AGSAR(model, tokenizer, config=config)

    print(f"\n{'#' * 70}")
    print(f"# VERSION: v19 (Hinge-Risk Architecture)")
    print(f"{'#' * 70}")

    results = {}

    # Test 1: FaithEval Counterfactual (JSD/Deception test)
    auroc_cf, cf_details = test_faitheval_counterfactual(agsar, tokenizer, model, num_samples=50)
    results['faitheval_counterfactual'] = auroc_cf

    # Test 2: FaithEval Unanswerable (V×D/Confusion test)
    auroc_ua, ua_details = test_faitheval_unanswerable(agsar, tokenizer, model, num_samples=50)
    results['faitheval_unanswerable'] = auroc_ua

    # Test 3: RAGBench (Full pipeline test)
    auroc_rb, rb_details = test_ragbench(agsar, num_samples=50)
    results['ragbench'] = auroc_rb

    # Summary
    print(f"\n{'=' * 70}")
    print("FAITHFULNESS VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  FaithEval Counterfactual (JSD test):  AUROC = {results['faitheval_counterfactual']:.4f}")
    print(f"  FaithEval Unanswerable (V×D test):    AUROC = {results['faitheval_unanswerable']:.4f}")
    print(f"  RAGBench Adherence (Full pipeline):   AUROC = {results['ragbench']:.4f}")
    print(f"  Average:                              AUROC = {np.mean(list(results.values())):.4f}")

    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)
    print("""
  FaithEval Counterfactual:
    - High AUROC = JSD detects when FFN overrides counterfactual context
    - This proves v19 catches "Parametric Override" (Holy Grail of RAG safety)

  FaithEval Unanswerable:
    - High AUROC = V×D detects when model fabricates from empty context
    - This proves v19 catches "Epistemic Instability"

  RAGBench:
    - High AUROC = Full pipeline correlates with human adherence judgments
    - This validates v19 on realistic RAG outputs

  Combined with Core Benchmarks:
    | Dataset                  | v19 AUROC | Signal Used        |
    |--------------------------|-----------|-------------------|
    | HaluEval QA              | 0.84      | V×D (Confusion)   |
    | HaluEval Summ            | 0.61      | V×D (Confusion)   |
    | RAGTruth QA              | 0.72      | JSD (Deception)   |
    | RAGTruth Summ            | 0.56      | JSD (Deception)   |
    | FaithEval Counterfactual | {:.2f}      | JSD (Override)    |
    | FaithEval Unanswerable   | {:.2f}      | V×D (Fabrication) |
    | RAGBench                 | {:.2f}      | Both Streams      |
""".format(results['faitheval_counterfactual'],
           results['faitheval_unanswerable'],
           results['ragbench']))


if __name__ == "__main__":
    main()
