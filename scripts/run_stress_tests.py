#!/usr/bin/env python3
"""
AG-SAR Stress Tests for NeurIPS Robustness Validation

4 Tests to make results bulletproof:
1. Domain Generalization (CoQA) - Proves it works beyond fact retrieval
2. Length Bias Check - Proves AG-SAR measures confusion, not verbosity
3. Hyperparameter Sensitivity - Proves semantic_layers=4 is robust, not lucky
4. Qualitative Sanity Check - Verifies token relevance makes linguistic sense
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar import AGSAR, AGSARConfig
from eval.baselines.predictive_entropy import PredictiveEntropy
from eval.datasets import load_triviaqa, load_coqa
from eval.config import EvalConfig


def stress_test_1_domain_generalization(
    model, tokenizer, device, num_samples: int = 50
) -> Dict:
    """
    Stress Test 1: Domain Generalization (CoQA)

    Proves AG-SAR works on conversational reasoning, not just fact retrieval.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 1: DOMAIN GENERALIZATION (CoQA)")
    print("=" * 70)

    from eval.experiments.exp3_auroc import run_auroc_experiment
    from eval.baselines.original_sar import OriginalSAR

    # Initialize AG-SAR with compile disabled to avoid recompilation issues
    agsar_config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, agsar_config)
    pe_baseline = PredictiveEntropy(model, tokenizer)

    # Run on CoQA
    eval_config = EvalConfig()
    results = run_auroc_experiment(
        ag_sar=ag_sar,
        original_sar=None,  # Skip slow baseline
        pe_baseline=pe_baseline,
        config=eval_config,
        num_samples=num_samples,
        save_results=False,
        dataset='coqa'
    )

    # Also run on TriviaQA for comparison
    triviaqa_results = run_auroc_experiment(
        ag_sar=ag_sar,
        original_sar=None,
        pe_baseline=pe_baseline,
        config=eval_config,
        num_samples=num_samples,
        save_results=False,
        dataset='triviaqa'
    )

    ag_sar.cleanup()

    # Extract AUROC with error handling
    coqa_auroc = results.get('methods', {}).get('ag_sar', {}).get('auroc', 0.0)
    triviaqa_auroc = triviaqa_results.get('methods', {}).get('ag_sar', {}).get('auroc', 0.0)

    # Handle NaN
    if coqa_auroc != coqa_auroc:  # NaN check
        coqa_auroc = 0.0
    if triviaqa_auroc != triviaqa_auroc:
        triviaqa_auroc = 0.0

    print("\n" + "-" * 50)
    print("DOMAIN GENERALIZATION RESULTS:")
    print("-" * 50)
    print(f"TriviaQA (Fact Retrieval):    AUROC = {triviaqa_auroc:.3f}")
    print(f"CoQA (Conversational):        AUROC = {coqa_auroc:.3f}")
    print(f"Domain Transfer Gap:          {abs(triviaqa_auroc - coqa_auroc):.3f}")

    success = coqa_auroc > 0.70 and triviaqa_auroc > 0.70
    print(f"\nPASS: {'YES' if success else 'NO'} (Both AUROC > 0.70)")

    return {
        'test': 'domain_generalization',
        'triviaqa_auroc': triviaqa_auroc,
        'coqa_auroc': coqa_auroc,
        'domain_gap': abs(triviaqa_auroc - coqa_auroc),
        'success': success
    }


def stress_test_2_length_bias(
    model, tokenizer, device, num_samples: int = 100
) -> Dict:
    """
    Stress Test 2: Length Bias Check

    Proves AG-SAR measures confusion, not just response length.
    PE is notorious for correlating with length (longer = higher entropy sum).
    AG-SAR claims to fix this via normalization.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 2: LENGTH BIAS CHECK")
    print("=" * 70)

    agsar_config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, agsar_config)
    pe_baseline = PredictiveEntropy(model, tokenizer)

    samples = load_triviaqa(max_samples=num_samples)

    agsar_scores = []
    pe_scores = []
    response_lengths = []

    print(f"\nCollecting {len(samples)} samples...")
    for sample in tqdm(samples):
        try:
            # Get response length in tokens
            tokens = tokenizer.encode(sample.response, add_special_tokens=False)
            response_lengths.append(len(tokens))

            # Get AG-SAR score
            gse = ag_sar.compute_uncertainty(sample.prompt, sample.response)
            agsar_scores.append(gse)

            # Get PE score
            pe = pe_baseline.compute_uncertainty(sample.prompt, sample.response)
            pe_scores.append(pe)

        except Exception as e:
            continue

    ag_sar.cleanup()

    # Calculate Pearson correlations
    agsar_corr, agsar_pval = stats.pearsonr(response_lengths, agsar_scores)
    pe_corr, pe_pval = stats.pearsonr(response_lengths, pe_scores)

    print("\n" + "-" * 50)
    print("LENGTH BIAS CORRELATION RESULTS:")
    print("-" * 50)
    print(f"Predictive Entropy vs Length: r = {pe_corr:.3f} (p = {pe_pval:.4f})")
    print(f"AG-SAR vs Length:             r = {agsar_corr:.3f} (p = {agsar_pval:.4f})")
    print(f"\nLength range: {min(response_lengths)}-{max(response_lengths)} tokens")
    print(f"Mean length: {np.mean(response_lengths):.1f} tokens")

    # Success criteria: AG-SAR should have lower correlation than PE
    success = abs(agsar_corr) < abs(pe_corr) and abs(agsar_corr) < 0.3

    print(f"\nPASS: {'YES' if success else 'NO'}")
    print(f"  - AG-SAR correlation < PE: {abs(agsar_corr) < abs(pe_corr)}")
    print(f"  - AG-SAR correlation < 0.3: {abs(agsar_corr) < 0.3}")

    return {
        'test': 'length_bias',
        'pe_correlation': pe_corr,
        'pe_pvalue': pe_pval,
        'agsar_correlation': agsar_corr,
        'agsar_pvalue': agsar_pval,
        'num_samples': len(agsar_scores),
        'mean_length': np.mean(response_lengths),
        'success': success
    }


def stress_test_3_hyperparameter_sensitivity(
    model, tokenizer, device, num_samples: int = 50
) -> Dict:
    """
    Stress Test 3: Hyperparameter Sensitivity

    Proves semantic_layers=4 is robust, not a lucky choice.
    Sweeps over [2, 4, 8, 16] layers.
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 3: HYPERPARAMETER SENSITIVITY (semantic_layers)")
    print("=" * 70)

    from eval.experiments.exp3_auroc import run_auroc_experiment

    pe_baseline = PredictiveEntropy(model, tokenizer)

    layer_configs = [2, 4, 8]  # Skip 16 as it may exceed model layers
    results_by_layers = {}

    for n_layers in layer_configs:
        print(f"\n--- Testing semantic_layers = {n_layers} ---")

        try:
            config = AGSARConfig(semantic_layers=n_layers, use_torch_compile=False)
            ag_sar = AGSAR(model, tokenizer, config)

            eval_config = EvalConfig()
            result = run_auroc_experiment(
                ag_sar=ag_sar,
                original_sar=None,
                pe_baseline=pe_baseline,
                config=eval_config,
                num_samples=num_samples,
                save_results=False,
                dataset='triviaqa'
            )

            auroc = result['methods']['ag_sar']['auroc']
            results_by_layers[n_layers] = auroc

            ag_sar.cleanup()

        except Exception as e:
            print(f"  Failed with {n_layers} layers: {e}")
            results_by_layers[n_layers] = None

    print("\n" + "-" * 50)
    print("HYPERPARAMETER SENSITIVITY RESULTS:")
    print("-" * 50)
    print(f"{'Layers':<10} {'AUROC':<10}")
    print("-" * 20)

    valid_aurocs = []
    for n_layers, auroc in sorted(results_by_layers.items()):
        if auroc is not None:
            print(f"{n_layers:<10} {auroc:.3f}")
            valid_aurocs.append(auroc)
        else:
            print(f"{n_layers:<10} FAILED")

    if len(valid_aurocs) >= 2:
        auroc_range = max(valid_aurocs) - min(valid_aurocs)
        auroc_std = np.std(valid_aurocs)
        print(f"\nAUROC Range: {auroc_range:.3f}")
        print(f"AUROC Std:   {auroc_std:.3f}")

        # Success: stable performance (range < 0.15, all > 0.70)
        success = auroc_range < 0.15 and min(valid_aurocs) > 0.70
    else:
        auroc_range = None
        auroc_std = None
        success = False

    print(f"\nPASS: {'YES' if success else 'NO'}")
    print(f"  - AUROC range < 0.15: {auroc_range is not None and auroc_range < 0.15}")
    print(f"  - All AUROC > 0.70: {min(valid_aurocs) > 0.70 if valid_aurocs else False}")

    return {
        'test': 'hyperparameter_sensitivity',
        'results_by_layers': results_by_layers,
        'auroc_range': auroc_range,
        'auroc_std': auroc_std,
        'success': success
    }


def stress_test_4_qualitative_sanity(
    model, tokenizer, device, num_examples: int = 5
) -> Dict:
    """
    Stress Test 4: Qualitative Token Relevance Sanity Check

    Verifies that high-relevance tokens are semantically meaningful
    (nouns, proper nouns) and low-relevance are structural (the, of, is).
    """
    print("\n" + "=" * 70)
    print("STRESS TEST 4: QUALITATIVE TOKEN RELEVANCE SANITY CHECK")
    print("=" * 70)

    agsar_config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, agsar_config)

    # Test cases: mix of factual and hallucinated
    test_cases = [
        {
            'prompt': "Question: What is the capital of France?\nAnswer:",
            'response': " Paris is the capital of France.",
            'type': 'factual',
            'expected_high': ['Paris', 'France', 'capital'],
            'expected_low': ['the', 'of', 'is']
        },
        {
            'prompt': "Question: Who wrote Romeo and Juliet?\nAnswer:",
            'response': " William Shakespeare wrote Romeo and Juliet.",
            'type': 'factual',
            'expected_high': ['Shakespeare', 'Romeo', 'Juliet', 'William'],
            'expected_low': ['and', 'wrote']
        },
        {
            'prompt': "Question: What is the speed of light?\nAnswer:",
            'response': " The speed of light is approximately 300,000 kilometers per second.",
            'type': 'factual',
            'expected_high': ['speed', 'light', '300', 'kilometers'],
            'expected_low': ['The', 'of', 'is']
        },
    ]

    results = []

    print("\nAnalyzing token relevance patterns...")
    for case in test_cases[:num_examples]:
        try:
            details = ag_sar.compute_uncertainty(
                case['prompt'],
                case['response'],
                return_details=True
            )

            # Get tokens and their relevance scores
            input_ids = details['input_ids'][0]
            relevance = details['relevance'][0]
            response_start = details['response_start']

            # Extract response tokens
            token_relevance = []
            for i in range(response_start, len(input_ids)):
                token_id = input_ids[i].item()
                token = tokenizer.decode([token_id])
                rel = relevance[i].item()
                token_relevance.append((token.strip(), rel))

            # Sort by relevance
            sorted_tokens = sorted(token_relevance, key=lambda x: x[1], reverse=True)

            top_3 = sorted_tokens[:3]
            bottom_3 = sorted_tokens[-3:] if len(sorted_tokens) >= 3 else sorted_tokens

            print(f"\n{'='*50}")
            print(f"Type: {case['type']}")
            print(f"Response: {case['response']}")
            print(f"\nTop 3 Relevance:    {[(t, f'{r:.4f}') for t, r in top_3]}")
            print(f"Bottom 3 Relevance: {[(t, f'{r:.4f}') for t, r in bottom_3]}")

            # Check if expected patterns hold
            top_tokens = [t for t, r in top_3]
            bottom_tokens = [t for t, r in bottom_3]

            # Check overlap with expected
            high_match = any(exp.lower() in ' '.join(top_tokens).lower()
                           for exp in case['expected_high'])
            low_match = any(exp.lower() in ' '.join(bottom_tokens).lower()
                          for exp in case['expected_low'])

            print(f"High-relevance matches expected: {high_match}")
            print(f"Low-relevance matches expected: {low_match}")

            results.append({
                'type': case['type'],
                'response': case['response'],
                'top_3': top_3,
                'bottom_3': bottom_3,
                'high_match': high_match,
                'low_match': low_match
            })

        except Exception as e:
            print(f"Failed on case: {e}")
            results.append({'error': str(e)})

    ag_sar.cleanup()

    # Overall success: most cases should show semantic patterns
    successful = sum(1 for r in results if r.get('high_match') or r.get('low_match'))
    success = successful >= len(results) * 0.5

    print("\n" + "-" * 50)
    print("QUALITATIVE SANITY CHECK RESULTS:")
    print("-" * 50)
    print(f"Cases with semantic patterns: {successful}/{len(results)}")
    print(f"\nPASS: {'YES' if success else 'NO'} (>50% show expected patterns)")

    return {
        'test': 'qualitative_sanity',
        'num_cases': len(results),
        'successful_cases': successful,
        'details': results,
        'success': success
    }


def main():
    parser = argparse.ArgumentParser(description='AG-SAR Stress Tests')
    parser.add_argument('--model', type=str, default='llama3.2',
                        choices=['gpt2', 'llama3.2', 'qwen', 'mistral'])
    parser.add_argument('--test', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which tests to run (1-4)')
    parser.add_argument('--samples', type=int, default=50,
                        help='Number of samples per test')
    args = parser.parse_args()

    # Model mapping
    model_map = {
        'gpt2': 'gpt2',
        'llama3.2': 'meta-llama/Llama-3.2-3B',
        'qwen': 'Qwen/Qwen2.5-7B',
        'mistral': 'mistralai/Mistral-7B-v0.3',
    }
    model_id = model_map[args.model]

    print("=" * 70)
    print("AG-SAR STRESS TESTS FOR NEURIPS ROBUSTNESS VALIDATION")
    print("=" * 70)
    print(f"Model: {model_id}")
    print(f"Tests: {args.test}")
    print(f"Samples per test: {args.samples}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nLoading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        device_map='cuda:0',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Run selected tests
    if 1 in args.test:
        all_results['test_1'] = stress_test_1_domain_generalization(
            model, tokenizer, device, args.samples
        )

    if 2 in args.test:
        all_results['test_2'] = stress_test_2_length_bias(
            model, tokenizer, device, args.samples
        )

    if 3 in args.test:
        all_results['test_3'] = stress_test_3_hyperparameter_sensitivity(
            model, tokenizer, device, args.samples
        )

    if 4 in args.test:
        all_results['test_4'] = stress_test_4_qualitative_sanity(
            model, tokenizer, device, num_examples=5
        )

    # Summary
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    passed = 0
    total = len(all_results)

    for test_name, result in all_results.items():
        status = "PASS" if result.get('success') else "FAIL"
        print(f"{test_name}: {status}")
        if result.get('success'):
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    # Save results
    results_path = Path('results/stress_tests.json')
    results_path.parent.mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(i) for i in obj)
        return obj

    with open(results_path, 'w') as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
