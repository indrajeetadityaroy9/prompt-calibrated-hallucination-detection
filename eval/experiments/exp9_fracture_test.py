"""
Experiment 9: The Fracture Test - Mechanistic Smoking Gun

TWO DISTINCT TESTS:

1. SCIENCE (Pure Topology): Does the attention graph actually fracture?
   - Metric: Gini coefficient of Relevance distribution
   - Purpose: Prove the MECHANISM works INDEPENDENTLY of probability
   - Expected: Factual → Sharp (high Gini), Hallucination → Diffuse (low Gini)

2. PRODUCT (Graph-Shifted Surprisal): Does the combined metric detect hallucinations?
   - Metric: GSS = Σ Surprisal(t) × Relevance(t)
   - Purpose: State-of-the-art detector combining likelihood + topology
   - Expected: Hallucination → High GSS, Factual → Low GSS

The distinction matters:
- If we only use Surprisal, we prove nothing about the graph (just that model knows Rome is unlikely)
- If we only use Gini, we prove the mechanism but might miss likelihood signal
- The COMBINATION (GSS) is the strongest detector

Success Criteria:
- Pure Topology: Factual Gini > Hallucination Gini (graph fractures on hallucinations)
- GSS Metric: Hallucination GSS > Factual GSS (combined metric detects hallucinations)
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ag_sar import AGSAR, AGSARConfig
from ag_sar.uncertainty import compute_token_surprisal, compute_graph_shifted_surprisal
from transformers import AutoModelForCausalLM, AutoTokenizer


def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute Gini coefficient - measure of inequality/sharpness.

    Range: 0 (perfect equality) to 1 (perfect inequality)
    Higher Gini = more concentrated attention on fewer tokens (sharper focus)
    Lower Gini = more diffuse attention across many tokens (scattered/fractured)
    """
    x = np.abs(x).flatten()
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0

    # Sort values
    sorted_x = np.sort(x)
    n = len(x)

    # Gini formula
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def run_fracture_test(
    model_id: str = "meta-llama/Llama-3.2-3B",
    save_results: bool = True,
    save_plots: bool = True
) -> Dict:
    """
    Run the Fracture Test with TWO analyses:
    1. Pure Topology (Gini) - Proves mechanism
    2. Graph-Shifted Surprisal (GSS) - Strongest detector
    """
    print("="*70)
    print("EXPERIMENT 9: THE FRACTURE TEST")
    print("Mechanistic Proof + Detection Metric")
    print("="*70)

    # Load model
    print(f"\nLoading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    )

    # Initialize AG-SAR
    config = AGSARConfig(use_torch_compile=False)
    ag_sar = AGSAR(model, tokenizer, config)

    # Test pairs: (Prompt, Factual Response, Hallucinated Response)
    test_pairs = [
        ("The capital of France is", " Paris.", " Rome."),
        ("The primary colors are red, yellow, and", " blue.", " green."),
        ("Steve Jobs was the co-founder of", " Apple.", " Microsoft."),
        ("Water boils at", " 100 degrees Celsius.", " 50 degrees Celsius."),
        ("The currency of Japan is the", " Yen.", " Won."),
        ("The chemical symbol for gold is", " Au.", " Ag."),
        ("Shakespeare wrote", " Hamlet.", " Don Quixote."),
        ("The speed of light is approximately", " 300,000 km/s.", " 150,000 km/s."),
        ("The largest planet in our solar system is", " Jupiter.", " Saturn."),
        ("The Great Wall is located in", " China.", " India."),
    ]

    results = []

    print(f"\n{'='*90}")
    print(f"{'Type':<10} | {'Response':<20} | {'Gini':>8} | {'Surprisal':>10} | {'GSS':>10}")
    print(f"{'='*90}")

    for prompt, fact, hall in test_pairs:
        pair_result = {'prompt': prompt}

        for label, response in [('FACT', fact), ('HALLUC', hall)]:
            # Forward pass to get logits and relevance
            text = prompt + response
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask

            # Get AG-SAR details (relevance from attention graph)
            details = ag_sar.compute_uncertainty(prompt, response, return_details=True)
            relevance = details['relevance']  # (1, seq_len)
            response_start = details['response_start']

            # Get logits for surprisal
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()

            # Compute surprisal (NLL of actual tokens)
            surprisal = compute_token_surprisal(logits, input_ids)

            # Extract response-only metrics
            response_relevance = relevance[:, response_start:]
            response_surprisal = surprisal[:, response_start:]

            # Create response mask
            response_mask = torch.zeros_like(attention_mask)
            response_mask[:, response_start:] = 1

            # === METRIC 1: Pure Topology (Gini) ===
            # This proves the graph fractures, independent of probability
            rel_np = response_relevance[0].cpu().numpy()
            gini = gini_coefficient(rel_np)

            # === METRIC 2: Graph-Shifted Surprisal (GSS) ===
            # This is the strongest detection metric
            gss = compute_graph_shifted_surprisal(
                surprisal, relevance, attention_mask=response_mask
            ).item()

            # Average surprisal for comparison
            avg_surprisal = response_surprisal.mean().item()

            # Store results
            pair_result[label.lower()] = {
                'response': response,
                'gini': gini,
                'avg_surprisal': avg_surprisal,
                'gss': gss,
                'relevance': rel_np.tolist(),
                'surprisal': response_surprisal[0].cpu().numpy().tolist()
            }

            print(f"{label:<10} | {response.strip():<20} | {gini:>8.4f} | {avg_surprisal:>10.4f} | {gss:>10.4f}")

        print("-"*90)
        results.append(pair_result)

    ag_sar.cleanup()

    # === ANALYSIS ===
    print("\n" + "="*70)
    print("ANALYSIS: PURE TOPOLOGY (Gini Coefficient)")
    print("="*70)

    # Count topology successes (Fact Gini > Halluc Gini)
    topology_wins = 0
    gini_diffs = []
    for r in results:
        fact_gini = r['fact']['gini']
        hall_gini = r['halluc']['gini']
        diff = fact_gini - hall_gini
        gini_diffs.append(diff)
        if fact_gini > hall_gini:
            topology_wins += 1

    avg_fact_gini = np.mean([r['fact']['gini'] for r in results])
    avg_hall_gini = np.mean([r['halluc']['gini'] for r in results])

    print(f"\nFactual Gini (avg):       {avg_fact_gini:.4f}")
    print(f"Hallucination Gini (avg): {avg_hall_gini:.4f}")
    print(f"Topology Wins:            {topology_wins}/{len(results)}")

    topology_success = topology_wins > len(results) / 2
    print(f"\n{'✅ PASS' if topology_success else '❌ FAIL'}: Graph fractures on hallucinations")

    print("\n" + "="*70)
    print("ANALYSIS: GRAPH-SHIFTED SURPRISAL (GSS)")
    print("="*70)

    # Count GSS successes (Halluc GSS > Fact GSS)
    gss_wins = 0
    gss_diffs = []
    for r in results:
        fact_gss = r['fact']['gss']
        hall_gss = r['halluc']['gss']
        diff = hall_gss - fact_gss
        gss_diffs.append(diff)
        if hall_gss > fact_gss:
            gss_wins += 1

    avg_fact_gss = np.mean([r['fact']['gss'] for r in results])
    avg_hall_gss = np.mean([r['halluc']['gss'] for r in results])

    print(f"\nFactual GSS (avg):       {avg_fact_gss:.4f}")
    print(f"Hallucination GSS (avg): {avg_hall_gss:.4f}")
    print(f"GSS Wins:                {gss_wins}/{len(results)}")

    gss_success = gss_wins > len(results) / 2
    print(f"\n{'✅ PASS' if gss_success else '❌ FAIL'}: GSS detects hallucinations")

    # Compare to pure surprisal
    print("\n" + "="*70)
    print("COMPARISON: GSS vs Pure Surprisal")
    print("="*70)

    surprisal_wins = 0
    for r in results:
        if r['halluc']['avg_surprisal'] > r['fact']['avg_surprisal']:
            surprisal_wins += 1

    print(f"Pure Surprisal Wins: {surprisal_wins}/{len(results)}")
    print(f"GSS Wins:            {gss_wins}/{len(results)}")

    if gss_wins >= surprisal_wins:
        print("\n✅ GSS performs as well or better than pure surprisal")
        print("   The graph topology provides useful signal!")
    else:
        print("\n⚠️  Pure surprisal outperforms GSS on this test")

    # Summary
    summary = {
        'experiment': 'fracture_test_v2',
        'model': model_id,
        'num_pairs': len(results),
        'topology': {
            'avg_fact_gini': float(avg_fact_gini),
            'avg_hall_gini': float(avg_hall_gini),
            'wins': topology_wins,
            'success': bool(topology_success)
        },
        'gss': {
            'avg_fact_gss': float(avg_fact_gss),
            'avg_hall_gss': float(avg_hall_gss),
            'wins': gss_wins,
            'success': bool(gss_success)
        },
        'surprisal_wins': surprisal_wins,
        'pairs': results
    }

    # Save results
    if save_results:
        results_dir = Path(__file__).parent.parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / 'exp9_fracture_test.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {results_dir / 'exp9_fracture_test.json'}")

    # Generate plots
    if save_plots:
        plot_fracture_results(results, results_dir)

    return summary


def plot_fracture_results(results: List[Dict], save_dir: Path):
    """Generate comparative plots for the fracture test."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Gini comparison (Pure Topology)
    ax = axes[0, 0]
    x = np.arange(len(results))
    width = 0.35

    fact_ginis = [r['fact']['gini'] for r in results]
    hall_ginis = [r['halluc']['gini'] for r in results]

    ax.bar(x - width/2, fact_ginis, width, label='Factual', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, hall_ginis, width, label='Hallucination', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Gini Coefficient (Sharpness)')
    ax.set_title('Pure Topology: Attention Graph Sharpness')
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(results))], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: GSS comparison
    ax = axes[0, 1]
    fact_gss = [r['fact']['gss'] for r in results]
    hall_gss = [r['halluc']['gss'] for r in results]

    ax.bar(x - width/2, fact_gss, width, label='Factual', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, hall_gss, width, label='Hallucination', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Graph-Shifted Surprisal')
    ax.set_title('Detection Metric: GSS (Lower = More Confident)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(results))], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Token-level relevance for first example
    ax = axes[1, 0]
    example = results[0]

    tokens_fact = list(range(len(example['fact']['relevance'])))
    tokens_hall = list(range(len(example['halluc']['relevance'])))

    ax.plot(tokens_fact, example['fact']['relevance'], 'b-o',
            label=f"Fact: {example['fact']['response'].strip()}", linewidth=2, markersize=8)
    ax.plot(tokens_hall, example['halluc']['relevance'], 'r-s',
            label=f"Halluc: {example['halluc']['response'].strip()}", linewidth=2, markersize=8)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Relevance')
    ax.set_title(f'Relevance Distribution: "{example["prompt"]}"')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Token-level surprisal for first example
    ax = axes[1, 1]

    ax.plot(tokens_fact, example['fact']['surprisal'], 'b-o',
            label=f"Fact: {example['fact']['response'].strip()}", linewidth=2, markersize=8)
    ax.plot(tokens_hall, example['halluc']['surprisal'], 'r-s',
            label=f"Halluc: {example['halluc']['response'].strip()}", linewidth=2, markersize=8)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Surprisal (NLL)')
    ax.set_title(f'Surprisal Distribution: "{example["prompt"]}"')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'exp9_fracture_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {save_dir / 'exp9_fracture_analysis.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Fracture Test")
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model to use')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')

    args = parser.parse_args()

    results = run_fracture_test(
        model_id=args.model,
        save_results=True,
        save_plots=not args.no_plots
    )
