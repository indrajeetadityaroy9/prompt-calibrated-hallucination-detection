"""
Experiment 2: POS-Tag Correlation

Validates that AG-SAR relevance scores correlate with information-carrying
parts of speech (nouns, verbs, adjectives) rather than function words.

Key Hypothesis: If AG-SAR correctly identifies semantically relevant tokens,
relevance should correlate positively with content words.

Success Criteria: Spearman ρ > 0.3 between relevance and content word mask
"""

from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import torch

from ..config import EvalConfig
from ..datasets import load_wikitext
from ..metrics.pos_correlation import (
    compute_pos_correlation,
    analyze_relevance_distribution
)


def run_pos_correlation_experiment(
    ag_sar,
    config: EvalConfig,
    num_samples: int = 200,
    save_results: bool = True
) -> Dict:
    """
    Run POS-tag correlation experiment.

    Measures Spearman rank correlation between AG-SAR relevance scores
    and a binary mask of content words (Noun, Verb, Adj, Adv, PropN).

    Args:
        ag_sar: AGSAR instance
        config: Evaluation configuration
        num_samples: Number of WikiText samples
        save_results: Whether to save results

    Returns:
        Dict with correlation results
    """
    print("=" * 60)
    print("Experiment 2: POS-Tag Correlation")
    print("=" * 60)

    # Load WikiText samples
    print(f"\nLoading WikiText-103 ({num_samples} samples)...")
    samples = load_wikitext(max_samples=num_samples)

    results = {
        'experiment': 'pos_correlation',
        'num_samples': len(samples),
        'correlations': [],
        'p_values': [],
        'pos_mass_breakdown': [],
        'content_mass': [],
        'function_mass': [],
        'punct_mass': []
    }

    print(f"\nProcessing {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

        text = sample.prompt + (sample.response or "")
        if len(text.strip()) < 20:
            continue

        try:
            # Get relevance scores from AG-SAR
            # Use empty prompt, full text as response for single-text analysis
            gse_result = ag_sar.compute_uncertainty("", text, return_details=True)
            relevance = gse_result.get('relevance', None)

            if relevance is None:
                continue

            # Convert to list
            if hasattr(relevance, 'tolist'):
                relevance = relevance.tolist()
            elif hasattr(relevance, 'cpu'):
                relevance = relevance.cpu().numpy().tolist()

            # Flatten if nested (batch dimension)
            if isinstance(relevance, list) and len(relevance) > 0 and isinstance(relevance[0], list):
                relevance = relevance[0]

            # Get tokens
            tokens = ag_sar.tokenizer.tokenize(text)

            # Align lengths
            min_len = min(len(tokens), len(relevance))
            tokens = tokens[:min_len]
            relevance = relevance[:min_len]

            if len(tokens) < 5:
                continue

            # Compute correlation
            corr, p_val = compute_pos_correlation(tokens, relevance)

            if not np.isnan(corr):
                results['correlations'].append(corr)
                results['p_values'].append(p_val)

            # Get full distribution analysis
            dist = analyze_relevance_distribution(tokens, relevance)
            results['content_mass'].append(dist['content_word_mass'])
            results['function_mass'].append(dist['function_word_mass'])
            results['punct_mass'].append(dist['punctuation_mass'])

        except Exception as e:
            print(f"  Warning: Failed on sample {i}: {e}")
            continue

    # Compute summary statistics
    correlations = results['correlations']
    if correlations:
        results['summary'] = {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'median_correlation': float(np.median(correlations)),
            'significant_samples': sum(
                1 for p in results['p_values'] if p < 0.05
            ),
            'total_samples': len(correlations),
            'mean_content_mass': float(np.mean(results['content_mass'])),
            'mean_function_mass': float(np.mean(results['function_mass'])),
            'mean_punct_mass': float(np.mean(results['punct_mass']))
        }
    else:
        results['summary'] = {
            'mean_correlation': 0.0,
            'error': 'No valid correlations computed'
        }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    summary = results['summary']
    print(f"\nSpearman Correlation (Relevance vs Content Words):")
    print(f"  Mean ρ: {summary.get('mean_correlation', 0):.3f}")
    print(f"  Std ρ:  {summary.get('std_correlation', 0):.3f}")
    print(f"  Median ρ: {summary.get('median_correlation', 0):.3f}")

    print(f"\nStatistical Significance (p < 0.05):")
    sig = summary.get('significant_samples', 0)
    total = summary.get('total_samples', 1)
    print(f"  {sig}/{total} samples ({sig/total*100:.1f}%)")

    print(f"\nMass Distribution:")
    print(f"  Content words: {summary.get('mean_content_mass', 0):.1%}")
    print(f"  Function words: {summary.get('mean_function_mass', 0):.1%}")
    print(f"  Punctuation: {summary.get('mean_punct_mass', 0):.1%}")

    # Success check
    mean_corr = summary.get('mean_correlation', 0)
    success = mean_corr > 0.3

    results['success'] = success
    results['success_criteria'] = 'Mean Spearman ρ > 0.3'

    print(f"\n{'=' * 60}")
    print(f"RESULT: {'PASS' if success else 'FAIL'}")
    print(f"Mean correlation: {mean_corr:.3f} (threshold: 0.3)")
    print(f"{'=' * 60}")

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp2_pos_correlation.json'

        json_results = {
            'experiment': results['experiment'],
            'num_samples': results['num_samples'],
            'success': results['success'],
            'success_criteria': results['success_criteria'],
            'summary': results['summary']
        }

        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_correlation_distribution(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot histogram of correlation values across samples.
    """
    import matplotlib.pyplot as plt

    correlations = results.get('correlations', [])

    if not correlations:
        print("No correlation data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Correlation histogram
    ax1 = axes[0]
    ax1.hist(correlations, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=0.3, color='green', linestyle='--', label='Threshold (0.3)')
    ax1.axvline(x=np.mean(correlations), color='red', linestyle='-',
                label=f'Mean ({np.mean(correlations):.3f})')
    ax1.set_xlabel('Spearman Correlation (ρ)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Relevance-Content Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mass distribution pie chart
    ax2 = axes[1]
    summary = results.get('summary', {})
    content_mass = summary.get('mean_content_mass', 0.25)
    function_mass = summary.get('mean_function_mass', 0.25)
    punct_mass = summary.get('mean_punct_mass', 0.25)
    other_mass = max(0, 1 - content_mass - function_mass - punct_mass)

    labels = ['Content Words', 'Function Words', 'Punctuation', 'Other']
    sizes = [content_mass, function_mass, punct_mass, other_mass]
    colors = ['#2ecc71', '#f39c12', '#95a5a6', '#bdc3c7']

    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90)
    ax2.set_title('Average Relevance Mass by Word Type')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_relevance_heatmap(
    ag_sar,
    text: str,
    save_path: Optional[Path] = None
):
    """
    Create heatmap visualization of relevance overlaid on text.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Get relevance - use empty prompt, full text as response
    result = ag_sar.compute_uncertainty("", text, return_details=True)
    relevance = result.get('relevance', [])

    if hasattr(relevance, 'cpu'):
        relevance = relevance.cpu().numpy()
    relevance = np.array(relevance)

    # Flatten batch dimension if present: (1, seq_len) -> (seq_len,)
    if relevance.ndim > 1:
        relevance = relevance.flatten()

    # Get tokens
    tokens = ag_sar.tokenizer.tokenize(text)

    # Align
    min_len = min(len(tokens), len(relevance))
    tokens = tokens[:min_len]
    relevance = relevance[:min_len]

    # Normalize relevance for coloring
    relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-10)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Use a colormap
    cmap = plt.cm.YlOrRd

    # Plot tokens with background colors
    x_pos = 0
    y_pos = 0.5
    line_height = 0.12
    max_width = 12

    for i, (token, rel) in enumerate(zip(tokens, relevance_norm)):
        # Clean token for display
        display_token = token.replace('Ġ', ' ').replace('Ċ', '\n')

        # Get color based on relevance
        color = cmap(rel)

        # Estimate token width
        token_width = len(display_token) * 0.08 + 0.05

        # Check if we need a new line
        if x_pos + token_width > max_width:
            x_pos = 0
            y_pos -= line_height

        # Draw background rectangle
        rect = plt.Rectangle((x_pos, y_pos - 0.04), token_width, 0.08,
                            facecolor=color, edgecolor='none')
        ax.add_patch(rect)

        # Draw token text
        ax.text(x_pos + token_width/2, y_pos, display_token,
               ha='center', va='center', fontsize=8,
               fontfamily='monospace')

        x_pos += token_width

    ax.set_xlim(-0.5, max_width + 0.5)
    ax.set_ylim(y_pos - 0.15, 0.65)
    ax.axis('off')
    ax.set_title('AG-SAR Relevance Heatmap (Yellow=Low, Red=High)', fontsize=12)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.5)
    cbar.set_label('Normalized Relevance')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from ag_sar import AGSAR

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Initialize AG-SAR
    ag_sar = AGSAR(model, tokenizer)

    # Run experiment
    config = EvalConfig()
    results = run_pos_correlation_experiment(
        ag_sar, config, num_samples=200
    )

    # Plot distribution
    plot_correlation_distribution(
        results, config.results_dir / 'exp2_pos_correlation.png'
    )

    # Example heatmap
    sample_text = """The quick brown fox jumps over the lazy dog.
    Scientists discovered a new species of deep-sea fish in the Pacific Ocean."""
    visualize_relevance_heatmap(
        ag_sar, sample_text,
        config.results_dir / 'exp2_heatmap_example.png'
    )
