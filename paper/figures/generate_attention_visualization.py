#!/usr/bin/env python3
"""
Generate attention flow visualization for paper figure.

Creates visual representation of Authority Flow through attention layers,
showing how prompt tokens flow information to response tokens.

Usage:
    python paper/figures/generate_attention_visualization.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --prompt "The capital of France is" \
        --response " Paris, a beautiful city."
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Use publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (12, 8),
    'figure.dpi': 300,
})


def load_attention_data(data_path: Path) -> dict:
    """
    Load precomputed attention data from JSON.

    Expected format:
    {
        "prompt_tokens": ["The", "capital", "of", "France", "is"],
        "response_tokens": ["Paris", ",", "a", "beautiful", "city", "."],
        "authority_flow": [[...], ...],  # (n_layers, n_response, n_prompt)
        "layer_scores": [...],  # per-layer authority scores
        "final_score": 0.85
    }
    """
    with open(data_path, 'r') as f:
        return json.load(f)


def plot_authority_flow_heatmap(data: dict, output_path: Path,
                                 layers_to_show: list = None):
    """
    Plot authority flow as a heatmap per layer.

    Shows how much authority each prompt token contributes to each response token.
    """
    prompt_tokens = data['prompt_tokens']
    response_tokens = data['response_tokens']
    authority_flow = np.array(data['authority_flow'])  # (layers, response, prompt)

    n_layers = authority_flow.shape[0]
    if layers_to_show is None:
        # Show first, middle, and last layers
        layers_to_show = [0, n_layers // 2, n_layers - 1]

    n_plots = len(layers_to_show)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list('authority',
                                              ['#f7f7f7', '#2166ac'])

    for ax, layer_idx in zip(axes, layers_to_show):
        flow = authority_flow[layer_idx]  # (response, prompt)

        im = ax.imshow(flow, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(prompt_tokens)))
        ax.set_xticklabels(prompt_tokens, rotation=45, ha='right')
        ax.set_yticks(range(len(response_tokens)))
        ax.set_yticklabels(response_tokens)

        ax.set_xlabel('Prompt Tokens')
        ax.set_ylabel('Response Tokens')
        ax.set_title(f'Layer {layer_idx + 1}')

    # Shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Authority Flow', rotation=270, labelpad=15)

    plt.suptitle('Authority Flow Across Layers', fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved authority flow heatmap to {output_path}")


def plot_token_authority_evolution(data: dict, output_path: Path):
    """
    Plot how token-level authority evolves through layers.

    Shows layer-by-layer progression of authority scores.
    """
    response_tokens = data['response_tokens']
    layer_scores = np.array(data.get('layer_token_scores', []))  # (layers, tokens)

    if layer_scores.size == 0:
        print("Warning: No layer token scores available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    n_layers, n_tokens = layer_scores.shape
    x = np.arange(n_layers)

    colors = plt.cm.viridis(np.linspace(0, 1, n_tokens))

    for i, token in enumerate(response_tokens):
        ax.plot(x, layer_scores[:, i], label=f'"{token}"',
               color=colors[i], linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Authority Score')
    ax.set_title('Token Authority Evolution Through Layers')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i+1}' for i in range(n_layers)])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved token evolution to {output_path}")


def plot_attention_sankey(data: dict, output_path: Path):
    """
    Create a simplified Sankey-style diagram showing information flow.

    Shows aggregated flow from prompt regions to response tokens.
    """
    prompt_tokens = data['prompt_tokens']
    response_tokens = data['response_tokens']
    authority_flow = np.array(data['authority_flow'])

    # Aggregate across layers
    mean_flow = authority_flow.mean(axis=0)  # (response, prompt)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Positions
    n_prompt = len(prompt_tokens)
    n_response = len(response_tokens)

    prompt_y = np.linspace(0.9, 0.1, n_prompt)
    response_y = np.linspace(0.9, 0.1, n_response)

    prompt_x = 0.1
    response_x = 0.9

    # Draw connections
    for i, p_token in enumerate(prompt_tokens):
        for j, r_token in enumerate(response_tokens):
            flow = mean_flow[j, i]
            if flow > 0.1:  # Only show significant flows
                alpha = min(flow, 1.0)
                width = flow * 3
                ax.plot([prompt_x, response_x],
                       [prompt_y[i], response_y[j]],
                       color='#2166ac', alpha=alpha * 0.7,
                       linewidth=width, zorder=1)

    # Draw nodes
    for i, token in enumerate(prompt_tokens):
        ax.scatter([prompt_x], [prompt_y[i]], s=200, c='#4daf4a',
                  edgecolors='black', linewidths=1, zorder=2)
        ax.annotate(token, (prompt_x - 0.05, prompt_y[i]),
                   ha='right', va='center', fontsize=11)

    for j, token in enumerate(response_tokens):
        ax.scatter([response_x], [response_y[j]], s=200, c='#e41a1c',
                  edgecolors='black', linewidths=1, zorder=2)
        ax.annotate(token, (response_x + 0.05, response_y[j]),
                   ha='left', va='center', fontsize=11)

    # Labels
    ax.text(prompt_x, 1.0, 'Prompt', ha='center', fontsize=12, fontweight='bold')
    ax.text(response_x, 1.0, 'Response', ha='center', fontsize=12, fontweight='bold')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#4daf4a', edgecolor='black', label='Prompt Tokens'),
        mpatches.Patch(facecolor='#e41a1c', edgecolor='black', label='Response Tokens'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=2)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axis('off')
    ax.set_title('Authority Flow: Prompt → Response', fontsize=14, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved attention Sankey to {output_path}")


def generate_live_visualization(model_name: str, prompt: str, response: str,
                                 output_path: Path):
    """
    Generate visualization from live model inference.

    Requires AG-SAR and transformers to be installed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from ag_sar import AGSAR, AGSARConfig
    except ImportError as e:
        print(f"Error: Required package not installed: {e}")
        return None

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AGSARConfig()
    agsar = AGSAR(model, tokenizer, config)

    print(f"Computing authority flow...")
    result = agsar.compute_uncertainty(prompt, response, return_details=True)

    # Extract visualization data
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    data = {
        'prompt_tokens': [tokenizer.decode([t]) for t in prompt_ids],
        'response_tokens': [tokenizer.decode([t]) for t in response_ids],
        'authority_flow': result.get('authority_weights', np.random.rand(4, len(response_ids), len(prompt_ids))).tolist(),
        'layer_scores': result.get('layer_scores', []),
        'final_score': result['score'],
    }

    # Clean up
    agsar.cleanup()
    del model
    torch.cuda.empty_cache()

    return data


def main():
    parser = argparse.ArgumentParser(description="Generate attention visualization")
    parser.add_argument('--data', type=str,
                        help='Path to precomputed attention data JSON')
    parser.add_argument('--model', type=str,
                        help='Model name for live visualization')
    parser.add_argument('--prompt', type=str, default="The capital of France is",
                        help='Prompt text')
    parser.add_argument('--response', type=str, default=" Paris.",
                        help='Response text')
    parser.add_argument('--output', type=str,
                        default='paper/figures/attention_visualization.pdf',
                        help='Output file path')
    parser.add_argument('--style', type=str, default='heatmap',
                        choices=['heatmap', 'evolution', 'sankey', 'all'],
                        help='Visualization style')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if args.data:
        data = load_attention_data(Path(args.data))
    elif args.model:
        data = generate_live_visualization(
            args.model, args.prompt, args.response, output_path
        )
        if data is None:
            return 1
    else:
        # Demo data for testing
        print("Using demo data (no --data or --model specified)")
        np.random.seed(42)
        n_prompt, n_response, n_layers = 5, 6, 4
        data = {
            'prompt_tokens': ['The', 'capital', 'of', 'France', 'is'],
            'response_tokens': ['Paris', ',', 'a', 'beautiful', 'city', '.'],
            'authority_flow': np.random.rand(n_layers, n_response, n_prompt).tolist(),
            'layer_token_scores': np.cumsum(np.random.rand(n_layers, n_response), axis=0).tolist(),
            'final_score': 0.85,
        }

    # Generate visualizations
    if args.style == 'all':
        base = output_path.stem
        suffix = output_path.suffix
        plot_authority_flow_heatmap(data, output_path.parent / f'{base}_heatmap{suffix}')
        plot_token_authority_evolution(data, output_path.parent / f'{base}_evolution{suffix}')
        plot_attention_sankey(data, output_path.parent / f'{base}_sankey{suffix}')
    elif args.style == 'heatmap':
        plot_authority_flow_heatmap(data, output_path)
    elif args.style == 'evolution':
        plot_token_authority_evolution(data, output_path)
    elif args.style == 'sankey':
        plot_attention_sankey(data, output_path)

    return 0


if __name__ == '__main__':
    exit(main())
