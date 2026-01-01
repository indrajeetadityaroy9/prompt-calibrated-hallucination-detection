"""
Heatmap visualizations for AG-SAR relevance scores.

Creates attention-style heatmaps showing token relevance.
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_relevance_heatmap(
    tokens: List[str],
    relevance: List[float],
    title: str = "AG-SAR Token Relevance",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = 'YlOrRd',
    save_path: Optional[Path] = None
):
    """
    Create heatmap visualization of token relevance.

    Args:
        tokens: List of tokens
        relevance: Relevance scores (same length as tokens)
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return

    # Ensure same length
    min_len = min(len(tokens), len(relevance))
    tokens = tokens[:min_len]
    relevance = np.array(relevance[:min_len])

    # Normalize relevance
    relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-10)

    fig, ax = plt.subplots(figsize=figsize)

    # Use colormap
    colormap = plt.cm.get_cmap(cmap)

    # Layout tokens with colored backgrounds
    x_pos = 0
    y_pos = 0.5
    line_height = 0.12
    max_width = 12

    for i, (token, rel) in enumerate(zip(tokens, relevance_norm)):
        # Clean token
        display_token = token.replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
        if display_token.startswith(' '):
            display_token = ' ' + display_token.strip()

        # Get color
        color = colormap(rel)

        # Token width
        token_width = len(display_token) * 0.08 + 0.05

        # New line if needed
        if x_pos + token_width > max_width:
            x_pos = 0
            y_pos -= line_height

        # Background rectangle
        rect = plt.Rectangle(
            (x_pos, y_pos - 0.04), token_width, 0.08,
            facecolor=color, edgecolor='gray', linewidth=0.5
        )
        ax.add_patch(rect)

        # Token text
        text_color = 'white' if rel > 0.5 else 'black'
        ax.text(
            x_pos + token_width/2, y_pos, display_token,
            ha='center', va='center', fontsize=8,
            fontfamily='monospace', color=text_color
        )

        x_pos += token_width

    ax.set_xlim(-0.5, max_width + 0.5)
    ax.set_ylim(y_pos - 0.15, 0.65)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, shrink=0.5)
    cbar.set_label('Normalized Relevance (Low → High)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_attention_matrix_heatmap(
    attention_matrix: np.ndarray,
    tokens: Optional[List[str]] = None,
    title: str = "Attention Graph",
    save_path: Optional[Path] = None
):
    """
    Visualize attention matrix as heatmap.

    Args:
        attention_matrix: 2D numpy array of attention weights
        tokens: Optional list of tokens for labels
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto')

    if tokens is not None:
        # Clean tokens for display
        display_tokens = [
            t.replace('Ġ', '').replace('▁', '')[:6]
            for t in tokens
        ]

        n_tokens = len(display_tokens)
        if n_tokens <= 30:
            ax.set_xticks(range(n_tokens))
            ax.set_yticks(range(n_tokens))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(display_tokens, fontsize=7)

    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_multi_sample_heatmap(
    samples: List[Dict],
    max_samples: int = 10,
    title: str = "AG-SAR Relevance Across Samples",
    save_path: Optional[Path] = None
):
    """
    Create stacked heatmap for multiple samples.

    Args:
        samples: List of dicts with 'tokens' and 'relevance' keys
        max_samples: Maximum number of samples to show
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return

    samples = samples[:max_samples]
    n_samples = len(samples)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, (ax, sample) in enumerate(zip(axes, samples)):
        tokens = sample['tokens']
        relevance = np.array(sample['relevance'])

        # Normalize
        relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-10)

        # Create heatmap row
        ax.imshow(
            relevance_norm.reshape(1, -1),
            cmap='YlOrRd',
            aspect='auto'
        )

        # Token labels
        n_tokens = len(tokens)
        if n_tokens <= 40:
            display_tokens = [t.replace('Ġ', '').replace('▁', '')[:5] for t in tokens]
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=6)

        ax.set_yticks([])
        ax.set_ylabel(f'Sample {i+1}', fontsize=8)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_pos_relevance_heatmap(
    tokens: List[str],
    relevance: List[float],
    pos_tags: List[str],
    title: str = "Relevance by Part of Speech",
    save_path: Optional[Path] = None
):
    """
    Create heatmap with POS tag annotations.

    Args:
        tokens: List of tokens
        relevance: Relevance scores
        pos_tags: POS tags for each token
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for visualization")
        return

    # Color code by POS
    pos_colors = {
        'NOUN': '#e74c3c',      # Red
        'VERB': '#3498db',      # Blue
        'ADJ': '#2ecc71',       # Green
        'ADV': '#9b59b6',       # Purple
        'PROPN': '#f39c12',     # Orange
        'DET': '#95a5a6',       # Gray
        'ADP': '#bdc3c7',       # Light gray
        'PUNCT': '#7f8c8d',     # Dark gray
    }

    min_len = min(len(tokens), len(relevance), len(pos_tags))
    tokens = tokens[:min_len]
    relevance = np.array(relevance[:min_len])
    pos_tags = pos_tags[:min_len]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 4), height_ratios=[1, 0.3])

    # Relevance heatmap
    relevance_norm = (relevance - relevance.min()) / (relevance.max() - relevance.min() + 1e-10)
    ax1.imshow(relevance_norm.reshape(1, -1), cmap='YlOrRd', aspect='auto')

    if len(tokens) <= 40:
        display_tokens = [t.replace('Ġ', '').replace('▁', '')[:5] for t in tokens]
        ax1.set_xticks(range(len(tokens)))
        ax1.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=7)

    ax1.set_yticks([])
    ax1.set_ylabel('Relevance')

    # POS color strip
    pos_colors_arr = [pos_colors.get(p, '#cccccc') for p in pos_tags]
    for i, color in enumerate(pos_colors_arr):
        ax2.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))

    ax2.set_xlim(0, len(tokens))
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_ylabel('POS')

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, label=pos)
        for pos, color in pos_colors.items()
    ]
    ax2.legend(handles=handles, loc='upper right', ncol=4, fontsize=6)

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
