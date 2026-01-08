"""
Anatomy of a Hallucination Visualization

Visualizes how AG-SAR's three signals evolve during hallucination onset:
  - Authority Score (blue): Information provenance - drops when hallucinating
  - Agreement Gate (orange): MLP-Attention consensus - diverges at hallucination
  - Semantic Dispersion (red): Top-k embedding consistency - spikes at hallucination

The visualization reveals:
  - WHEN hallucination begins (onset detection)
  - HOW the three signals interact during failure
  - WHY certain tokens are flagged as uncertain

Usage:
  # From experiment results:
  python experiments/analysis/plot_hallucination_anatomy.py \\
      --dir results/anatomy_visualization

  # Demo mode with synthetic data (no GPU needed):
  python experiments/analysis/plot_hallucination_anatomy.py --demo

  # Specify output format:
  python experiments/analysis/plot_hallucination_anatomy.py --demo --format pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def smooth(values: List[float], window: int = 3) -> np.ndarray:
    """
    Apply light moving average for visualization.

    Per plan: "Apply window=3 moving average to dispersion line"
    This is visualization-only - raw data stays unsmoothed.

    Args:
        values: Raw per-token values
        window: Moving average window size

    Returns:
        Smoothed values as numpy array
    """
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode='same')


def generate_demo_data() -> Dict:
    """
    Generate synthetic per-token data demonstrating hallucination onset.

    Simulates a response that starts faithful to context (tokens 0-15)
    then begins hallucinating (tokens 16+).

    Returns:
        dict with per_token data structure matching actual experiment output
    """
    np.random.seed(42)
    num_tokens = 40

    # Hallucination onset at token 16
    onset_idx = 16

    # Authority: High before onset, drops sharply at hallucination
    authority = []
    for i in range(num_tokens):
        if i < onset_idx:
            # Faithful region: high authority (0.7-0.9)
            authority.append(0.8 + np.random.normal(0, 0.05))
        else:
            # Hallucinating region: low authority (0.2-0.5), slight recovery
            decay = 0.3 + 0.15 * (i - onset_idx) / (num_tokens - onset_idx)
            authority.append(decay + np.random.normal(0, 0.08))

    # Gate: Stable before onset, drops when MLP diverges from attention
    gate = []
    for i in range(num_tokens):
        if i < onset_idx:
            # Faithful: high gate (MLP agrees with attention)
            gate.append(0.85 + np.random.normal(0, 0.05))
        else:
            # Hallucinating: gate drops as MLP makes unsupported claims
            drop = 0.5 - 0.2 * np.sin((i - onset_idx) * 0.3)
            gate.append(drop + np.random.normal(0, 0.1))

    # Dispersion: Low before onset, spikes when model is uncertain about semantics
    dispersion = []
    for i in range(num_tokens):
        if i < onset_idx:
            # Faithful: low dispersion (top-k embeddings consistent)
            dispersion.append(0.1 + np.random.normal(0, 0.03))
        elif i == onset_idx:
            # Onset: sharp spike in dispersion
            dispersion.append(0.7 + np.random.normal(0, 0.05))
        else:
            # Hallucinating: elevated dispersion with oscillation
            base = 0.5 + 0.15 * np.sin((i - onset_idx) * 0.5)
            dispersion.append(base + np.random.normal(0, 0.08))

    # Synthetic tokens (illustrative)
    tokens = [
        "The", " capital", " of", " France", " is", " Paris", ",",
        " which", " is", " known", " for", " the", " Eiffel", " Tower", ".", " ",
        # Hallucination onset
        "The", " tower", " was", " built", " in", " 1920",  # Wrong! Built 1889
        " by", " Napoleon", # Wrong! Gustave Eiffel
        " III", " as", " a", " military", " structure", # Wrong! World's Fair
        ",", " standing", " 500", " meters", " tall", "."  # Wrong! 330m
    ]

    # Ensure token count matches
    tokens = tokens[:num_tokens]
    while len(tokens) < num_tokens:
        tokens.append(f"[{len(tokens)}]")

    return {
        "authority": authority,
        "gate": gate,
        "dispersion": dispersion,
        "tokens": tokens,
        "response_start": 0,
        "hallucination_onset": onset_idx,  # Demo-only metadata
    }


def load_per_token_data(results_dir: Path) -> List[Dict]:
    """
    Load per-token data from experiment results.

    Expects JSONL format with per_token field in extra dict.

    Args:
        results_dir: Directory containing results JSONL files

    Returns:
        List of per-token data dicts
    """
    samples = []

    # Look for predictions or scores files
    for pattern in ["predictions*.jsonl", "scores*.jsonl", "*.jsonl"]:
        for jsonl_path in results_dir.glob(pattern):
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        # Check for per_token data
                        extra = record.get("extra", {})
                        per_token = extra.get("per_token")
                        if per_token and per_token.get("authority"):
                            samples.append(per_token)
                    except json.JSONDecodeError:
                        continue

    return samples


def plot_hallucination_anatomy(
    per_token: Dict,
    output_path: Optional[Path] = None,
    title: str = "Anatomy of a Hallucination",
    show_onset: bool = True,
    smooth_dispersion: bool = True,
) -> None:
    """
    Generate the hallucination anatomy visualization.

    Creates a 3-line plot showing how AG-SAR's signals evolve:
      - Authority (blue): Drops when hallucinating
      - Gate (orange): Diverges when MLP makes unsupported claims
      - Dispersion (red): Spikes when semantically uncertain

    Args:
        per_token: Dict with authority, gate, dispersion, tokens arrays
        output_path: Where to save figure (if None, displays interactively)
        title: Plot title
        show_onset: Whether to shade hallucination region (if metadata available)
        smooth_dispersion: Apply window=3 smoothing to dispersion line
    """
    authority = np.array(per_token["authority"])
    gate = np.array(per_token["gate"]) if per_token.get("gate") else None
    dispersion = np.array(per_token["dispersion"]) if per_token.get("dispersion") else None
    tokens = per_token.get("tokens", [])

    num_tokens = len(authority)
    x = np.arange(num_tokens)

    # Create figure with publication-quality settings
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

    # Plot Authority (blue)
    ax.plot(x, authority, color='#1f77b4', linewidth=2, label='Authority Score', marker='o', markersize=3)

    # Plot Gate (orange) if available
    if gate is not None:
        ax.plot(x, gate, color='#ff7f0e', linewidth=2, label='Agreement Gate', marker='s', markersize=3)

    # Plot Dispersion (red) if available - with optional smoothing
    if dispersion is not None:
        if smooth_dispersion:
            dispersion_plot = smooth(dispersion.tolist(), window=3)
        else:
            dispersion_plot = dispersion
        ax.plot(x, dispersion_plot, color='#d62728', linewidth=2, label='Semantic Dispersion', marker='^', markersize=3)

    # Shade hallucination onset region if available
    onset_idx = per_token.get("hallucination_onset")
    if show_onset and onset_idx is not None:
        ax.axvspan(onset_idx, num_tokens, alpha=0.15, color='red', label='Hallucination Region')
        ax.axvline(x=onset_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

        # Add onset annotation
        ax.annotate(
            'Onset',
            xy=(onset_idx, 0.85),
            xytext=(onset_idx + 2, 0.95),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
            color='red',
            weight='bold',
        )

    # Styling
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Signal Value', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(-0.5, num_tokens - 0.5)

    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add token labels on x-axis for small sequences
    if num_tokens <= 50 and tokens:
        # Show every 5th token to avoid clutter
        tick_positions = list(range(0, num_tokens, 5))
        tick_labels = [tokens[i] if i < len(tokens) else '' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # Add interpretation text box
    interpretation = (
        "Interpretation:\n"
        "- Authority drops: Token derives from memory, not context\n"
        "- Gate diverges: MLP output conflicts with attention pattern\n"
        "- Dispersion spikes: Model uncertain about semantic meaning"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, interpretation, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props, family='monospace')

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {output_path}")

        # Also save PDF if PNG was requested
        if output_path.suffix == '.png':
            pdf_path = output_path.with_suffix('.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"Saved: {pdf_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AG-SAR signals during hallucination onset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", type=str, default=None,
        help="Directory containing experiment results with per-token scores"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Use synthetic demo data (no GPU or experiment results needed)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path (default: figures/anatomy_hallucination.png)"
    )
    parser.add_argument(
        "--format", type=str, choices=["png", "pdf", "svg"], default="png",
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0,
        help="Which sample to visualize (default: 0, first sample)"
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Disable smoothing on dispersion line"
    )

    args = parser.parse_args()

    # Determine data source
    if args.demo:
        print("Using synthetic demo data...")
        per_token = generate_demo_data()
    elif args.dir:
        results_dir = Path(args.dir)
        if not results_dir.exists():
            print(f"Error: Results directory not found: {results_dir}")
            return 1

        samples = load_per_token_data(results_dir)
        if not samples:
            print(f"Error: No per-token data found in {results_dir}")
            print("Did you run with save_per_token_scores: true?")
            return 1

        if args.sample_idx >= len(samples):
            print(f"Error: Sample index {args.sample_idx} out of range (0-{len(samples)-1})")
            return 1

        per_token = samples[args.sample_idx]
        print(f"Loaded sample {args.sample_idx} of {len(samples)} from {results_dir}")
    else:
        print("Error: Must specify --dir or --demo")
        parser.print_help()
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("figures") / f"anatomy_hallucination.{args.format}"

    # Generate visualization
    title = "Anatomy of a Hallucination" if args.demo else "AG-SAR Signal Dynamics"
    plot_hallucination_anatomy(
        per_token=per_token,
        output_path=output_path,
        title=title,
        show_onset=args.demo,  # Only show onset region for demo data
        smooth_dispersion=not args.no_smooth,
    )

    return 0


if __name__ == "__main__":
    exit(main())
