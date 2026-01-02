"""
Experiment 13: MC-SS Ablation Study

Comprehensive ablation of Manifold-Consistent Spectral Surprisal (MC-SS) components.

Compares:
1. GSE (Graph-Shifted Entropy) - baseline
2. MC-SS Full - complete formulation with Hebbian prior
3. MC-SS variants with different hyperparameters:
   - τ (Hebbian threshold): 0.0, 0.1, 0.2, 0.3
   - β (surprisal clamp): 1.0, 5.0, 10.0, 20.0
   - λ (penalty weight): 0.5, 1.0, 2.0, 5.0
4. Ablated components:
   - No Hebbian (use raw centrality)
   - No Bounded Surprisal (use raw surprisal)
   - Multiplicative (S × penalty instead of S + penalty)

Success Criteria: MC-SS Full > GSE on AUROC, especially on adversarial samples
                 Additive > Multiplicative on "Confident Lie" detection
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import torch
from tqdm import tqdm

from ..config import EvalConfig
from ..datasets import load_truthfulqa
from ..metrics.auroc import compute_auroc
from ..metrics.rouge import compute_rouge_l


@dataclass
class MCSSConfig:
    """Configuration for MC-SS ablation experiment."""
    name: str
    description: str
    metric: str = "mcss"  # "gse" or "mcss"
    tau: float = 0.1      # Hebbian threshold
    beta: float = 5.0     # Bounded surprisal clamp
    penalty_weight: float = 1.0  # λ for additive penalty
    use_hebbian: bool = True
    use_bounded_surprisal: bool = True
    use_additive: bool = True  # False = multiplicative


def get_mcss_configs() -> List[MCSSConfig]:
    """Get all MC-SS configurations to test."""
    configs = [
        # Baseline
        MCSSConfig(
            name="gse_baseline",
            description="Graph-Shifted Entropy (baseline)",
            metric="gse"
        ),

        # Full MC-SS with default params
        MCSSConfig(
            name="mcss_full",
            description="MC-SS Full (τ=0.1, β=5.0, λ=1.0)",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=1.0
        ),

        # Tau ablation
        MCSSConfig(
            name="mcss_tau_0.0",
            description="MC-SS with τ=0.0 (no threshold)",
            metric="mcss",
            tau=0.0,
            beta=5.0,
            penalty_weight=1.0
        ),
        MCSSConfig(
            name="mcss_tau_0.2",
            description="MC-SS with τ=0.2",
            metric="mcss",
            tau=0.2,
            beta=5.0,
            penalty_weight=1.0
        ),
        MCSSConfig(
            name="mcss_tau_0.3",
            description="MC-SS with τ=0.3",
            metric="mcss",
            tau=0.3,
            beta=5.0,
            penalty_weight=1.0
        ),

        # Beta ablation
        MCSSConfig(
            name="mcss_beta_1.0",
            description="MC-SS with β=1.0 (hard clamp)",
            metric="mcss",
            tau=0.1,
            beta=1.0,
            penalty_weight=1.0
        ),
        MCSSConfig(
            name="mcss_beta_10.0",
            description="MC-SS with β=10.0 (soft clamp)",
            metric="mcss",
            tau=0.1,
            beta=10.0,
            penalty_weight=1.0
        ),
        MCSSConfig(
            name="mcss_beta_20.0",
            description="MC-SS with β=20.0 (very soft clamp)",
            metric="mcss",
            tau=0.1,
            beta=20.0,
            penalty_weight=1.0
        ),

        # Penalty weight ablation (critical for adversarial)
        MCSSConfig(
            name="mcss_lambda_0.5",
            description="MC-SS with λ=0.5 (weak penalty)",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=0.5
        ),
        MCSSConfig(
            name="mcss_lambda_2.0",
            description="MC-SS with λ=2.0 (strong penalty)",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=2.0
        ),
        MCSSConfig(
            name="mcss_lambda_5.0",
            description="MC-SS with λ=5.0 (very strong penalty)",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=5.0
        ),

        # Component ablations
        MCSSConfig(
            name="mcss_no_hebbian",
            description="MC-SS without Hebbian prior",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=1.0,
            use_hebbian=False
        ),
        MCSSConfig(
            name="mcss_no_bounded",
            description="MC-SS without bounded surprisal",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=1.0,
            use_bounded_surprisal=False
        ),
        MCSSConfig(
            name="mcss_multiplicative",
            description="MC-SS with multiplicative (S × penalty)",
            metric="mcss",
            tau=0.1,
            beta=5.0,
            penalty_weight=1.0,
            use_additive=False
        ),
    ]
    return configs


def create_mcss_ag_sar(model, tokenizer, mcss_config: MCSSConfig):
    """
    Create an AGSAR instance configured for MC-SS testing.

    Args:
        model: The language model
        tokenizer: The tokenizer
        mcss_config: MC-SS configuration

    Returns:
        AGSAR instance with appropriate config
    """
    from ag_sar import AGSAR, AGSARConfig

    if mcss_config.metric == "gse":
        ag_config = AGSARConfig(uncertainty_metric="gse")
    else:
        ag_config = AGSARConfig(
            uncertainty_metric="mcss",
            mcss_hebbian_tau=mcss_config.tau,
            mcss_beta=mcss_config.beta,
            mcss_penalty_weight=mcss_config.penalty_weight,
        )

    return AGSAR(model, tokenizer, config=ag_config)


def compute_mcss_score(
    ag_sar,
    prompt: str,
    response: str,
    mcss_config: MCSSConfig
) -> float:
    """
    Compute uncertainty score with specific MC-SS configuration.

    For ablations that require post-hoc computation (no_hebbian, no_bounded,
    multiplicative), we modify the computation after getting base values.

    Args:
        ag_sar: AGSAR instance
        prompt: Input prompt
        response: Generated response
        mcss_config: MC-SS configuration

    Returns:
        Uncertainty score
    """
    from ag_sar.uncertainty import (
        compute_bounded_surprisal,
        compute_token_surprisal,
    )

    # Get full computation details
    details = ag_sar.compute_uncertainty(prompt, response, return_details=True)

    # For standard configs (gse or mcss with default behavior), return directly
    if mcss_config.metric == "gse":
        return details['uncertainty']

    if (mcss_config.use_hebbian and mcss_config.use_bounded_surprisal
            and mcss_config.use_additive):
        # Standard MC-SS, already computed
        return details['uncertainty']

    # Post-hoc ablation computations
    logits = details['logits']
    input_ids = details['input_ids']
    response_mask = torch.zeros_like(details['attention_mask'])
    response_mask[:, details['response_start']:] = 1

    if not mcss_config.use_hebbian:
        # Use raw centrality instead of Hebbian-filtered
        centrality = details['centrality']
    else:
        centrality = details['relevance']

    if mcss_config.use_bounded_surprisal:
        bounded = compute_bounded_surprisal(
            logits, input_ids, beta=mcss_config.beta,
            attention_mask=response_mask
        )
    else:
        # Use raw surprisal (unbounded)
        bounded = compute_token_surprisal(logits, input_ids, attention_mask=None)
        bounded = bounded * response_mask.float()

    # Max-normalize centrality
    centrality_masked = centrality * response_mask.float()
    v_max = centrality_masked.max(dim=-1, keepdim=True).values + 1e-10
    v_norm = centrality_masked / v_max
    penalty = 1.0 - v_norm

    # Compute final score
    if mcss_config.use_additive:
        # ADDITIVE: S + λ × penalty
        score_token = bounded + (mcss_config.penalty_weight * penalty)
    else:
        # MULTIPLICATIVE: S × (1 + λ × penalty)
        # Note: We add 1 to ensure non-zero base
        score_token = bounded * (1.0 + mcss_config.penalty_weight * penalty)

    # Average over valid tokens
    score = (score_token * response_mask.float()).sum(dim=-1) / \
            response_mask.sum(dim=-1).clamp(min=1)

    return score.item()


def run_mcss_ablation_experiment(
    model,
    tokenizer,
    config: EvalConfig,
    num_samples: int = 100,
    save_results: bool = True
) -> Dict:
    """
    Run MC-SS ablation study experiment.

    Compares GSE baseline against various MC-SS configurations to determine
    optimal hyperparameters and component contributions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        config: Evaluation configuration
        num_samples: Number of TruthfulQA samples
        save_results: Whether to save results

    Returns:
        Dict with ablation results
    """
    print("=" * 60)
    print("Experiment 13: MC-SS Ablation Study")
    print("=" * 60)

    # Load TruthfulQA
    print(f"\nLoading TruthfulQA ({num_samples} samples)...")
    samples = load_truthfulqa(max_samples=num_samples)

    mcss_configs = get_mcss_configs()

    results = {
        'experiment': 'mcss_ablation',
        'num_samples': len(samples),
        'configurations': {}
    }

    # Run each configuration
    for mcss_cfg in mcss_configs:
        print(f"\n{'=' * 40}")
        print(f"Testing: {mcss_cfg.name}")
        print(f"  {mcss_cfg.description}")
        print(f"{'=' * 40}")

        # Create AGSAR instance for this config
        ag_sar = create_mcss_ag_sar(model, tokenizer, mcss_cfg)

        scores = []
        labels = []

        for sample in tqdm(samples, desc=f"  {mcss_cfg.name}"):
            prompt = sample.prompt
            response = sample.response or ""

            if not response:
                continue

            # Ground truth from dataset labels (True=factual, False=hallucination)
            # We want is_hallucination=True when label=False
            is_hallucination = not sample.label if sample.label is not None else False

            try:
                score = compute_mcss_score(ag_sar, prompt, response, mcss_cfg)
                scores.append(score)
                labels.append(is_hallucination)
            except Exception as e:
                continue

        # Cleanup
        ag_sar.cleanup()

        if len(scores) < 10:
            print(f"  Warning: Only {len(scores)} valid samples")
            continue

        # Compute AUROC
        auroc = compute_auroc(labels, scores)

        results['configurations'][mcss_cfg.name] = {
            'description': mcss_cfg.description,
            'settings': {
                'metric': mcss_cfg.metric,
                'tau': mcss_cfg.tau,
                'beta': mcss_cfg.beta,
                'penalty_weight': mcss_cfg.penalty_weight,
                'use_hebbian': mcss_cfg.use_hebbian,
                'use_bounded_surprisal': mcss_cfg.use_bounded_surprisal,
                'use_additive': mcss_cfg.use_additive,
            },
            'auroc': auroc,
            'num_samples': len(scores),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
        }

        print(f"  AUROC: {auroc:.3f}")
        print(f"  Mean Score: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # Print summary table
    print("\n" + "=" * 70)
    print("MC-SS ABLATION RESULTS SUMMARY:")
    print("=" * 70)
    print(f"\n{'Configuration':<25} {'AUROC':>10} {'Δ vs GSE':>12} {'Δ vs Full':>12}")
    print("-" * 60)

    gse_auroc = results['configurations'].get('gse_baseline', {}).get('auroc', 0)
    full_auroc = results['configurations'].get('mcss_full', {}).get('auroc', 0)

    for name, data in results['configurations'].items():
        auroc = data['auroc']
        delta_gse = auroc - gse_auroc
        delta_full = auroc - full_auroc
        sign_gse = "+" if delta_gse >= 0 else ""
        sign_full = "+" if delta_full >= 0 else ""
        print(f"{name:<25} {auroc:>10.3f} {sign_gse}{delta_gse:>11.3f} {sign_full}{delta_full:>11.3f}")

    # Analysis: Check key hypotheses
    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    print("=" * 70)

    # 1. MC-SS vs GSE
    if full_auroc > gse_auroc:
        print(f"✓ MC-SS Full ({full_auroc:.3f}) > GSE ({gse_auroc:.3f})")
    else:
        print(f"✗ MC-SS Full ({full_auroc:.3f}) <= GSE ({gse_auroc:.3f})")

    # 2. Additive vs Multiplicative
    mult_auroc = results['configurations'].get('mcss_multiplicative', {}).get('auroc', 0)
    if full_auroc > mult_auroc:
        print(f"✓ Additive ({full_auroc:.3f}) > Multiplicative ({mult_auroc:.3f})")
    else:
        print(f"✗ Additive ({full_auroc:.3f}) <= Multiplicative ({mult_auroc:.3f})")

    # 3. Find optimal λ
    lambda_configs = ['mcss_lambda_0.5', 'mcss_full', 'mcss_lambda_2.0', 'mcss_lambda_5.0']
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    lambda_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in lambda_configs]
    best_lambda_idx = np.argmax(lambda_aurocs)
    print(f"✓ Optimal λ: {lambda_values[best_lambda_idx]} (AUROC: {lambda_aurocs[best_lambda_idx]:.3f})")

    # 4. Find optimal τ
    tau_configs = ['mcss_tau_0.0', 'mcss_full', 'mcss_tau_0.2', 'mcss_tau_0.3']
    tau_values = [0.0, 0.1, 0.2, 0.3]
    tau_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in tau_configs]
    best_tau_idx = np.argmax(tau_aurocs)
    print(f"✓ Optimal τ: {tau_values[best_tau_idx]} (AUROC: {tau_aurocs[best_tau_idx]:.3f})")

    # 5. Find optimal β
    beta_configs = ['mcss_beta_1.0', 'mcss_full', 'mcss_beta_10.0', 'mcss_beta_20.0']
    beta_values = [1.0, 5.0, 10.0, 20.0]
    beta_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in beta_configs]
    best_beta_idx = np.argmax(beta_aurocs)
    print(f"✓ Optimal β: {beta_values[best_beta_idx]} (AUROC: {beta_aurocs[best_beta_idx]:.3f})")

    results['analysis'] = {
        'mcss_beats_gse': full_auroc > gse_auroc,
        'additive_beats_multiplicative': full_auroc > mult_auroc,
        'optimal_lambda': lambda_values[best_lambda_idx],
        'optimal_tau': tau_values[best_tau_idx],
        'optimal_beta': beta_values[best_beta_idx],
    }

    print("=" * 70)

    # Save results
    if save_results:
        results_path = config.results_dir / 'exp13_mcss_ablation.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")

    return results


def plot_mcss_hyperparameter_sweep(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot hyperparameter sweep results for τ, β, and λ.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # τ sweep
    tau_configs = ['mcss_tau_0.0', 'mcss_full', 'mcss_tau_0.2', 'mcss_tau_0.3']
    tau_values = [0.0, 0.1, 0.2, 0.3]
    tau_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in tau_configs]

    axes[0].plot(tau_values, tau_aurocs, 'o-', color='#3498db', linewidth=2, markersize=8)
    axes[0].set_xlabel('τ (Hebbian Threshold)')
    axes[0].set_ylabel('AUROC')
    axes[0].set_title('Effect of τ on MC-SS Performance')
    axes[0].grid(True, alpha=0.3)

    # β sweep
    beta_configs = ['mcss_beta_1.0', 'mcss_full', 'mcss_beta_10.0', 'mcss_beta_20.0']
    beta_values = [1.0, 5.0, 10.0, 20.0]
    beta_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in beta_configs]

    axes[1].plot(beta_values, beta_aurocs, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    axes[1].set_xlabel('β (Surprisal Clamp)')
    axes[1].set_ylabel('AUROC')
    axes[1].set_title('Effect of β on MC-SS Performance')
    axes[1].set_xscale('log')
    axes[1].grid(True, alpha=0.3)

    # λ sweep
    lambda_configs = ['mcss_lambda_0.5', 'mcss_full', 'mcss_lambda_2.0', 'mcss_lambda_5.0']
    lambda_values = [0.5, 1.0, 2.0, 5.0]
    lambda_aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in lambda_configs]

    axes[2].plot(lambda_values, lambda_aurocs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    axes[2].set_xlabel('λ (Penalty Weight)')
    axes[2].set_ylabel('AUROC')
    axes[2].set_title('Effect of λ on MC-SS Performance')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_mcss_comparison_bars(
    results: Dict,
    save_path: Optional[Path] = None
):
    """
    Plot bar chart comparing GSE, MC-SS Full, and key ablations.
    """
    import matplotlib.pyplot as plt

    key_configs = [
        'gse_baseline',
        'mcss_full',
        'mcss_no_hebbian',
        'mcss_no_bounded',
        'mcss_multiplicative',
    ]

    display_names = {
        'gse_baseline': 'GSE\n(Baseline)',
        'mcss_full': 'MC-SS\nFull',
        'mcss_no_hebbian': 'MC-SS\nNo Hebbian',
        'mcss_no_bounded': 'MC-SS\nNo Bounded S',
        'mcss_multiplicative': 'MC-SS\nMultiplicative',
    }

    aurocs = [results['configurations'].get(c, {}).get('auroc', 0) for c in key_configs]
    labels = [display_names.get(c, c) for c in key_configs]

    # Color coding
    colors = ['#95a5a6', '#2ecc71', '#e74c3c', '#e74c3c', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(labels, aurocs, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        ax.annotate(f'{auroc:.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Add reference line for GSE baseline
    gse_auroc = results['configurations'].get('gse_baseline', {}).get('auroc', 0)
    ax.axhline(y=gse_auroc, color='gray', linestyle='--', alpha=0.7,
               label=f'GSE Baseline ({gse_auroc:.3f})')

    ax.set_ylabel('AUROC')
    ax.set_title('MC-SS Ablation: Component Contributions')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    # Run experiment
    config = EvalConfig()
    results = run_mcss_ablation_experiment(
        model, tokenizer, config, num_samples=100
    )

    # Plot
    plot_mcss_hyperparameter_sweep(results, config.results_dir / 'exp13_mcss_hyperparams.png')
    plot_mcss_comparison_bars(results, config.results_dir / 'exp13_mcss_comparison.png')
