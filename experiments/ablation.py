#!/usr/bin/env python3
"""
AG-SAR Signal Ablation Study.

Leave-one-out ablation: for each of the 6 signals, disables it during
aggregation and measures delta-AUROC to quantify its contribution.

The ablation works at the fusion layer — signals are still computed
(to preserve hook lifecycle) but excluded from the weighted combination
via the `disabled_signals` parameter in PromptAnchoredAggregator.

Usage:
    python experiments/ablation.py --model meta-llama/Llama-3.1-8B-Instruct --dataset triviaqa --n-samples 100
    python experiments/ablation.py --config experiments/configs/ablation.yaml
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.answer_matching import max_f1_score, compute_adaptive_f1_threshold
from evaluation.loaders.triviaqa import load_triviaqa
from evaluation.loaders.squad import load_squad

ALL_SIGNALS = ["cus", "pos", "dps", "dola", "cgd", "std"]

DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "squad": load_squad,
}


def load_config(config_path: str) -> Dict:
    """Load YAML experiment config, resolving base_config if present."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Resolve base_config
    if "base_config" in config:
        base_path = Path(config_path).parent / config["base_config"]
        if base_path.exists():
            with open(base_path) as f:
                base = yaml.safe_load(f) or {}
            # Merge: ablation config overrides base
            for key in base:
                if key not in config:
                    config[key] = base[key]
    return config


def run_ablation_evaluation(
    model,
    tokenizer,
    samples: List[Dict],
    disabled_signals: Set[str],
    max_new_tokens: int = 64,
    f1_threshold: float = 0.3,
) -> Dict:
    """Run evaluation with specified signals disabled. Returns AUROC or None."""
    from ag_sar.detector import Detector
    from evaluation.metrics import compute_metrics

    detector = Detector(model, tokenizer)

    scores = []
    labels = []
    f1_values = []

    from tqdm import tqdm
    disabled_str = ",".join(sorted(disabled_signals)) if disabled_signals else "none"
    for sample in tqdm(samples, desc=f"Ablation (disabled={disabled_str})"):
        result = detector.detect(
            question=sample["question"],
            context=sample["context"],
            max_new_tokens=max_new_tokens,
        )

        generated = result.generated_text.strip()
        f1 = max_f1_score(generated, sample["answers"])
        f1_values.append(f1)

        # Re-aggregate with disabled signals
        if disabled_signals:
            # Re-run aggregation with signal masking
            response_signals = {}
            for s in result.token_signals:
                for sig in ALL_SIGNALS:
                    response_signals.setdefault(sig, []).append(getattr(s, sig))
            response_signals = {k: np.array(v) for k, v in response_signals.items()}

            agg_result = detector.aggregator.compute_risk(
                detector._prompt_stats,
                response_signals,
                disabled_signals=disabled_signals,
            )
            risk = agg_result.risk
        else:
            risk = result.response_risk

        scores.append(risk)

    # Adaptive F1 threshold
    adaptive_threshold = compute_adaptive_f1_threshold(f1_values)
    threshold = adaptive_threshold if adaptive_threshold != f1_threshold else f1_threshold
    labels = [int(f < threshold) for f in f1_values]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return {"auroc": None, "error": "Single-class labels"}

    metrics = compute_metrics(scores, labels)
    return {"auroc": metrics.auroc, "auprc": metrics.auprc}


def main():
    parser = argparse.ArgumentParser(description="AG-SAR Signal Ablation Study")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset", choices=list(DATASET_LOADERS.keys()), default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = {}
    if args.config:
        config = load_config(args.config)

    model_name = args.model or config.get("model", {}).get("name", "meta-llama/Llama-3.1-8B-Instruct")
    n_samples = args.n_samples or config.get("evaluation", {}).get("n_samples", 100)
    max_new_tokens = args.max_new_tokens or config.get("evaluation", {}).get("max_new_tokens", 64)
    f1_threshold = config.get("evaluation", {}).get("f1_threshold", 0.3)
    signals_to_ablate = config.get("ablation", {}).get("signals", ALL_SIGNALS)
    output_dir = config.get("output", {}).get("dir", "results/ablation")

    dataset_name = args.dataset or (config.get("evaluation", {}).get("datasets", ["triviaqa"])[0])

    model_short = model_name.split("/")[-1]
    torch_dtype_str = config.get("model", {}).get("torch_dtype", "bfloat16")
    torch_dtype = getattr(torch, torch_dtype_str, torch.bfloat16)
    attn_impl = config.get("model", {}).get("attn_implementation", "eager")

    token = os.environ.get("HF_TOKEN")
    print(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=token,
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model loaded in {time.time()-t0:.1f}s")

    samples = DATASET_LOADERS[dataset_name](n_samples=n_samples)
    if not samples:
        print(f"No samples loaded for {dataset_name}")
        return

    # Baseline: all signals enabled
    print(f"\n{'='*65}")
    print("Running baseline (all signals enabled)...")
    baseline = run_ablation_evaluation(
        model, tokenizer, samples,
        disabled_signals=set(),
        max_new_tokens=max_new_tokens,
        f1_threshold=f1_threshold,
    )
    baseline_auroc = baseline.get("auroc")
    print(f"Baseline AUROC: {baseline_auroc:.4f}" if baseline_auroc else "Baseline: degenerate labels")

    # Leave-one-out ablation
    ablation_results = {"baseline": baseline}
    for signal in signals_to_ablate:
        print(f"\nAblating: {signal}...")
        result = run_ablation_evaluation(
            model, tokenizer, samples,
            disabled_signals={signal},
            max_new_tokens=max_new_tokens,
            f1_threshold=f1_threshold,
        )
        ablation_results[f"without_{signal}"] = result

    # Print summary table
    print(f"\n{'='*65}")
    print(f"  AG-SAR Leave-One-Out Signal Ablation: {dataset_name.upper()}")
    print(f"{'='*65}")
    print(f"  {'Condition':<25} {'AUROC':>8} {'Delta':>8}")
    print(f"  {'-'*41}")
    print(f"  {'All signals (baseline)':<25} {baseline_auroc:>8.4f} {'---':>8}" if baseline_auroc else "")

    for signal in signals_to_ablate:
        key = f"without_{signal}"
        auroc = ablation_results[key].get("auroc")
        if auroc is not None and baseline_auroc is not None:
            delta = auroc - baseline_auroc
            print(f"  {'w/o ' + signal:<25} {auroc:>8.4f} {delta:>+8.4f}")
        else:
            print(f"  {'w/o ' + signal:<25} {'N/A':>8} {'N/A':>8}")

    print(f"{'='*65}\n")

    # Save results
    out_path = args.output or os.path.join(output_dir, f"ablation_{dataset_name}_{model_short}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    print(f"Ablation results saved to {out_path}")


if __name__ == "__main__":
    main()
