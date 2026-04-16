import copy

import numpy as np
from tqdm import tqdm

from src.fusion import compute_cusum_risks
from src.detector import Detector

from experiments.answer_matching import compute_adaptive_f1_threshold, max_f1_score
from experiments.common import PROMPT_TEMPLATE, load_samples, save_results
from experiments.metrics import compute_metrics
from experiments.schema import ExperimentConfig

_SIGNAL_ORDER = ["rho", "phi", "spf", "mlp", "ent"]


def _generate_baseline(detector: Detector, samples: list[dict], config: ExperimentConfig) -> list[dict]:
    cached = []
    for sample in tqdm(samples, desc="Generating baseline"):
        prompt = PROMPT_TEMPLATE.format(context=sample["context"], question=sample["question"])
        result = detector.detect(prompt=prompt, max_new_tokens=config.evaluation.max_new_tokens)
        generated = result.generated_text.strip()
        cached.append({
            "response_signals": {sig: np.array([getattr(s, sig) for s in result.token_signals]) for sig in _SIGNAL_ORDER},
            "prompt_stats": copy.deepcopy(detector.prompt_stats),
            "response_risk": result.response_risk,
            "f1": max_f1_score(generated, sample["answers"]),
        })
    return cached


def _evaluate_condition(cached: list[dict], labels: list[int], disabled_signals: set[str]) -> dict:
    scores = []
    for c in cached:
        if not disabled_signals:
            scores.append(c["response_risk"])
        else:
            signal_matrix = np.column_stack([c["response_signals"][sig] for sig in _SIGNAL_ORDER])
            for i, sig in enumerate(_SIGNAL_ORDER):
                if sig in disabled_signals:
                    signal_matrix[:, i] = c["prompt_stats"].mu[i]
            _, _, risk, _, _ = compute_cusum_risks(signal_matrix, c["prompt_stats"])
            scores.append(risk)
    metrics = compute_metrics(scores, labels)
    return {"auroc": metrics.auroc, "auprc": metrics.auprc}


def run_ablation(model, tokenizer, config: ExperimentConfig) -> dict:
    detector = Detector(model, tokenizer)
    model_short = config.model.name.split("/")[-1]
    dataset_name = config.evaluation.datasets[0]
    samples = load_samples(dataset_name, config.evaluation.n_samples, config.evaluation.max_context_chars)

    print(f"\n{'='*65}\nGenerating baseline (all signals, single pass)...")
    cached = _generate_baseline(detector, samples, config)

    threshold = compute_adaptive_f1_threshold([c["f1"] for c in cached])
    labels = [int(c["f1"] < threshold) for c in cached]

    baseline = _evaluate_condition(cached, labels, disabled_signals=set())
    baseline_auroc = baseline["auroc"]
    print(f"Baseline AUROC: {baseline_auroc:.4f}")

    ablation_results = {"baseline": baseline}
    for signal in config.ablation_signals:
        print(f"\nAblating: {signal}...")
        ablation_results[f"without_{signal}"] = _evaluate_condition(cached, labels, disabled_signals={signal})

    print(f"\n{'='*65}\n  Leave-One-Out Signal Ablation: {dataset_name.upper()}\n{'='*65}")
    print(f"  {'Condition':<25} {'AUROC':>8} {'Delta':>8}\n  {'-'*41}")
    print(f"  {'All signals (baseline)':<25} {baseline_auroc:>8.4f} {'---':>8}")
    for signal in config.ablation_signals:
        auroc = ablation_results[f"without_{signal}"]["auroc"]
        print(f"  {'w/o ' + signal:<25} {auroc:>8.4f} {auroc - baseline_auroc:>+8.4f}")
    print(f"{'='*65}\n")

    save_results(ablation_results, f"{config.output_dir}/ablation_{dataset_name}_{model_short}.json")
    return ablation_results
