import numpy as np
from tqdm import tqdm

from src.config import SIGNAL_NAMES
from src.detector import Detector
from src.fusion import calibrate_cusum, compute_cusum_risks
from src.numerics import otsu

from experiments.answer_matching import max_f1_score
from experiments.common import PROMPT_TEMPLATE, load_samples, save_results
from experiments.metrics import compute_metrics
from experiments.schema import ExperimentConfig


def _generate_baseline(detector: Detector, samples: list[dict], config: ExperimentConfig) -> list[dict]:
    cached = []
    for sample in tqdm(samples, desc="Generating baseline"):
        prompt = PROMPT_TEMPLATE.format(context=sample["context"], question=sample["question"])
        result = detector.detect(prompt=prompt, max_new_tokens=config.evaluation.max_new_tokens)
        cached.append({
            "cal_matrix": detector.prompt_signals,
            "resp_matrix": np.column_stack([[getattr(s, n) for s in result.token_signals] for n in SIGNAL_NAMES]),
            "response_risk": result.response_risk,
            "f1": max_f1_score(result.generated_text.strip(), sample["answers"]),
        })
    return cached


def _evaluate_condition(cached: list[dict], labels: list[int], disabled: set[str]) -> dict:
    kept = [i for i, s in enumerate(SIGNAL_NAMES) if s not in disabled]
    scores = []
    for c in cached:
        if not disabled:
            scores.append(c["response_risk"])
            continue
        stats = calibrate_cusum(c["cal_matrix"][:, kept])
        _, _, risk, _, _ = compute_cusum_risks(c["resp_matrix"][:, kept], stats)
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

    threshold = otsu([c["f1"] for c in cached])[0]
    labels = [int(c["f1"] < threshold) for c in cached]

    baseline = _evaluate_condition(cached, labels, disabled=set())
    baseline_auroc = baseline["auroc"]
    print(f"Baseline AUROC: {baseline_auroc:.4f}")

    ablation_results = {"baseline": baseline}
    for signal in config.ablation_signals:
        print(f"\nAblating: {signal}...")
        ablation_results[f"without_{signal}"] = _evaluate_condition(cached, labels, disabled={signal})

    print(f"\n{'='*65}\n  Leave-One-Out Signal Ablation: {dataset_name.upper()}\n{'='*65}")
    print(f"  {'Condition':<25} {'AUROC':>8} {'Delta':>8}\n  {'-'*41}")
    print(f"  {'All signals (baseline)':<25} {baseline_auroc:>8.4f} {'---':>8}")
    for signal in config.ablation_signals:
        auroc = ablation_results[f"without_{signal}"]["auroc"]
        print(f"  {'w/o ' + signal:<25} {auroc:>8.4f} {auroc - baseline_auroc:>+8.4f}")
    print(f"{'='*65}\n")

    save_results(ablation_results, f"{config.output_dir}/ablation_{dataset_name}_{model_short}.json")
    return ablation_results
