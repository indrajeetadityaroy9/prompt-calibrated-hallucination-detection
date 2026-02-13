"""
Signal Generalization Experiments for Hallucination Detection.

This script runs comprehensive experiments to determine which signals
generalize best across different data types.

Experiments:
- Phase 1: Individual signal evaluation (15 signals × 6 datasets)
- Phase 2: Correlation and redundancy analysis
- Phase 3: Combination strategies (Noisy-OR, complementarity)
- Phase 4: Length control (residualization, stratification)
- Phase 5: Task-specific vs task-agnostic analysis

Key metric: Delta above length baseline (AUROC - Length_AUROC)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import random
import json
import argparse
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM

EPS = 1e-10


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SignalScore:
    """Per-example signal scores."""
    signal_name: str
    score: float
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class DatasetResult:
    """Results for one signal on one dataset."""
    signal_name: str
    dataset_name: str
    raw_auroc: float
    length_auroc: float
    delta_auroc: float
    signal_length_corr: float
    pos_mean: float
    neg_mean: float
    n_examples: int


@dataclass
class ExperimentResults:
    """Container for all experiment results."""
    signal_matrix: Dict[str, Dict[str, DatasetResult]] = field(default_factory=dict)
    correlations: Dict[str, np.ndarray] = field(default_factory=dict)
    combinations: Dict[str, Dict] = field(default_factory=dict)
    length_control: Dict[str, Dict] = field(default_factory=dict)
    task_analysis: Dict[str, Dict] = field(default_factory=dict)


# =============================================================================
# Signal Implementations
# =============================================================================

class SignalComputer:
    """Compute all signals for a given example."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def compute_all_signals(
        self,
        hidden_states: List[torch.Tensor],
        logits: torch.Tensor,
        prompt_len: int,
        context_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Compute all available signals for an example.

        Returns dict mapping signal_name -> score (higher = more risk).
        """
        signals = {}
        n_layers = len(hidden_states)
        seq_len = hidden_states[0].shape[1]
        response_len = seq_len - prompt_len

        if response_len < 1:
            return {name: 0.5 for name in self._signal_names()}

        # Response logits and hidden states
        response_logits = logits[prompt_len:].float()
        probs = torch.softmax(response_logits, dim=-1)

        # === Probability Family ===
        # Entropy (normalized)
        entropy = -torch.sum(probs * torch.log2(probs + EPS), dim=-1)
        max_entropy = np.log2(probs.shape[-1])
        entropy_norm = entropy / max_entropy
        signals['entropy'] = float(entropy_norm.mean().cpu())

        # Inverse margin
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        margin = top2_probs[:, 0] - top2_probs[:, 1]
        inv_margin = 1 - margin
        signals['inv_margin'] = float(inv_margin.mean().cpu())

        # Target probability (of argmax token)
        target_prob = probs.max(dim=-1).values
        signals['target_prob'] = float((1 - target_prob).mean().cpu())  # Invert: low prob = high risk

        # === Internal Dynamics Family ===
        # JSD between early and late layers
        early_layer = n_layers // 4
        late_layer = n_layers - 1

        early_hidden = hidden_states[early_layer][0, prompt_len:].float()
        late_hidden = hidden_states[late_layer][0, prompt_len:].float()

        # Project to logits via lm_head
        lm_head_dtype = self.model.lm_head.weight.dtype
        with torch.no_grad():
            early_logits = self.model.lm_head(early_hidden.to(lm_head_dtype))
            late_logits = self.model.lm_head(late_hidden.to(lm_head_dtype))

        early_probs = torch.softmax(early_logits, dim=-1)
        late_probs = torch.softmax(late_logits, dim=-1)

        # JSD
        m = 0.5 * (early_probs + late_probs)
        jsd = 0.5 * (
            torch.sum(early_probs * torch.log2((early_probs + EPS) / (m + EPS)), dim=-1) +
            torch.sum(late_probs * torch.log2((late_probs + EPS) / (m + EPS)), dim=-1)
        )
        signals['jsd_cand'] = float(jsd.mean().cpu())

        # LCI (Late Commitment Index) - simplified version
        # Check if final prediction was in top-10 at early layers
        final_tokens = response_logits.argmax(dim=-1)
        early_topk = early_logits.topk(k=10, dim=-1).indices
        in_topk = (early_topk == final_tokens.unsqueeze(-1)).any(dim=-1)
        lci = 1 - in_topk.float().mean()  # Higher = later commitment
        signals['lci_cand'] = float(lci.cpu())

        # VarLogP - variance of log probability across layers
        log_probs_per_layer = []
        for layer_idx in range(0, n_layers, max(1, n_layers // 8)):  # Sample 8 layers
            h = hidden_states[layer_idx][0, prompt_len:].to(lm_head_dtype)
            with torch.no_grad():
                layer_logits = self.model.lm_head(h)
            layer_probs = torch.softmax(layer_logits, dim=-1)
            # Get log prob of final token
            log_p = torch.log(layer_probs.gather(-1, final_tokens.unsqueeze(-1)).squeeze(-1) + EPS)
            log_probs_per_layer.append(log_p)

        log_probs_stack = torch.stack(log_probs_per_layer, dim=0)
        var_logp = log_probs_stack.var(dim=0).mean()
        signals['var_logp_cand'] = float(torch.clamp(var_logp / 10, 0, 1).cpu())  # Normalize

        # === Hidden State Family ===
        # Hidden norm (last layer)
        last_hidden = hidden_states[-1][0, prompt_len:].float()
        hidden_norm = torch.norm(last_hidden, dim=-1)
        # Normalize by sqrt(dim)
        hidden_norm_normalized = hidden_norm / np.sqrt(last_hidden.shape[-1])
        signals['hidden_norm'] = float(hidden_norm_normalized.mean().cpu())

        # Hidden variance across layers
        hidden_per_layer = []
        for layer_idx in range(0, n_layers, max(1, n_layers // 8)):
            h = hidden_states[layer_idx][0, prompt_len:].float()
            hidden_per_layer.append(h)
        hidden_stack = torch.stack(hidden_per_layer, dim=0)
        hidden_var = hidden_stack.var(dim=0).mean()
        signals['hidden_var'] = float(torch.clamp(hidden_var / 100, 0, 1).cpu())  # Normalize

        # === Context Grounding Family ===
        signals['context_grounding'] = self._compute_context_grounding(
            hidden_states, prompt_len
        )
        signals['attention_grounding'] = 0.5  # Placeholder - would need attention weights

        # === SOTA Signals ===
        # EigenScore (simplified - spectral entropy of response hidden states)
        signals['eigenscore'] = self._compute_eigenscore(hidden_states, prompt_len)

        # ISE (Internal Semantic Entropy) - simplified
        signals['ise'] = self._compute_ise(hidden_states, prompt_len)

        # LSD (Latent-Output Divergence)
        signals['lsd'] = self._compute_lsd(hidden_states, prompt_len, final_tokens)

        # === Self-Normalizing ===
        # p_conflict = JSD / ln(2)
        signals['p_conflict'] = float(torch.clamp(jsd / np.log(2), 0, 1).mean().cpu())

        # p_uncertain = rank-normalized entropy
        signals['p_uncertain'] = signals['entropy']  # Already normalized

        return signals

    def _compute_context_grounding(
        self,
        hidden_states: List[torch.Tensor],
        prompt_len: int,
    ) -> float:
        """Compute context grounding score."""
        n_layers = len(hidden_states)
        mid_layer = n_layers // 2

        hidden = hidden_states[mid_layer][0].float()
        context_hidden = hidden[:prompt_len]
        response_hidden = hidden[prompt_len:]

        if response_hidden.shape[0] < 1 or context_hidden.shape[0] < 2:
            return 0.5

        # Compute context basis via SVD
        context_mean = context_hidden.mean(dim=0, keepdim=True)
        context_centered = context_hidden - context_mean

        try:
            U, S, Vh = torch.linalg.svd(context_centered, full_matrices=False)
            total_var = (S ** 2).sum()
            cumvar = (S ** 2).cumsum(dim=0) / (total_var + EPS)
            k = int((cumvar < 0.95).sum()) + 1
            k = min(k, len(S), context_hidden.shape[0])

            if k < 10:
                # Low-rank: max cosine similarity
                context_norm = context_hidden / (torch.norm(context_hidden, dim=-1, keepdim=True) + EPS)
                response_norm = response_hidden / (torch.norm(response_hidden, dim=-1, keepdim=True) + EPS)
                similarity = response_norm @ context_norm.T
                max_sim, _ = similarity.max(dim=-1)
                grounding = max_sim.clamp(0, 1).mean()
            else:
                # High-rank: SVD projection
                basis = Vh[:k]
                response_centered = response_hidden - context_mean
                coefficients = response_centered @ basis.T
                projection = coefficients @ basis
                response_norms = torch.norm(response_centered, dim=-1)
                projection_norms = torch.norm(projection, dim=-1)
                grounding = (projection_norms / (response_norms + EPS)).clamp(0, 1).mean()

            return float((1 - grounding).cpu())  # Risk = 1 - grounding
        except:
            return 0.5

    def _compute_eigenscore(
        self,
        hidden_states: List[torch.Tensor],
        prompt_len: int,
    ) -> float:
        """Compute EigenScore (spectral entropy of trajectory)."""
        mid_layer = len(hidden_states) // 2
        hidden = hidden_states[mid_layer][0, prompt_len:].float()

        if hidden.shape[0] < 3:
            return 0.5

        # Compute covariance matrix
        hidden_centered = hidden - hidden.mean(dim=0)
        cov = hidden_centered.T @ hidden_centered / (hidden.shape[0] - 1)

        try:
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=EPS)
            probs = eigenvalues / eigenvalues.sum()
            entropy = -(probs * torch.log(probs + EPS)).sum()
            max_entropy = np.log(min(hidden.shape))
            return float(torch.clamp(entropy / max_entropy, 0, 1).cpu())
        except:
            return 0.5

    def _compute_ise(
        self,
        hidden_states: List[torch.Tensor],
        prompt_len: int,
    ) -> float:
        """Compute Internal Semantic Entropy (layer disagreement)."""
        n_layers = len(hidden_states)
        # Use last 50% of layers
        layer_indices = list(range(n_layers // 2, n_layers, max(1, n_layers // 8)))

        predictions = []
        lm_head_dtype = self.model.lm_head.weight.dtype
        for layer_idx in layer_indices:
            h = hidden_states[layer_idx][0, prompt_len:].to(lm_head_dtype)
            with torch.no_grad():
                logits = self.model.lm_head(h)
            pred = logits.argmax(dim=-1)
            predictions.append(pred)

        if len(predictions) < 2:
            return 0.5

        # Measure disagreement
        predictions = torch.stack(predictions, dim=0)  # [n_layers, seq]
        disagreement = (predictions != predictions[0]).float().mean()
        return float(disagreement.cpu())

    def _compute_lsd(
        self,
        hidden_states: List[torch.Tensor],
        prompt_len: int,
        final_tokens: torch.Tensor,
    ) -> float:
        """Compute Latent-Output Divergence."""
        last_hidden = hidden_states[-1][0, prompt_len:].float()

        # Get token embeddings
        token_embeddings = self.model.lm_head.weight[final_tokens].float()

        # Cosine similarity
        hidden_norm = last_hidden / (torch.norm(last_hidden, dim=-1, keepdim=True) + EPS)
        embed_norm = token_embeddings / (torch.norm(token_embeddings, dim=-1, keepdim=True) + EPS)

        cos_sim = (hidden_norm * embed_norm).sum(dim=-1)
        lod = 1 - cos_sim  # Divergence

        return float(lod.mean().cpu())

    def _signal_names(self) -> List[str]:
        """List of all signal names."""
        return [
            'entropy', 'inv_margin', 'target_prob',
            'jsd_cand', 'lci_cand', 'var_logp_cand',
            'hidden_norm', 'hidden_var',
            'context_grounding', 'attention_grounding',
            'eigenscore', 'ise', 'lsd',
            'p_conflict', 'p_uncertain',
        ]


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dataset_examples(dataset_name: str, num_examples: int = 200) -> List[Dict]:
    """Load examples from a dataset."""
    random.seed(42)

    if dataset_name == "ragtruth_qa":
        return _load_ragtruth(num_examples, task_filter="QA")
    elif dataset_name == "ragtruth_summ":
        return _load_ragtruth(num_examples, task_filter="Summary")
    elif dataset_name == "halueval_qa":
        return _load_halueval("qa_samples", num_examples)
    elif dataset_name == "halueval_summ":
        return _load_halueval("summarization_samples", num_examples)
    elif dataset_name == "halueval_dialogue":
        return _load_halueval("dialogue_samples", num_examples)
    elif dataset_name == "truthfulqa":
        return _load_truthfulqa(num_examples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_ragtruth(num_examples: int, task_filter: str = None) -> List[Dict]:
    """Load RAGTruth dataset."""
    base_path = Path(__file__).parent.parent / "data" / "ragtruth_repo" / "dataset"

    sources = {}
    with open(base_path / "source_info.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sources[d["source_id"]] = d

    examples = []
    with open(base_path / "response.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] != "test":
                continue

            source_id = d["source_id"]
            if source_id not in sources:
                continue

            source = sources[source_id]
            task_type = source.get("task_type", "")

            if task_filter and task_filter not in task_type:
                continue

            source_info = source.get("source_info", "")
            if isinstance(source_info, dict):
                context = f"Question: {source_info.get('question', '')}\n\nPassages: {source_info.get('passages', '')}"
            else:
                context = source_info

            examples.append({
                "context": context[:4000],
                "response": d["response"],
                "label": 1 if d["labels"] else 0,
            })

    random.shuffle(examples)
    return examples[:num_examples]


def _load_halueval(split: str, num_examples: int) -> List[Dict]:
    """Load HaluEval dataset."""
    dataset = load_dataset("pminervini/HaluEval", split, split="data")

    if split == "qa_samples":
        def get_data(ex):
            return ex['knowledge'], ex['answer'], 1 if ex['hallucination'] == 'yes' else 0
    elif split == "summarization_samples":
        def get_data(ex):
            return ex['document'][:4000], ex['summary'], 1 if ex['hallucination'] == 'yes' else 0
    elif split == "dialogue_samples":
        def get_data(ex):
            return ex['dialogue_history'], ex['response'], 1 if ex['hallucination'] == 'yes' else 0

    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    examples = []
    for idx in indices:
        context, response, label = get_data(dataset[idx])
        examples.append({
            "context": context,
            "response": response,
            "label": label,
        })
    return examples


def _load_truthfulqa(num_examples: int) -> List[Dict]:
    """Load TruthfulQA dataset."""
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    examples = []
    for idx in indices:
        ex = dataset[idx]
        context = ex["question"]

        # Randomly select correct or incorrect answer
        if random.random() < 0.5 and ex["correct_answers"]:
            response = random.choice(ex["correct_answers"])
            label = 0
        elif ex["incorrect_answers"]:
            response = random.choice(ex["incorrect_answers"])
            label = 1
        else:
            response = ex.get("best_answer", "")
            label = 0

        examples.append({
            "context": context,
            "response": response,
            "label": label,
        })
    return examples


# =============================================================================
# Experiment Functions
# =============================================================================

def run_experiment_1_signal_matrix(
    model,
    tokenizer,
    signal_computer: SignalComputer,
    datasets: List[str],
    num_examples: int = 200,
) -> Dict[str, Dict[str, DatasetResult]]:
    """
    Experiment 1: Full Signal Matrix Evaluation.

    Evaluate all signals on all datasets with length baseline.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Full Signal Matrix Evaluation")
    print("="*70)

    results = {}

    for dataset_name in datasets:
        print(f"\n--- Evaluating on {dataset_name} ---")

        examples = load_dataset_examples(dataset_name, num_examples)
        if not examples:
            print(f"  Skipping {dataset_name}: no examples")
            continue

        # Collect signals for all examples
        all_signals = {name: [] for name in signal_computer._signal_names()}
        all_labels = []
        all_lengths = []

        for i, ex in enumerate(examples):
            if not ex["context"] or not ex["response"]:
                continue

            # Tokenize
            prompt = f"Context: {ex['context']}\n\nResponse: "
            full_text = prompt + ex["response"]

            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
            prompt_inputs = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_inputs.input_ids.shape[1]

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            logits = outputs.logits[0]
            response_len = inputs['input_ids'].shape[1] - prompt_len

            # Compute signals
            signals = signal_computer.compute_all_signals(
                hidden_states, logits, prompt_len
            )

            for name, score in signals.items():
                all_signals[name].append(score)
            all_labels.append(ex["label"])
            all_lengths.append(response_len)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(examples)} examples")

        # Compute metrics for each signal
        labels = np.array(all_labels)
        lengths = np.array(all_lengths)

        # Length baseline AUROC
        try:
            length_auroc = roc_auc_score(labels, lengths)
        except:
            length_auroc = 0.5

        for signal_name, scores in all_signals.items():
            scores = np.array(scores)

            try:
                raw_auroc = roc_auc_score(labels, scores)
            except:
                raw_auroc = 0.5

            try:
                corr, _ = pearsonr(scores, lengths)
            except:
                corr = 0.0

            pos_mean = float(scores[labels == 1].mean()) if labels.sum() > 0 else 0
            neg_mean = float(scores[labels == 0].mean()) if (1 - labels).sum() > 0 else 0

            result = DatasetResult(
                signal_name=signal_name,
                dataset_name=dataset_name,
                raw_auroc=float(raw_auroc),
                length_auroc=float(length_auroc),
                delta_auroc=float(raw_auroc - length_auroc),
                signal_length_corr=float(corr),
                pos_mean=pos_mean,
                neg_mean=neg_mean,
                n_examples=len(scores),
            )

            if signal_name not in results:
                results[signal_name] = {}
            results[signal_name][dataset_name] = result

        print(f"  Completed {dataset_name}: {len(labels)} examples, length AUROC = {length_auroc:.3f}")

    return results


def run_experiment_3_correlation_matrix(
    model,
    tokenizer,
    signal_computer: SignalComputer,
    dataset_name: str,
    num_examples: int = 200,
) -> Tuple[np.ndarray, List[str]]:
    """
    Experiment 3: Pairwise Signal Correlation Matrix.

    Compute correlations between all signal pairs.
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT 3: Correlation Matrix on {dataset_name}")
    print("="*70)

    examples = load_dataset_examples(dataset_name, num_examples)

    signal_names = signal_computer._signal_names()
    all_signals = {name: [] for name in signal_names}

    for i, ex in enumerate(examples):
        if not ex["context"] or not ex["response"]:
            continue

        prompt = f"Context: {ex['context']}\n\nResponse: "
        full_text = prompt + ex["response"]

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        signals = signal_computer.compute_all_signals(
            outputs.hidden_states, outputs.logits[0], prompt_len
        )

        for name, score in signals.items():
            all_signals[name].append(score)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(examples)} examples")

    # Build correlation matrix
    n_signals = len(signal_names)
    corr_matrix = np.zeros((n_signals, n_signals))

    for i, name_i in enumerate(signal_names):
        for j, name_j in enumerate(signal_names):
            try:
                corr, _ = pearsonr(all_signals[name_i], all_signals[name_j])
                corr_matrix[i, j] = corr
            except:
                corr_matrix[i, j] = 0.0

    return corr_matrix, signal_names


def run_experiment_5_combination_ablation(
    signal_matrix: Dict[str, Dict[str, DatasetResult]],
    datasets: List[str],
) -> Dict[str, Dict]:
    """
    Experiment 5: Noisy-OR Combination Ablation.

    Test different signal subsets with Noisy-OR fusion.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: Combination Ablation")
    print("="*70)

    # Define signal subsets to test
    subsets = {
        "prob_only": ("entropy", "inv_margin"),
        "dynamics_only": ("jsd_cand", "lci_cand", "var_logp_cand"),
        "grounding_only": ("context_grounding",),
        "hidden_only": ("hidden_norm", "hidden_var"),
        "cross_family_1": ("jsd_cand", "entropy", "context_grounding"),
        "cross_family_2": ("entropy", "hidden_var", "context_grounding"),
        "cross_family_3": ("jsd_cand", "hidden_norm", "context_grounding"),
        "contrastive_1": ("entropy", "context_grounding"),
        "contrastive_2": ("jsd_cand", "hidden_var"),
        "all_signals": tuple(signal_matrix.keys()),
    }

    results = {}

    for subset_name, signal_subset in subsets.items():
        subset_results = {"signals": signal_subset, "datasets": {}}

        for dataset in datasets:
            # Compute Noisy-OR combination of delta_auroc
            # (This is a proxy - real combination would need per-example scores)
            deltas = []
            for signal_name in signal_subset:
                if signal_name in signal_matrix and dataset in signal_matrix[signal_name]:
                    deltas.append(signal_matrix[signal_name][dataset].delta_auroc)

            if deltas:
                # Noisy-OR: P(any) = 1 - prod(1 - p_i)
                # For AUROCs centered at 0.5, shift to [0, 1]
                noisy_or = 1 - np.prod([1 - max(0, d + 0.5) for d in deltas])
                avg_delta = np.mean(deltas)
                max_delta = np.max(deltas)
            else:
                noisy_or = 0.5
                avg_delta = 0.0
                max_delta = 0.0

            subset_results["datasets"][dataset] = {
                "noisy_or": float(noisy_or),
                "avg_delta": float(avg_delta),
                "max_delta": float(max_delta),
            }

        # Average across datasets
        avg_noisy_or = float(np.mean([d["noisy_or"] for d in subset_results["datasets"].values()]))
        avg_delta_all = float(np.mean([d["avg_delta"] for d in subset_results["datasets"].values()]))
        subset_results["avg_noisy_or"] = avg_noisy_or
        subset_results["avg_delta"] = avg_delta_all

        results[subset_name] = subset_results
        print(f"  {subset_name}: avg_delta = {avg_delta_all:+.4f}")

    return results


def run_experiment_8_residualized(
    model,
    tokenizer,
    signal_computer: SignalComputer,
    dataset_name: str,
    num_examples: int = 200,
) -> Dict[str, Dict]:
    """
    Experiment 8: Length-Residualized Evaluation.

    Evaluate signals after regressing out length.
    """
    print("\n" + "="*70)
    print(f"EXPERIMENT 8: Residualized Evaluation on {dataset_name}")
    print("="*70)

    examples = load_dataset_examples(dataset_name, num_examples)

    signal_names = signal_computer._signal_names()
    all_signals = {name: [] for name in signal_names}
    all_labels = []
    all_lengths = []

    for i, ex in enumerate(examples):
        if not ex["context"] or not ex["response"]:
            continue

        prompt = f"Context: {ex['context']}\n\nResponse: "
        full_text = prompt + ex["response"]

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        response_len = inputs['input_ids'].shape[1] - prompt_len
        signals = signal_computer.compute_all_signals(
            outputs.hidden_states, outputs.logits[0], prompt_len
        )

        for name, score in signals.items():
            all_signals[name].append(score)
        all_labels.append(ex["label"])
        all_lengths.append(response_len)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(examples)} examples")

    labels = np.array(all_labels)
    lengths = np.array(all_lengths).reshape(-1, 1)

    results = {}

    for signal_name, scores in all_signals.items():
        scores = np.array(scores)

        # Raw AUROC
        try:
            raw_auroc = roc_auc_score(labels, scores)
        except:
            raw_auroc = 0.5

        # Residualize
        lr = LinearRegression()
        lr.fit(lengths, scores)
        residuals = scores - lr.predict(lengths)

        # Residualized AUROC
        try:
            residual_auroc = roc_auc_score(labels, residuals)
        except:
            residual_auroc = 0.5

        results[signal_name] = {
            "raw_auroc": float(raw_auroc),
            "residual_auroc": float(residual_auroc),
            "auroc_drop": float(raw_auroc - residual_auroc),
            "length_coef": float(lr.coef_[0]),
        }

        print(f"  {signal_name}: raw={raw_auroc:.3f}, residual={residual_auroc:.3f}, drop={raw_auroc - residual_auroc:+.3f}")

    return results


def compute_consistency_metrics(
    signal_matrix: Dict[str, Dict[str, DatasetResult]],
    datasets: List[str],
) -> Dict[str, Dict]:
    """Compute consistency metrics across datasets."""
    results = {}

    for signal_name, dataset_results in signal_matrix.items():
        deltas = []
        for dataset in datasets:
            if dataset in dataset_results:
                deltas.append(dataset_results[dataset].delta_auroc)

        if deltas:
            results[signal_name] = {
                "mean_delta": float(np.mean(deltas)),
                "std_delta": float(np.std(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "cv": float(np.std(deltas) / (np.abs(np.mean(deltas)) + EPS)),
                "n_datasets": len(deltas),
            }

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--output-dir", default="results/generalization_experiments")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    signal_computer = SignalComputer(model, tokenizer)

    # Define datasets
    datasets = [
        "ragtruth_qa",
        "ragtruth_summ",
        "halueval_qa",
        "halueval_summ",
        "halueval_dialogue",
        "truthfulqa",
    ]

    # Phase 1: Signal Matrix
    if args.phase == 0 or args.phase == 1:
        signal_matrix = run_experiment_1_signal_matrix(
            model, tokenizer, signal_computer, datasets, args.num_examples
        )

        # Save results
        matrix_results = {}
        for signal, ds_results in signal_matrix.items():
            matrix_results[signal] = {ds: asdict(r) for ds, r in ds_results.items()}

        with open(output_dir / "signal_matrix.json", "w") as f:
            json.dump(matrix_results, f, indent=2)

        # Consistency metrics
        consistency = compute_consistency_metrics(signal_matrix, datasets)
        with open(output_dir / "consistency.json", "w") as f:
            json.dump(consistency, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("SIGNAL RANKING (by mean delta above length baseline)")
        print("="*70)
        sorted_signals = sorted(consistency.items(), key=lambda x: x[1]["mean_delta"], reverse=True)
        print(f"{'Signal':<20} {'Mean Δ':>10} {'Std Δ':>10} {'Min Δ':>10}")
        print("-" * 50)
        for signal, metrics in sorted_signals:
            print(f"{signal:<20} {metrics['mean_delta']:>+10.4f} {metrics['std_delta']:>10.4f} {metrics['min_delta']:>+10.4f}")

    # Phase 2: Correlation Matrix (on RAGTruth QA)
    if args.phase == 0 or args.phase == 2:
        corr_matrix, signal_names = run_experiment_3_correlation_matrix(
            model, tokenizer, signal_computer, "ragtruth_qa", args.num_examples
        )

        with open(output_dir / "correlation_matrix.json", "w") as f:
            json.dump({
                "signal_names": signal_names,
                "correlation_matrix": corr_matrix.tolist(),
            }, f, indent=2)

        # Find complementary pairs (r < 0.3)
        print("\n" + "="*70)
        print("COMPLEMENTARY SIGNAL PAIRS (r < 0.3)")
        print("="*70)
        for i in range(len(signal_names)):
            for j in range(i+1, len(signal_names)):
                if abs(corr_matrix[i, j]) < 0.3:
                    print(f"  {signal_names[i]} <-> {signal_names[j]}: r = {corr_matrix[i, j]:.3f}")

    # Phase 3: Combination Ablation
    if args.phase == 0 or args.phase == 3:
        if 'signal_matrix' not in dir():
            with open(output_dir / "signal_matrix.json") as f:
                matrix_data = json.load(f)
            signal_matrix = {}
            for signal, ds_results in matrix_data.items():
                signal_matrix[signal] = {
                    ds: DatasetResult(**r) for ds, r in ds_results.items()
                }

        combinations = run_experiment_5_combination_ablation(signal_matrix, datasets)
        with open(output_dir / "combination_ablation.json", "w") as f:
            json.dump(combinations, f, indent=2)

    # Phase 4: Residualized Evaluation
    if args.phase == 0 or args.phase == 4:
        residualized = run_experiment_8_residualized(
            model, tokenizer, signal_computer, "ragtruth_qa", args.num_examples
        )
        with open(output_dir / "length_control.json", "w") as f:
            json.dump(residualized, f, indent=2)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
