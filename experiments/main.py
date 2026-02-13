#!/usr/bin/env python3
"""
Unified Benchmark Runner for ICML Submission.

Executes the evaluation matrix defined in the YAML config.
Architectural maturity:
- Config-driven execution
- Unified interface for diverse datasets
- Automated metric reporting
"""

import argparse
import yaml
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig
from ag_sar.evaluation.modes import ForcedDecodingEvaluator
from ag_sar.evaluation.metrics import compute_metrics, compute_span_metrics


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_name: str, device_map: str = "auto", torch_dtype: str = "float16"):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=getattr(torch, torch_dtype),
    )
    model.eval()
    return model, tokenizer

def load_ragtruth(config: Dict[str, Any]) -> List[Dict]:
    """
    Load RAGTruth dataset for span-level hallucination detection.

    RAGTruth (ACL 2024) provides character-level span annotations for hallucinations
    in RAG-generated responses. Labels include:
    - Conflict: Contradicts retrieved context
    - Baseless: Information not supported by context

    Fields: id, context, query, output, hallucination_labels (JSON spans)
    """
    path = config.get('path', config['name'])
    split = config.get('split', 'test')
    limit = config.get('limit', config.get('max_examples', float('inf')))
    print(f"Loading RAGTruth from {path} ({split})...")

    dataset = load_dataset(path, split=split)

    examples = []
    for ex in dataset:
        if len(examples) >= limit:
            break

        if config.get('task_types') and ex['task_type'] not in config['task_types']:
            continue

        try:
            labels = json.loads(ex['hallucination_labels']) if ex['hallucination_labels'] else []
        except:
            labels = []

        # Parse labels into spans - only count actual hallucinations
        spans = []
        has_hallucination = False
        for l in labels:
            label_type = l.get('label_type', '')
            is_implicit_true = l.get('implicit_true', False)
            # Conflict and Baseless are hallucinations (excluding implicit_true)
            if not is_implicit_true and ('Conflict' in label_type or 'Baseless' in label_type):
                spans.append((l['start'], l['end']))
                has_hallucination = True

        examples.append({
            "id": str(ex['id']),
            "context": ex['context'],
            "question": ex['query'],
            "response": ex['output'],
            "spans": spans,
            "label": 1 if has_hallucination else 0,
            "task_type": ex.get('task_type', ''),
        })

    n_pos = sum(1 for e in examples if e['label'] == 1)
    n_neg = len(examples) - n_pos
    n_spans = sum(len(e['spans']) for e in examples)
    print(f"Loaded {len(examples)} RAGTruth examples (hallucinated={n_pos}, faithful={n_neg}, total_spans={n_spans})")
    return examples

def load_halueval(config: Dict[str, Any]) -> List[Dict]:
    """
    Load HaluEval dataset.

    For ICML evaluation, use '_samples' splits (qa_samples, summarization_samples)
    which have explicit hallucination labels for both positive and negative examples.

    The non-samples splits (qa, summarization) have paired right/hallucinated responses
    which are better suited for contrastive training, not evaluation.
    """
    path = config.get('path', config['name'])
    split_name = config['split']
    print(f"Loading HaluEval from {path} ({split_name})")

    try:
        dataset = load_dataset(path, split_name, split='data')
    except:
        # Fallback for some HF dataset structures
        dataset = load_dataset(path, split=split_name)

    examples = []
    limit = config.get('limit', config.get('max_examples', float('inf')))

    # Check if this is a _samples split (has explicit labels)
    is_samples_split = '_samples' in split_name or 'samples' in split_name

    for idx, ex in enumerate(dataset):
        if len(examples) >= limit:
            break

        if is_samples_split:
            # _samples splits have: answer/summary + hallucination label
            # Handle different field names for qa_samples vs summarization_samples
            if 'question' in ex:
                # qa_samples format
                response = ex.get('answer', '')
                question = ex.get('question', '')
                context = ex.get('knowledge', '')
            else:
                # summarization_samples format
                response = ex.get('summary', '')
                question = "Summarize the document."
                context = ex.get('document', '')

            # Parse hallucination label
            label_raw = ex.get('hallucination', ex.get('label', ''))
            if isinstance(label_raw, str):
                label = 1 if label_raw.lower() in ['yes', 'true', '1'] else 0
            else:
                label = int(label_raw) if label_raw else 0

            examples.append({
                "id": str(ex.get('id', idx)),
                "context": context,
                "question": question,
                "response": response,
                "spans": [],
                "label": label
            })
        else:
            # Paired format (qa, summarization): create examples from BOTH responses
            # This gives us balanced positive/negative examples
            if 'question' in ex:
                # qa format
                context = ex.get('knowledge', '')
                question = ex.get('question', '')
                right_response = ex.get('right_answer', '')
                hallucinated_response = ex.get('hallucinated_answer', '')
            else:
                # summarization format
                context = ex.get('document', '')
                question = "Summarize the document."
                right_response = ex.get('right_summary', '')
                hallucinated_response = ex.get('hallucinated_summary', '')

            # Add right (non-hallucinated) example
            if right_response:
                examples.append({
                    "id": f"{idx}_right",
                    "context": context,
                    "question": question,
                    "response": right_response,
                    "spans": [],
                    "label": 0  # Not hallucinated
                })

            # Add hallucinated example
            if hallucinated_response and len(examples) < limit:
                examples.append({
                    "id": f"{idx}_hallucinated",
                    "context": context,
                    "question": question,
                    "response": hallucinated_response,
                    "spans": [],
                    "label": 1  # Hallucinated
                })

    # Log class balance
    n_pos = sum(1 for e in examples if e['label'] == 1)
    n_neg = len(examples) - n_pos
    print(f"Loaded {len(examples)} HaluEval examples (hallucinated={n_pos}, faithful={n_neg})")
    return examples

def load_faitheval(config: Dict[str, Any]) -> List[Dict]:
    """
    DEPRECATED: FaithEval is NOT a hallucination detection dataset.

    FaithEval is a faithfulness evaluation benchmark that tests model BEHAVIOR
    (whether models follow context correctly), not pre-labeled hallucination responses.

    Dataset formats:
    - counterfactual: MC format (question, answer, answerKey, choices, context)
    - unanswerable: QA with acceptable "I don't know" responses
    - inconsistent: QA with (old, new) answer pairs

    For hallucination DETECTION benchmarks, use:
    - HaluEval (qa_samples, summarization_samples)
    - RAGTruth (test split)
    - FAVA (fava-uw/fava-data)
    - TruthfulQA (validation split)

    This loader is kept for backwards compatibility but will log a warning.
    """
    print(
        "FaithEval is NOT a hallucination detection dataset. "
        "It tests model faithfulness BEHAVIOR, not pre-labeled hallucination responses. "
        "Consider using HaluEval, RAGTruth, FAVA, or TruthfulQA instead."
    )

    subset = config.get('subset', config.get('split', 'counterfactual'))
    limit = config.get('limit', config.get('max_examples', float('inf')))

    # Map subset to HuggingFace dataset names
    FAITHEVAL_DATASETS = {
        'counterfactual': 'Salesforce/FaithEval-counterfactual-v1.0',
        'unanswerable': 'Salesforce/FaithEval-unanswerable-v1.0',
        'inconsistent': 'Salesforce/FaithEval-inconsistent-v1.0',
    }

    # Determine which datasets to load
    if subset == 'all':
        datasets_to_load = list(FAITHEVAL_DATASETS.keys())
    else:
        datasets_to_load = [subset]

    examples = []
    for subset_name in datasets_to_load:
        ds_path = FAITHEVAL_DATASETS.get(subset_name)
        if not ds_path:
            print(f"Unknown FaithEval subset: {subset_name}")
            continue

        print(f"Loading FaithEval ({subset_name}) from {ds_path}...")
        try:
            dataset = load_dataset(ds_path, split='test')
        except Exception as e:
            print(f"Failed to load {ds_path}: {e}")
            continue

        for idx, ex in enumerate(dataset):
            if len(examples) >= limit:
                break

            # FaithEval fields vary by subset
            context = ex.get('context', ex.get('passage', ex.get('document', '')))
            question = ex.get('question', ex.get('query', ex.get('prompt', '')))
            response = ex.get('response', ex.get('answer', ex.get('output', '')))

            # Label: For FaithEval, the provided responses are typically unfaithful
            # counterfactual=1 (unfaithful), faithful=0
            label_raw = ex.get('label', ex.get('is_hallucination', None))
            if label_raw is not None:
                if isinstance(label_raw, bool):
                    label = 1 if label_raw else 0
                elif isinstance(label_raw, int):
                    label = 1 if label_raw > 0 else 0
                elif isinstance(label_raw, str):
                    label = 1 if label_raw.lower() in ['true', 'yes', '1', 'unfaithful'] else 0
                else:
                    label = 1
            else:
                # Default: counterfactual/unanswerable/inconsistent are unfaithful
                label = 1

            examples.append({
                "id": f"{subset_name}_{idx}",
                "context": context,
                "question": question,
                "response": response,
                "spans": [],
                "label": label,
                "subset": subset_name,
            })

    n_pos = sum(1 for e in examples if e['label'] == 1)
    n_neg = len(examples) - n_pos
    print(f"Loaded {len(examples)} FaithEval examples (unfaithful={n_pos}, faithful={n_neg})")
    return examples


def load_truthfulqa(config: Dict[str, Any]) -> List[Dict]:
    """
    Load TruthfulQA dataset from HuggingFace.

    TruthfulQA only has a 'validation' split with 817 questions.
    Uses multiple_choice config for MC1 evaluation.

    Creates balanced pairs: one correct answer (label=0) and one incorrect (label=1)
    per question, avoiding artificial dataset inflation.
    """
    limit = config.get('limit', config.get('max_examples', float('inf')))
    print("Loading TruthfulQA from HuggingFace...")

    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    except Exception as e:
        print(f"Failed to load TruthfulQA: {e}")
        return []

    examples = []
    for idx, ex in enumerate(dataset):
        if len(examples) >= limit:
            break

        question = ex.get('question', '')
        mc1_targets = ex.get('mc1_targets', {})
        choices = mc1_targets.get('choices', [])
        labels = mc1_targets.get('labels', [])

        if not choices or not labels:
            continue

        # Find correct and first incorrect answer
        correct_idx = None
        incorrect_idx = None
        for i, label in enumerate(labels):
            if label == 1 and correct_idx is None:
                correct_idx = i
            elif label == 0 and incorrect_idx is None:
                incorrect_idx = i
            if correct_idx is not None and incorrect_idx is not None:
                break

        if correct_idx is None or incorrect_idx is None:
            continue

        # Add correct answer (label=0, faithful/factual)
        examples.append({
            "id": f"tqa_{idx}_correct",
            "context": "",  # TruthfulQA is closed-book
            "question": question,
            "response": choices[correct_idx],
            "spans": [],
            "label": 0,
            "category": ex.get('category', ''),
        })

        if len(examples) >= limit:
            break

        # Add incorrect answer (label=1, hallucination/unfactual)
        examples.append({
            "id": f"tqa_{idx}_incorrect",
            "context": "",
            "question": question,
            "response": choices[incorrect_idx],
            "spans": [],
            "label": 1,
            "category": ex.get('category', ''),
        })

    n_pos = sum(1 for e in examples if e['label'] == 1)
    n_neg = len(examples) - n_pos
    print(f"Loaded {len(examples)} TruthfulQA examples (incorrect={n_pos}, correct={n_neg})")
    return examples


def load_fava(config: Dict[str, Any]) -> List[Dict]:
    """
    Load FAVA dataset for fine-grained hallucination detection.

    FAVA-Bench (arXiv 2024) provides span-level annotations with 6 hallucination types:
    - Factual errors, Unverifiable claims, Subjective statements
    - Invented information, Contradictory statements, Relationship errors

    Uses the processed version (wandb/fava-data-processed) which has:
    - query: The prompt given to the model
    - output: The model's response
    - is_hallucination: Binary label (0 or 1)
    - annotated_text: Span-level hallucination markers

    Reference: https://arxiv.org/abs/2401.06855
    """
    limit = config.get('limit', config.get('max_examples', float('inf')))
    split = config.get('split', 'test')
    print(f"Loading FAVA from HuggingFace (split={split})...")

    try:
        # Use processed version which has simpler format for evaluation
        dataset = load_dataset("wandb/fava-data-processed", split=split)
    except Exception as e:
        print(f"Failed to load FAVA: {e}")
        return []

    examples = []
    for idx, ex in enumerate(dataset):
        if len(examples) >= limit:
            break

        # Extract fields from processed FAVA format
        query = ex.get('query', ex.get('prompt', ''))
        output = ex.get('output', ex.get('response', ''))

        # Binary label - try multiple field names for compatibility
        label = ex.get('is_hallucination', ex.get('has_hallucination', 0))
        if isinstance(label, bool):
            label = 1 if label else 0

        # Annotated text contains span markers (for future span-level evaluation)
        annotated = ex.get('annotated_text', '')

        examples.append({
            "id": f"fava_{idx}",
            "context": ex.get('context', ''),  # Usually empty in FAVA
            "question": query,
            "response": output,
            "spans": [],  # Could parse from annotated_text if needed
            "label": int(label),
            "annotated_text": annotated,
            "model": ex.get('model', ''),
            "subject": ex.get('subject', ''),
        })

    n_pos = sum(1 for e in examples if e['label'] == 1)
    n_neg = len(examples) - n_pos
    print(f"Loaded {len(examples)} FAVA examples (hallucinated={n_pos}, faithful={n_neg})")
    return examples


def run_evaluation(model, tokenizer, method_config: Dict, examples: List[Dict], output_dir: Path, dataset_name: str):
    print(f"Running method: {method_config['name']}")
    
    # Configure Detector
    # Merge defaults with method config
    config_dict = method_config['config']
    
    # Handle specific flags that are not in DetectorConfig (if any)
    # ...
    
    det_config = DetectorConfig(**config_dict)
    
    # Initialize Engine
    engine = AGSAR(model, tokenizer, det_config)
    
    # Initialize Evaluator (Forced Decoding for consistency)
    # Even for HaluEval, we can score the provided response
    evaluator = ForcedDecodingEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=det_config,
        signal_computer=engine.signal_computer,
        candidate_manager=engine.candidate_manager
    )
    
    results = []
    all_scores = []
    all_labels = []
    
    for ex in tqdm(examples, desc=f"Eval {method_config['name']}"):
        try:
            token_signals = evaluator.evaluate(
                context=ex['context'],
                question=ex['question'],
                response=ex['response']
            )
            
            # Simple aggregation for now (can use NoisyOR later)
            token_risks = []
            for ts in token_signals:
                if ts.eigenscore is not None:
                     # Combine EigenScore with local signals
                     # Local = (JSD + LCI + InvMargin)/3
                     local_risk = (ts.jsd_cand + ts.lci_cand + ts.inv_margin) / 3
                     # Global = EigenScore (normalized)
                     global_risk = min(1.0, ts.eigenscore / 10.0)
                     risk = (local_risk + global_risk) / 2
                else:
                    risk = (ts.jsd_cand + ts.lci_cand + ts.inv_margin) / 3
                token_risks.append(risk)
            
            response_risk = max(token_risks) if token_risks else 0.0
            
            results.append({
                "id": ex['id'],
                "response_risk": response_risk,
                "label": ex['label'],
                "token_risks": token_risks,
                # "token_signals": [ts.as_dict() for ts in token_signals] # Too large for JSON usually
            })
            
            all_scores.append(response_risk)
            all_labels.append(ex['label'])
            
        except Exception as e:
            print(f"Error evaluating example {ex['id']}: {e}")
            continue

    # Compute Metrics
    metrics = compute_metrics(all_scores, all_labels)
    print(f"Result {dataset_name} - {method_config['name']}: AUROC={metrics.auroc:.4f}")
    
    # Save Results
    method_name = method_config['name']
    out_file = output_dir / f"{dataset_name}_{method_name}.json"
    with open(out_file, "w") as f:
        json.dump({
            "config": config_dict,
            "metrics": {
                "auroc": metrics.auroc,
                "auprc": metrics.auprc,
            },
            "results": results
        }, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/icml_submission.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Setup Output
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Model
    model, tokenizer = load_model(config['model']['name'])
    
    # Run Matrix
    for dataset_conf in config['datasets']:
        # Match HF path or short name
        ds_name = dataset_conf['name'].lower()

        if 'ragtruth' in ds_name:
            examples = load_ragtruth(dataset_conf)
        elif 'halueval' in ds_name:
            examples = load_halueval(dataset_conf)
        elif 'fava' in ds_name:
            examples = load_fava(dataset_conf)
        elif 'truthful' in ds_name:
            examples = load_truthfulqa(dataset_conf)
        elif 'faitheval' in ds_name:
            # Deprecated - will log warning
            examples = load_faitheval(dataset_conf)
        else:
            print(f"Unknown dataset: {ds_name}")
            continue

        if not examples:
            print(f"No examples loaded for {ds_name}, skipping...")
            continue
            
        # Iterate over methods (dictionary or list)
        methods_config = config['methods']
        if isinstance(methods_config, dict):
            # Format: methods: { method_name: { class: ..., config: ... } }
            for method_name, method_details in methods_config.items():
                # Add name to details for consistency
                method_details_copy = method_details.copy()
                method_details_copy['name'] = method_name
                run_evaluation(
                    model, tokenizer, method_details_copy, examples, output_dir, dataset_conf['name']
                )
        elif isinstance(methods_config, list):
            # Format: methods: [ { name: ..., class: ..., config: ... } ]
            for method_conf in methods_config:
                run_evaluation(
                    model, tokenizer, method_conf, examples, output_dir, dataset_conf['name']
                )

if __name__ == "__main__":
    main()
