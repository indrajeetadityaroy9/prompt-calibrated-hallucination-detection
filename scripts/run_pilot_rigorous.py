"""
Rigorous evaluation on RAGTruth dataset.

Features:
- Train/Test Split (50/50 by default for pilot).
- Threshold Calibration on Train set.
- Evaluation on Test set using FIXED threshold.
- Reporting of AUROC, F1 (at fixed threshold), and AURC (Risk-Coverage).
- Latency measurement.
"""

import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from ag_sar.engine import AGSAR
from ag_sar.config import DetectorConfig
from ag_sar.evaluation.metrics import (
    compute_metrics,
    compute_aurc,
    compute_e_aurc,
    compute_risk_coverage
)


def evaluate_dataset(
    detector: HallucinationDetector,
    dataset: List[Dict],
    desc: str = "Evaluating"
) -> Dict[str, List[float]]:
    """
    Run detection on a dataset.
    Returns dict of scores and labels.
    """
    results = {
        "risks": [],
        "labels": [],
        "latencies": []
    }
    
    for example in tqdm(dataset, desc=desc):
        context = example['context']
        question = example['query']
        # We ignore the provided response and generate our own for "Generation Mode" evaluation
        # Or we use forced decoding.
        # For SOTA SE/EigenScore, we generally test on *generation* (Open-Ended).
        # RAGTruth labels are for specific responses.
        # If we generate new text, we don't have labels!
        # CRITICAL: For SE/EigenScore, we must use FORCED DECODING if we want to check against RAGTruth spans.
        # BUT SE is inherently a generation-based metric (sampling).
        # Comparison:
        # 1. EigenScore: Can run in Forced Decoding.
        # 2. Semantic Entropy: Requires Generation (Sampling).
        
        # Dilemma: We cannot validate SE against RAGTruth labels unless we manually label the new generations.
        # Workaround: Use RAGTruth's "Gold" response as the "Prompt" for SE? No, SE samples.
        
        # Compromise for Pilot:
        # For EigenScore: Use Forced Decoding on RAGTruth responses.
        # For SE: We technically can't evaluate accuracy against RAGTruth labels without generating new labels.
        # HOWEVER, we can use the "Model Confidence" proxy. 
        # Or, we assume the provided response is one of the samples?
        
        # Real SOTA Evaluation usually involves:
        # 1. Generate Answer.
        # 2. GPT-4 Judge checks if Answer is Hallucination.
        # 3. Compute SE/EigenScore.
        # 4. Correlate.
        
        # Since we don't have a GPT-4 judge loop here, we will stick to EIGENSCORE (Forced Decoding) 
        # for direct comparison with RAGTruth labels. 
        # SE will be run purely for latency/correlation checks in a separate script, 
        # or we accept that SE evaluation here is impossible without an oracle.
        
        # Wait, if we use Forced Decoding, we can measure "Internal Confusion" on the *hallucinated* tokens.
        # So we will use Forced Decoding Evaluator logic.
        
        # BUT: detector.generate_with_detection is for generation.
        # We need the ForcedDecodingEvaluator from modes.py.
        pass

    return results

def run_rigorous_pilot(
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_examples: int = 60,
    method: str = "eigenscore", # 'eigenscore', 'se', 'ise', 'lsd'
    zero_shot: bool = False,
):
    # Load RAGTruth
    print(f"Loading RAGTruth...")
    dataset = load_dataset("wandb/RAGTruth-processed", split="train")
    
    # Filter
    qa_data = [x for x in dataset if x['task_type'] == "QA"]
    
    # Check for labels
    labeled_data = []
    for x in qa_data:
        try:
            lbls = json.loads(x['hallucination_labels'])
            # Binary label: is there any non-implicit hallucination?
            is_hallucination = any(
                not l.get('implicit_true', False) and 
                ('Conflict' in l.get('label_type', '') or 'Baseless' in l.get('label_type', ''))
                for l in lbls
            )
            labeled_data.append({
                "context": x['context'],
                "query": x['query'],
                "output": x['output'], # The text to force-decode
                "label": 1 if is_hallucination else 0
            })
        except:
            continue
            
    # Subsample
    import random
    random.seed(42)
    if len(labeled_data) > num_examples:
        labeled_data = random.sample(labeled_data, num_examples)
        
    # Split or Zero-Shot
    if zero_shot:
        print(">>> ZERO-SHOT MODE: Using full dataset for testing. No threshold calibration.")
        train_data = []
        test_data = labeled_data
        optimal_threshold = 0.5 # Arbitrary default for F1
    else:
        # Split
        train_data, test_data = train_test_split(labeled_data, test_size=0.5, random_state=42)
        print(f"Data: {len(train_data)} Train, {len(test_data)} Test")
    
    # Load Model
    print(f"Loading Model: {model_name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto", 
        local_files_only=True
    )
    
    # Config
    config = DetectorConfig(
        eigenscore_enabled=(method == "eigenscore"),
        semantic_entropy_enabled=(method == "se"),
        ise_enabled=(method == "ise"),
        lsd_enabled=(method == "lsd"),
        num_samples=3 if method == "se" else 1,
    )
    
    from ag_sar.engine import AGSAR
    from ag_sar.evaluation.modes import ForcedDecodingEvaluator

    detector = AGSAR(model, tokenizer, config)
    
    if method == "se":
        print("Semantic Entropy requires generation. Skipping strict RAGTruth accuracy check.")
        return

    evaluator = ForcedDecodingEvaluator(
        model=model,
        tokenizer=tokenizer,
        config=config,
        signal_computer=detector.signal_computer,
        candidate_manager=detector.candidate_manager
    )
    
    # 1. Train Phase (Calibrate)
    if not zero_shot:
        print(">>> Training Phase (Calibration)")
        train_scores = []
        train_labels = []
        
        for ex in tqdm(train_data, desc="Calibrating"):
            token_signals = evaluator.evaluate(
                context=ex['context'],
                question=ex['query'],
                response=ex['output']
            )
            
            # Aggregate based on method
            scores = []
            for ts in token_signals:
                if method == "eigenscore": val = ts.eigenscore
                elif method == "ise": val = ts.ise
                elif method == "lsd": val = ts.lsd
                else: val = 0.0
                
                if val is not None: scores.append(val)

            if scores:
                risk = max(scores) # Conservative: max internal confusion
            else:
                risk = 0.0
                
            train_scores.append(risk)
            train_labels.append(ex['label'])
            
        # Find Optimal Threshold
        metrics_train = compute_metrics(train_scores, train_labels)
        optimal_threshold = metrics_train.threshold
        print(f"Optimal Threshold (Train): {optimal_threshold:.4f}")
        print(f"Train AUROC: {metrics_train.auroc:.4f}")
    
    # 2. Test Phase
    print(">>> Test Phase")
    test_scores = []
    test_labels = []
    start_time = time.time()
    
    for ex in tqdm(test_data, desc="Testing"):
        token_signals = evaluator.evaluate(
            context=ex['context'],
            question=ex['query'],
            response=ex['output']
        )
        scores = []
        for ts in token_signals:
            if method == "eigenscore": val = ts.eigenscore
            elif method == "ise": val = ts.ise
            elif method == "lsd": val = ts.lsd
            else: val = 0.0
            
            if val is not None: scores.append(val)

        risk = max(scores) if scores else 0.0
        test_scores.append(risk)
        test_labels.append(ex['label'])
        
    elapsed = time.time() - start_time
    latency_ms = (elapsed / len(test_data)) * 1000
    
    # Compute Metrics using FIXED (or default) threshold
    metrics_test = compute_metrics(test_scores, test_labels, threshold=optimal_threshold)
    aurc = compute_aurc(test_scores, test_labels)
    e_aurc = compute_e_aurc(test_scores, test_labels)
    
    print("="*40)
    print(f"RESULTS ({method.upper()})")
    print("="*40)
    print(f"Test AUROC: {metrics_test.auroc:.4f}")
    print(f"Test F1:    {metrics_test.f1:.4f} (at thresh={optimal_threshold:.4f})")
    print(f"Test AURC:  {aurc:.4f} (Risk-Coverage)")
    print(f"Test E-AURC:{e_aurc:.4f}")
    print(f"Latency:    {latency_ms:.1f} ms/example")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="eigenscore")
    parser.add_argument("--zero-shot", action="store_true", help="Run in true zero-shot mode (no calibration)")
    args = parser.parse_args()
    
    run_rigorous_pilot(method=args.method, zero_shot=args.zero_shot)
