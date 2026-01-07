#!/usr/bin/env python3
"""
Step 1: The "Black Box" Recorder - JEPA Latent Predictor Data Extraction.

Extracts the "Physics of Truth" from the model's internal states.
Records how the model's thoughts *should* move when generating coherent,
factual text (using WikiText-2 as ground truth for language dynamics).

Usage:
    python scripts/extract_embeddings.py --limit 200000
    python scripts/extract_embeddings.py --model meta-llama/Llama-3.1-8B-Instruct --stride 1
"""

import torch
import os
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np

# --- Configuration ---
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER_IDX = 16  # Middle layer (Abstract Semantic Space)
STRIDE = 1      # Predict next token embedding (k=1)
BUFFER_SIZE = 10_000  # Save to disk every N samples to save RAM
OUTPUT_DIR = "data/training/embeddings"

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def extract_and_save(model_name, dataset_name, limit, stride):
    print(f"Initializing Extraction Pipeline")
    print(f"   Model: {model_name}")
    print(f"   Layer: {LAYER_IDX} | Stride: +{stride}")
    print(f"   Dataset: {dataset_name}")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # 1. Load Trusted Data (WikiText-2 for General Language Physics)
    # We use 'validation' split to avoid training on massive train sets for this extraction
    ds = load_dataset(dataset_name, "wikitext-2-raw-v1", split="validation")

    # Buffer storage
    input_states = []  # h_t
    target_states = [] # h_{t+k}
    saved_chunks = 0
    total_samples = 0

    print(f"Processing {len(ds)} documents...")

    for i, row in enumerate(tqdm(ds)):
        if limit and total_samples >= limit:
            break

        text = row['text']
        if len(text) < 50: continue # Skip short fragments

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        # Skip if too short for stride
        if input_ids.shape[1] <= stride + 1:
            continue

        # 2. Forward Pass (No Gradients)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # 3. Extract Hidden Layer (Layer 16)
        # Shape: [Batch, Seq, Dim]
        hidden_state = outputs.hidden_states[LAYER_IDX].squeeze(0).cpu() # Move to CPU immediately

        # 4. Create Pairs (Input -> Target)
        # Input: All tokens except last 'stride'
        # Target: All tokens shifted by 'stride'
        # Example: "The cat sat on mat" (Stride=1)
        # Pair 0: "The" -> "cat"
        # Pair 1: "cat" -> "sat"

        # Slice: [0 : -stride] -> [stride : end]
        current_batch_inputs = hidden_state[:-stride, :]
        current_batch_targets = hidden_state[stride:, :]

        input_states.append(current_batch_inputs)
        target_states.append(current_batch_targets)

        # Update counts
        total_samples += current_batch_inputs.shape[0]

        # 5. Flush to Disk (Prevent RAM Explosion)
        if total_samples >= BUFFER_SIZE * (saved_chunks + 1):
            _save_chunk(input_states, target_states, saved_chunks)
            input_states = []
            target_states = []
            saved_chunks += 1

    # Final Save
    if input_states:
        _save_chunk(input_states, target_states, saved_chunks)

    print(f"Extraction Complete. Processed {total_samples} transitions.")

def _save_chunk(inputs, targets, chunk_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Concatenate lists of tensors
    X = torch.cat(inputs, dim=0)
    Y = torch.cat(targets, dim=0)

    filename = os.path.join(OUTPUT_DIR, f"chunk_{chunk_id}.pt")
    torch.save({"X": X, "Y": Y}, filename)
    print(f"   Saved chunk {chunk_id} ({X.shape[0]} pairs) to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--limit", type=int, default=100_000, help="Max transitions to record")
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    extract_and_save(args.model, args.dataset, args.limit, args.stride)
