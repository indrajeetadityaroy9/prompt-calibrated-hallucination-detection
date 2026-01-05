#!/usr/bin/env python3
"""
Step 2: Train the JEPA Latent Predictor.

Trains a lightweight MLP to predict h_{t+1} from h_t using the
embedding pairs extracted from WikiText-2.

The trained predictor learns the "Physics of Truth" - how thought
vectors should evolve during coherent text generation. Deviations
from this learned trajectory indicate hallucinations.

Usage:
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python scripts/train_jepa_predictor.py

Expected Output:
    - Initial Loss: ~0.5 - 1.0
    - Final Loss: ~0.005 - 0.02
    - Saved model: data/models/jepa_predictor.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import glob
import os
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ag_sar.modeling.predictor import JepaPredictor

# --- Config ---
BATCH_SIZE = 512
LEARNING_RATE = 3e-4
EPOCHS = 10
DATA_DIR = "data/training/embeddings"
SAVE_PATH = "data/models/jepa_predictor.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(data_dir):
    print(f"Loading training chunks from {data_dir}...")
    chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))

    if not chunk_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    all_X = []
    all_Y = []

    for f in tqdm(chunk_files, desc="Loading chunks"):
        data = torch.load(f)
        all_X.append(data["X"])
        all_Y.append(data["Y"])

    # Concatenate all chunks (3GB is fine for RAM)
    X = torch.cat(all_X, dim=0).to(torch.float32)  # Convert bfloat16 to float32
    Y = torch.cat(all_Y, dim=0).to(torch.float32)

    print(f"Data Loaded. Shape: {X.shape} | dtype: {X.dtype}")
    return TensorDataset(X, Y)


def train():
    # 1. Prepare Data
    dataset = load_data(DATA_DIR)

    # Split 90/10 Train/Val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4
    )

    # 2. Initialize Model
    print(f"Initializing JEPA Predictor on {DEVICE}...")
    model = JepaPredictor().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    criterion = nn.MSELoss()

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    print(f"   Architecture: {model.input_dim} -> {model.hidden_dim} -> {model.input_dim}")

    # 3. Training Loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    print(f"Starting Training ({EPOCHS} epochs, {len(train_loader)} batches/epoch)...")

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss_accum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            pred_y = model(batch_x)
            loss = criterion(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_accum += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.5f}"})

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                pred_y = model(batch_x)
                loss = criterion(pred_y, batch_y)
                val_loss_accum += loss.item()

        avg_val_loss = val_loss_accum / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"   Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"   Saved New Best Model to {SAVE_PATH}")

    print(f"Training Complete. Best Val Loss: {best_val_loss:.5f}")
    print(f"Model saved to: {SAVE_PATH}")

    # Print interpretation
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print(f"Val MSE Loss: {best_val_loss:.5f}")
    print(f"This means: Average squared error per dimension = {best_val_loss:.5f}")
    print(f"RMSE per dimension: {best_val_loss**0.5:.4f}")
    print("\nDrift Detection Threshold Guidance:")
    print(f"  - Normal transitions: MSE < {best_val_loss * 2:.4f}")
    print(f"  - Suspicious (1-2x): MSE in [{best_val_loss:.4f}, {best_val_loss * 2:.4f}]")
    print(f"  - Hallucination (>2x): MSE > {best_val_loss * 2:.4f}")


if __name__ == "__main__":
    train()
