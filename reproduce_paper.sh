#!/bin/bash
set -e  # Exit immediately if a command fails

# AG-SAR Reproduction Script (ICML 2025)
# This script reproduces the experimental results presented in Tables 1 and 2.

echo "========================================================"
echo "   AG-SAR: Universal Hallucination Sensor (v8.0 SOTA)   "
echo "========================================================"

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Virtual Environment Setup
echo "[0/6] Setting up Virtual Environment..."
if [ -d "venv" ]; then
    echo "   Found existing virtual environment, activating..."
    source venv/bin/activate
else
    echo "   No virtual environment found, creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "   Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e ".[all]"
fi
echo "   Virtual environment active: $VIRTUAL_ENV"

# 1. Environment Check
echo "[1/6] Verifying Environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ag_sar; print(f'AG-SAR Package: v{ag_sar.__version__}')"

# 2. Data Verification
echo "[2/6] Verifying Datasets..."
python scripts/verify_datasets.py

# 3. Table 1: Main SOTA Comparison (HaluEval QA + Summarization)
# Compares AG-SAR against Predictive Entropy and SelfCheckGPT
echo "[3/6] Running Table 1 Benchmark (SOTA Comparison)..."
python -m experiments.main --config experiments/configs/01_main_sota.yaml

# 4. Table 2: Generalization (RAGTruth)
# Demonstrates performance on Out-Of-Distribution RAG data
echo "[4/6] Running Table 2 Benchmark (Generalization)..."
python -m experiments.main --config experiments/configs/03_generalization.yaml

# 5. Ablation Studies (confirming component contributions)
echo "[5/6] Running Ablation Studies..."
echo "   - Run 1: No Unified Gating (v3.1 Baseline)"
python -m experiments.main --config experiments/configs/ablation_no_gating.yaml
echo "   - Run 2: No Semantic Dispersion (v7.0 Baseline)"
python -m experiments.main --config experiments/configs/ablation_no_dispersion.yaml

# 6. Artifact Generation
echo "[6/6] Generating Results Table..."
python scripts/print_h2h_table.py --dir results/ --format latex

echo "========================================================"
echo "   Reproduction Complete. Results saved to results/     "
echo "========================================================"
