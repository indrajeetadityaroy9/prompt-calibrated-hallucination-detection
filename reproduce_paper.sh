#!/bin/bash
set -e  # Exit immediately if a command fails

# AG-SAR Reproduction Script (ICML 2025)
# This script reproduces the experimental results presented in Tables 1 and 2.
#
# Usage:
#   ./reproduce_paper.sh               # Full reproduction
#   ./reproduce_paper.sh --skip-install  # Skip venv setup (assumes dependencies installed)
#   HF_TOKEN=your_token ./reproduce_paper.sh  # With HuggingFace token for gated models
#
# Requirements:
#   - Python 3.10+
#   - CUDA 12+ with compatible GPU (40GB+ VRAM recommended)
#   - ~50GB disk space for models and datasets
#
# Environment Variables:
#   HF_TOKEN: HuggingFace token for gated models (Llama, etc.)
#   AG_SAR_USE_TORCH: Set to "1" to force PyTorch backend (skip Triton)

echo "========================================================"
echo "   AG-SAR: Universal Hallucination Sensor (v8.0 SOTA)   "
echo "========================================================"

# Parse arguments
SKIP_INSTALL=false
for arg in "$@"; do
    case $arg in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
    esac
done

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models (Llama, Mistral) may fail."
    echo "         Set with: export HF_TOKEN=your_huggingface_token"
    echo ""
fi

# Virtual Environment Setup
if [ "$SKIP_INSTALL" = false ]; then
    echo "[0/7] Setting up Virtual Environment..."
    if [ -d "venv" ]; then
        echo "   Found existing virtual environment, activating..."
        source venv/bin/activate
    else
        echo "   No virtual environment found, creating one..."
        python3 -m venv venv
        source venv/bin/activate
        echo "   Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e ".[all]"
    fi
    echo "   Virtual environment active: $VIRTUAL_ENV"
else
    echo "[0/7] Skipping virtual environment setup (--skip-install)"
    # Try to activate existing venv if present
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || true
    fi
fi

# 1. Smoke Test (Pre-flight verification)
echo ""
echo "[1/7] Running Smoke Test..."
python smoke_test.py --skip-inference || {
    echo "ERROR: Smoke test failed. Please fix installation issues."
    exit 1
}

# 2. Environment Check
echo ""
echo "[2/7] Verifying Environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ag_sar; print(f'AG-SAR Package: v{ag_sar.__version__}')"

# Check CUDA
python -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Experiments will be slow on CPU.')
else:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)')
"

# 3. Data Verification
echo ""
echo "[3/7] Verifying Datasets..."
python -m experiments.scripts.verify_datasets || {
    echo "WARNING: Dataset verification had issues. Continuing anyway..."
}

# 4. Table 1: Main SOTA Comparison (HaluEval QA + Summarization)
# Compares AG-SAR against Predictive Entropy and SelfCheckGPT
echo ""
echo "[4/7] Running Table 1 Benchmark (SOTA Comparison)..."
python -m experiments.main --config experiments/configs/benchmarks/main_sota.yaml --deterministic

# 5. Table 2: Generalization (RAGTruth)
# Demonstrates performance on Out-Of-Distribution RAG data
echo ""
echo "[5/7] Running Table 2 Benchmark (Generalization)..."
python -m experiments.main --config experiments/configs/generalization/cross_dataset.yaml --deterministic

# 6. Ablation Studies (confirming component contributions)
echo ""
echo "[6/7] Running Ablation Studies..."
echo "   - Run 1: No Unified Gating (v3.1 Baseline)"
python -m experiments.main --config experiments/configs/ablations/no_gating.yaml --deterministic
echo "   - Run 2: No Semantic Dispersion (v7.0 Baseline)"
python -m experiments.main --config experiments/configs/ablations/no_dispersion.yaml --deterministic

# 7. Artifact Generation
echo ""
echo "[7/7] Generating Results Summary..."
# Find the latest results directory
LATEST_RESULTS=$(find results -type d -mindepth 1 -maxdepth 1 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_RESULTS" ]; then
    python -m experiments.scripts.print_h2h_table --results-dir "$LATEST_RESULTS" --format markdown
else
    echo "   No results directory found to generate table."
fi

echo ""
echo "========================================================"
echo "   Reproduction Complete. Results saved to results/     "
echo "========================================================"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Check individual experiment logs for detailed metrics"
echo "  3. Compare with paper Tables 1 and 2"
