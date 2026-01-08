#!/bin/bash
set -e  # Exit immediately if a command fails

# =============================================================================
# AG-SAR Paper Reproduction Script (ICML/NeurIPS 2025)
# =============================================================================
#
# This script reproduces all experimental results from the paper:
#   - Table 1: AG-SAR vs Fast Baselines (LogProb, Entropy, LLM-Check)
#   - Table 2: AG-SAR vs SOTA (SelfCheckGPT, EigenScore)
#   - Table 3: Ablation Studies (component contributions)
#   - Figure 3: Mechanism Visualization (anatomy plots)
#
# Usage:
#   ./reproduce_paper.sh                  # Full reproduction (~4-6 hours)
#   ./reproduce_paper.sh --skip-install   # Skip venv setup
#   ./reproduce_paper.sh --table1-only    # Only Table 1 (fast, ~30 min)
#   ./reproduce_paper.sh --table2-only    # Only Table 2 (slow, ~2 hours)
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
echo "   AG-SAR: Universal Hallucination Sensor              "
echo "   Paper Reproduction Script                           "
echo "========================================================"

# Parse arguments
SKIP_INSTALL=false
TABLE1_ONLY=false
TABLE2_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --table1-only)
            TABLE1_ONLY=true
            shift
            ;;
        --table2-only)
            TABLE2_ONLY=true
            shift
            ;;
    esac
done

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# H100 Optimization Flags
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Gated models (Llama, Mistral) may fail."
    echo "         Set with: export HF_TOKEN=your_huggingface_token"
    echo ""
fi

# =============================================================================
# Step 0: Virtual Environment Setup
# =============================================================================
if [ "$SKIP_INSTALL" = false ]; then
    echo "[0/7] Setting up Virtual Environment..."
    if [ -d "venv" ]; then
        echo "   Found existing virtual environment, activating..."
        source venv/bin/activate
    else
        echo "   Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        echo "   Installing dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    fi
    echo "   Virtual environment active: $VIRTUAL_ENV"
else
    echo "[0/7] Skipping virtual environment setup (--skip-install)"
    if [ -d "venv" ]; then
        source venv/bin/activate 2>/dev/null || true
    fi
fi

# =============================================================================
# Step 1: Smoke Test (Pre-flight verification)
# =============================================================================
echo ""
echo "[1/7] Running Smoke Test..."
python smoke_test.py --skip-inference || {
    echo "ERROR: Smoke test failed. Please fix installation issues."
    exit 1
}

# =============================================================================
# Step 2: Environment Verification
# =============================================================================
echo ""
echo "[2/7] Verifying Environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import ag_sar; print(f'AG-SAR Package: v{ag_sar.__version__}')"

python -c "
import torch
if not torch.cuda.is_available():
    print('WARNING: CUDA not available. Experiments will be slow on CPU.')
else:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)')
"

# =============================================================================
# Step 3: Dataset Verification
# =============================================================================
echo ""
echo "[3/7] Verifying Datasets..."
python -m experiments.scripts.verify_datasets || {
    echo "WARNING: Dataset verification had issues. Continuing anyway..."
}

START_TIME=$(date +%s)

# =============================================================================
# Step 4: Table 1 - AG-SAR vs Fast Baselines
# =============================================================================
# Methods: AG-SAR, LogProb, Entropy, LLM-Check (Attention + Hidden)
# Datasets: HaluEval QA, RAGTruth, HaluEval Summ, TruthfulQA
# Samples: 500 per dataset (statistical significance > 95%)
# Runtime: ~30 minutes
# =============================================================================
if [ "$TABLE2_ONLY" = false ]; then
    echo ""
    echo "[4/7] Running Table 1: AG-SAR vs Fast Baselines..."
    echo "   Config: experiments/configs/benchmarks/reproduce_main_results.yaml"
    echo "   Methods: AG-SAR, LogProb, Entropy, LLM-Check variants"
    echo "   Estimated runtime: ~30 minutes"
    python -m experiments.main \
        --config experiments/configs/benchmarks/reproduce_main_results.yaml \
        --deterministic
else
    echo ""
    echo "[4/7] Skipping Table 1 (--table2-only)"
fi

# =============================================================================
# Step 5: Table 2 - AG-SAR vs SOTA (Sampling Methods)
# =============================================================================
# Methods: AG-SAR, SelfCheckGPT (5x sampling), EigenScore, LLM-Check
# This is the "gold standard" comparison for NeurIPS/ICML reviewers
# Runtime: ~2 hours (SelfCheck dominates due to 5x forward passes)
# =============================================================================
if [ "$TABLE1_ONLY" = false ]; then
    echo ""
    echo "[5/7] Running Table 2: AG-SAR vs SOTA (SelfCheck, EigenScore)..."
    echo "   Config: experiments/configs/benchmarks/sota_core_claims.yaml"
    echo "   Methods: AG-SAR, SelfCheckGPT, EigenScore, LLM-Check"
    echo "   NOTE: This step takes ~2 hours due to sampling-based methods"
    python -m experiments.main \
        --config experiments/configs/benchmarks/sota_core_claims.yaml \
        --deterministic
else
    echo ""
    echo "[5/7] Skipping Table 2 (--table1-only)"
fi

# =============================================================================
# Step 6: Table 3 - Ablation Studies
# =============================================================================
# Confirms contribution of each component:
#   - No Gating: Disables Unified Gating (pure Authority Flow)
#   - No Dispersion: Disables Semantic Dispersion
# =============================================================================
if [ "$TABLE1_ONLY" = false ] && [ "$TABLE2_ONLY" = false ]; then
    echo ""
    echo "[6/7] Running Table 3: Ablation Studies..."

    echo "   Run 1/2: No Unified Gating (Pure Authority Flow)"
    python -m experiments.main \
        --config experiments/configs/ablations/no_gating.yaml \
        --deterministic

    echo "   Run 2/2: No Semantic Dispersion"
    python -m experiments.main \
        --config experiments/configs/ablations/no_dispersion.yaml \
        --deterministic
else
    echo ""
    echo "[6/7] Skipping Ablations (partial run mode)"
fi

# =============================================================================
# Step 7: Results Summary
# =============================================================================
echo ""
echo "[7/7] Generating Results Summary..."
LATEST_RESULTS=$(find results -type d -mindepth 1 -maxdepth 1 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_RESULTS" ]; then
    python -m experiments.scripts.print_h2h_table --results-dir "$LATEST_RESULTS" --format markdown
else
    echo "   No results directory found to generate table."
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "========================================================"
echo "   Reproduction Complete                               "
echo "========================================================"
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to: results/"
echo ""
echo "Next steps:"
echo "  1. Review results in results/ directory"
echo "  2. Check individual experiment logs for detailed metrics"
echo "  3. Compare with paper Tables 1, 2, and 3"
echo ""
echo "To generate paper figures:"
echo "  python paper/figures/generate_auroc_curves.py"
echo "  python paper/tables/generate_main_results.py"
