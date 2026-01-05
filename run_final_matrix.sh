#!/bin/bash
# AG-SAR Final Evaluation Campaign (2x H100)
#
# Split Evaluation Strategy:
#   - 01_main_sota: Fast methods (N=1000) for statistical power
#   - 01b_selfcheck: SelfCheck comparison (N=100) for Pareto frontier
#   - 02-06: Architecture and generalization tests
#
# Estimated runtime: ~2-3 hours

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
}

print_phase() {
    echo ""
    echo -e "${BLUE}--------------------------------------------${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}--------------------------------------------${NC}"
}

# Activate virtual environment
print_header "AG-SAR Final Evaluation Campaign"

if [[ -f "venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# H100 Optimization Flags
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=24
export TOKENIZERS_PARALLELISM=false

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs available: $GPU_COUNT"

START_TIME=$(date +%s)

# ============================================================
# Phase 1: Main SOTA (Fast methods, N=1000)
# ============================================================
print_phase "Phase 1: Main SOTA - Fast Methods (N=1000)"
echo "Methods: AG-SAR, LogProb, Entropy, LLMCheck-Attn, LLMCheck-Hidden"
echo "Estimated: ~25 minutes"

CUDA_VISIBLE_DEVICES=0 python -m experiments.main \
    --config experiments/configs/01_main_sota.yaml

# ============================================================
# Phase 2: Sampling Methods - SKIPPED (run separately)
# ============================================================
# print_phase "Phase 2: Sampling Methods Comparison (N=100)"
# echo "Methods: AG-SAR, SelfCheck, EigenScore"
# CUDA_VISIBLE_DEVICES=0 python -m experiments.main \
#     --config experiments/configs/01b_sota_selfcheck.yaml

# ============================================================
# Phase 2: Generalization (RAGTruth, N=1000)
# ============================================================
print_phase "Phase 2: Generalization - RAGTruth (N=1000)"
echo "Estimated: ~15 minutes"

CUDA_VISIBLE_DEVICES=0 python -m experiments.main \
    --config experiments/configs/03_generalization.yaml

# ============================================================
# Phase 3: Long Context (Summarization, N=500)
# ============================================================
print_phase "Phase 3: Long Context - Summarization (N=500)"
echo "Estimated: ~10 minutes"

CUDA_VISIBLE_DEVICES=0 python -m experiments.main \
    --config experiments/configs/04_long_context.yaml

# ============================================================
# Phase 4: Architecture Tests (Qwen 32B, N=500)
# ============================================================
print_phase "Phase 4: Qwen Architecture (N=500)"
echo "Estimated: ~20 minutes"

CUDA_VISIBLE_DEVICES=0 python -m experiments.main \
    --config experiments/configs/05_qwen_arch.yaml

# ============================================================
# Phase 5: Scaling Law (70B, N=500)
# ============================================================
print_phase "Phase 5: Scaling Law - 70B (N=500)"
echo "Requires both GPUs (balanced device_map)"
echo "Estimated: ~45 minutes"

python -m experiments.main \
    --config experiments/configs/02_scaling_law.yaml

# ============================================================
# Phase 6: MoE Architecture (Mixtral, N=500)
# ============================================================
print_phase "Phase 6: MoE Architecture - Mixtral (N=500)"
echo "Requires both GPUs (balanced device_map)"
echo "Estimated: ~45 minutes"

python -m experiments.main \
    --config experiments/configs/06_moe_arch.yaml

# ============================================================
# Summary
# ============================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

print_header "Campaign Complete"
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results:"
ls -la results/
echo ""
echo "Result directories:"
for dir in results/*/; do
    if [[ -d "$dir" ]]; then
        count=$(find "$dir" -name "*.jsonl" | wc -l)
        echo "  $dir ($count files)"
    fi
done
