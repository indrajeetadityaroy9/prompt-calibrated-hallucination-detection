#!/bin/bash
# AG-SAR Optimized Evaluation Script
# Hardware: 1× H100 80GB
# Estimated Runtime: ~4-5 hours (down from ~38 hours)
#
# Optimizations Applied:
#   1. SelfCheck limited to exp4 only (saves ~31 hours)
#   2. 70B skipped (requires 2× H100)
#   3. Mixtral run with reduced batch size (fits in 80GB)
#   4. Redundant configs consolidated
#   5. Validation stages use smaller samples

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Activated virtual environment${NC}"
fi

# H100 optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

print_header() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
}

print_time() {
    echo -e "${YELLOW}  ⏱  $(date '+%H:%M:%S') - $1${NC}"
}

run_config() {
    local config=$1
    local desc=$2
    local est_time=$3

    print_header "$desc"
    echo -e "  Config: $config"
    echo -e "  Est. Time: $est_time"
    print_time "Starting..."

    if python -m experiments.main --config "$config"; then
        echo -e "${GREEN}  ✓ Completed${NC}"
    else
        echo -e "${RED}  ✗ Failed${NC}"
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Pre-flight Checks
# ═══════════════════════════════════════════════════════════════
print_header "PHASE 0: Pre-flight Checks"

echo "  Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo "  Verifying datasets..."
python scripts/verify_datasets.py || { echo -e "${RED}Dataset verification failed${NC}"; exit 1; }
echo -e "${GREEN}  ✓ All checks passed${NC}"

START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Quick Validation (GPT-2) - ~5 min
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/burnin_smoke_test.yaml" \
    "PHASE 1: Pipeline Smoke Test (GPT-2)" \
    "~2 min"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Main Benchmark (Llama-8B) - ~45 min
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/exp1_halueval_qa.yaml" \
    "PHASE 2: Main Benchmark - HaluEval QA (Llama-8B, 2500 samples)" \
    "~45 min"

# ═══════════════════════════════════════════════════════════════
# PHASE 3: SOTA Comparison with SelfCheck - ~20 min (50 samples)
# Purpose: Get ONE data point for Latency vs Accuracy Pareto chart
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/exp4_baseline_comparison.yaml" \
    "PHASE 3: SOTA Comparison (includes SelfCheck, 50 samples)" \
    "~20 min"

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Distribution Shift (RAGTruth) - ~20 min
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/stage6_ragtruth.yaml" \
    "PHASE 4: RAGTruth Generalization (1000 samples)" \
    "~20 min"

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Architecture Sweep - ~30 min
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/stage3_mistral_nemo.yaml" \
    "PHASE 5a: Mistral-Nemo Architecture (500 samples)" \
    "~15 min"

# Note: Qwen-32B needs ~65GB, should fit
run_config "experiments/configs/stage3_architecture.yaml" \
    "PHASE 5b: Qwen-32B Architecture + 4× Scaling (500 samples)" \
    "~15 min"

run_config "experiments/configs/stage3_qwen_moe.yaml" \
    "PHASE 5c: Qwen-MoE Sparse Architecture (500 samples)" \
    "~10 min"

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Ablation Study - ~15 min
# ═══════════════════════════════════════════════════════════════
run_config "experiments/configs/exp5_ablation.yaml" \
    "PHASE 6: Ablation Study (1000 samples)" \
    "~15 min"

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

print_header "EVALUATION COMPLETE"
echo -e "  Total Time: ${HOURS}h ${MINUTES}m"
echo -e "  Results: results/"
echo ""
echo -e "  ${YELLOW}SKIPPED (requires 2× H100):${NC}"
echo -e "    - stage4_scale.yaml (Llama-70B)"
echo -e "    - stage3_mixtral_moe.yaml (Mixtral-8x7B, ~95GB VRAM)"
echo ""
echo -e "${GREEN}  Run './run_optimized.sh --full' on 2× H100 for complete suite${NC}"
