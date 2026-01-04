#!/bin/bash
# ============================================================================
# AG-SAR ICML/NeurIPS Staged Evaluation Suite
# ============================================================================
#
# Runs all 6 stages with validation checkpoints:
#   Stage 1: Metric validation (GPT-2)
#   Stage 2: Full metrics (Llama-8B)
#   Stage 3: Architecture sweep (Qwen, Mistral, Mixtral)
#   Stage 4: Scale test (70B)
#   Stage 5: Full baselines
#   Stage 6: RAGTruth generalization
#
# Prerequisites:
#   pip install -e ".[all]"
#
# Usage:
#   ./run_icml_staged.sh           # Run all stages
#   ./run_icml_staged.sh stage1    # Run specific stage
#   ./run_icml_staged.sh --dry-run # Print configs without running
#
# ============================================================================

set -e

# H100 NVLink optimizations
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=24  # Half of 52 vCPUs for optimal threading
export HF_TOKEN="${HF_TOKEN:-}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}[PASS] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
}

print_error() {
    echo -e "${RED}[FAIL] $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking dependencies..."

    python -c "import ag_sar" 2>/dev/null || {
        print_error "ag_sar not installed. Run: pip install -e ."
        exit 1
    }

    python -c "import sklearn" 2>/dev/null || {
        print_error "scikit-learn not installed. Run: pip install -e '.[eval]'"
        exit 1
    }

    python -c "import datasets" 2>/dev/null || {
        print_error "datasets not installed. Run: pip install -e '.[eval]'"
        exit 1
    }

    python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
        print_error "CUDA not available. Check your GPU setup."
        exit 1
    }

    print_success "All dependencies satisfied."
}

# Validate stage output
validate_stage() {
    local stage_name=$1
    local jsonl_path=$2

    print_header "Validating $stage_name..."

    python -c "
from experiments.core.validation import StageValidator
import json
import sys

validator = StageValidator()
results = []
with open('$jsonl_path', 'r') as f:
    for line in f:
        line = line.strip()
        if line and 'score' in line and 'label' in line:
            try:
                data = json.loads(line)
                if 'score' in data and 'label' in data:
                    results.append(data)
            except:
                continue

if not results:
    print('ERROR: No valid results found')
    sys.exit(1)

try:
    result = validator.validate_stage(results)
    print(f'Validation passed: {result.details}')
except ValueError as e:
    print(f'Validation failed: {e}')
    sys.exit(1)
" || {
        print_error "Stage validation failed for $stage_name"
        return 1
    }

    print_success "$stage_name validation passed"
    return 0
}

# Run single stage
run_stage() {
    local stage_name=$1
    local config_path=$2
    local dry_run=$3

    print_header "Running $stage_name"
    echo "Config: $config_path"

    if [[ ! -f "$config_path" ]]; then
        print_error "Config not found: $config_path"
        return 1
    fi

    if [[ "$dry_run" == "true" ]]; then
        python -m experiments.main --config "$config_path" --dry-run
    else
        python -m experiments.main --config "$config_path"

        # Find the latest JSONL file for validation
        local output_dir=$(grep -A1 "output_dir:" "$config_path" | tail -1 | tr -d ' "' | cut -d: -f2)
        if [[ -z "$output_dir" ]]; then
            output_dir="results"
        fi

        # Validate if JSONL exists
        local latest_jsonl=$(ls -t "$output_dir"/*.jsonl 2>/dev/null | head -1)
        if [[ -n "$latest_jsonl" ]]; then
            validate_stage "$stage_name" "$latest_jsonl" || {
                print_error "$stage_name validation failed. Stopping pipeline."
                exit 1
            }
        fi
    fi
}

# Main
main() {
    local dry_run="false"
    local stages=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            stage[1-6]*)
                stages+=("$1")
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                echo "Usage: $0 [--dry-run] [stage1] [stage2] [stage3] [stage4] [stage5] [stage6]"
                exit 1
                ;;
        esac
    done

    # Default to all stages if none specified
    if [[ ${#stages[@]} -eq 0 ]]; then
        stages=(
            "stage1"
            "stage2"
            "stage3"
            "stage4"
            "stage5"
            "stage6"
        )
    fi

    check_dependencies

    print_header "AG-SAR ICML/NeurIPS Staged Evaluation"
    echo "Stages to run: ${stages[*]}"
    echo "Dry run: $dry_run"
    echo ""

    # Create results directory
    mkdir -p results

    # Run stages
    local failed=()
    for stage in "${stages[@]}"; do
        case $stage in
            stage1)
                run_stage "Stage 1: Metric Validation (GPT-2)" \
                    "experiments/configs/stage1_metric_validation.yaml" "$dry_run" || failed+=("stage1")
                ;;
            stage2)
                run_stage "Stage 2: Full Metrics (Llama-8B)" \
                    "experiments/configs/stage2_full_metrics.yaml" "$dry_run" || failed+=("stage2")
                ;;
            stage3)
                # Stage 3 runs multiple models
                run_stage "Stage 3a: Qwen2.5-32B" \
                    "experiments/configs/stage3_architecture.yaml" "$dry_run" || failed+=("stage3a")
                run_stage "Stage 3b: Mistral-Nemo" \
                    "experiments/configs/stage3_mistral_nemo.yaml" "$dry_run" || failed+=("stage3b")
                run_stage "Stage 3c: Mixtral MoE" \
                    "experiments/configs/stage3_mixtral_moe.yaml" "$dry_run" || failed+=("stage3c")
                ;;
            stage4)
                run_stage "Stage 4: Scale Test (70B)" \
                    "experiments/configs/stage4_scale.yaml" "$dry_run" || failed+=("stage4")
                ;;
            stage5)
                run_stage "Stage 5: Full Baselines" \
                    "experiments/configs/stage5_baselines.yaml" "$dry_run" || failed+=("stage5")
                ;;
            stage6)
                run_stage "Stage 6: RAGTruth Generalization" \
                    "experiments/configs/stage6_ragtruth.yaml" "$dry_run" || failed+=("stage6")
                ;;
        esac
    done

    # Summary
    print_header "Evaluation Summary"
    echo "Total stages: ${#stages[@]}"
    echo "Failed: ${#failed[@]}"

    if [[ ${#failed[@]} -gt 0 ]]; then
        print_error "Failed stages: ${failed[*]}"
        exit 1
    else
        print_success "All stages completed successfully!"
        echo ""
        echo "Results saved to: results/"
        echo "  - results/stage*/  (per-stage outputs)"
        echo ""
        echo "Next steps:"
        echo "  1. Review JSONL files for per-sample scores"
        echo "  2. Check summary.json for aggregate metrics"
        echo "  3. Run: python experiments/analysis/generate_paper_plots.py"
    fi
}

main "$@"
