#!/bin/bash
# Reproduce all AG-SAR paper experiments
#
# This script runs the canonical experiments from the paper:
#   00: CI smoke test (fast validation)
#   01: Table 1 - SOTA comparison on HaluEval QA
#   02: Figure 2 - Scaling to Llama-3.1-70B
#   03: Table 2 - RAGTruth generalization
#   04: Discussion - MoE robustness (Mixtral)
#
# Prerequisites:
#   pip install -e ".[all]"
#
# Usage:
#   ./reproduce_paper.sh           # Run all experiments
#   ./reproduce_paper.sh 01        # Run specific experiment
#   ./reproduce_paper.sh --dry-run # Print configs without running

set -e

# H100 NVLink optimizations
export NCCL_P2P_LEVEL=NVL
export OMP_NUM_THREADS=16
export HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}============================================${NC}"
    echo ""
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking dependencies..."

    python -c "import ag_sar" 2>/dev/null || {
        print_error "ag_sar not installed. Run: pip install -e ."
        exit 1
    }

    python -c "import pydantic" 2>/dev/null || {
        print_error "pydantic not installed. Run: pip install -e '.[dev]'"
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

    echo "All dependencies satisfied."
}

# Verify datasets before running experiments
verify_datasets() {
    print_header "Verifying Datasets (Hard Gate)"

    python scripts/verify_datasets.py
    if [ $? -ne 0 ]; then
        print_error "Dataset verification failed. Fix issues before running experiments."
        exit 1
    fi

    echo -e "${GREEN}Dataset verification passed.${NC}"
}

# Run latency verification
run_latency_check() {
    local dry_run=$1

    print_header "Verifying Zero-Latency Constraint"

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY RUN] Would run: python -m experiments.analysis.benchmark_latency --model gpt2 --seq-len 128"
    else
        python -m experiments.analysis.benchmark_latency --model gpt2 --seq-len 128
    fi
}

# Run single experiment
run_experiment() {
    local exp_name=$1
    local config_path="experiments/configs/${exp_name}.yaml"
    local dry_run=$2
    
    if [[ ! -f "$config_path" ]]; then
        print_error "Config not found: $config_path"
        return 1
    fi
    
    print_header "Running $exp_name"
    echo "Config: $config_path"
    
    if [[ "$dry_run" == "true" ]]; then
        python -m experiments.main --config "$config_path" --dry-run
    else
        python -m experiments.main --config "$config_path"
    fi
}

# Main
main() {
    local dry_run="false"
    local experiments=()
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run="true"
                shift
                ;;
            0[0-4]*)
                experiments+=("$1")
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                echo "Usage: $0 [--dry-run] [00] [01] [02] [03] [04]"
                exit 1
                ;;
        esac
    done
    
    # Default to all experiments if none specified
    if [[ ${#experiments[@]} -eq 0 ]]; then
        experiments=(
            "00_ci_smoke_test"
            "01_main_sota"
            "02_scaling_law"
            "03_generalization"
            "04_moe_robustness"
        )
    else
        # Expand short names
        expanded=()
        for exp in "${experiments[@]}"; do
            case $exp in
                00) expanded+=("00_ci_smoke_test") ;;
                01) expanded+=("01_main_sota") ;;
                02) expanded+=("02_scaling_law") ;;
                03) expanded+=("03_generalization") ;;
                04) expanded+=("04_moe_robustness") ;;
                *) expanded+=("$exp") ;;
            esac
        done
        experiments=("${expanded[@]}")
    fi
    
    check_dependencies

    # Verify datasets (hard gate)
    verify_datasets

    # Run latency verification first
    run_latency_check "$dry_run"

    print_header "AG-SAR Paper Reproduction"
    echo "Experiments to run: ${experiments[*]}"
    echo "Dry run: $dry_run"
    
    # Create results directory
    mkdir -p results
    
    # Run experiments
    local failed=()
    for exp in "${experiments[@]}"; do
        if ! run_experiment "$exp" "$dry_run"; then
            failed+=("$exp")
        fi
    done
    
    # Summary
    print_header "Summary"
    echo "Total experiments: ${#experiments[@]}"
    echo "Failed: ${#failed[@]}"
    
    if [[ ${#failed[@]} -gt 0 ]]; then
        print_error "Failed experiments: ${failed[*]}"
        exit 1
    else
        echo -e "${GREEN}All experiments completed successfully!${NC}"
        echo ""
        echo "Results saved to: results/"
        echo "  - results/0*/*.jsonl (per-sample scores)"
        echo "  - results/0*/summary.json (metrics + CI)"
    fi
}

main "$@"
