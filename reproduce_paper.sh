#!/bin/bash
# Reproduce all AG-SAR paper experiments
# 
# This script runs all 5 canonical experiments from the paper:
#   exp1: HaluEval QA (Section 4.1)
#   exp2: HaluEval Summarization (Section 4.1)
#   exp3: RAGTruth Generalization (Section 4.2)
#   exp4: Full Baseline Comparison (Section 4.3)
#   exp5: Ablation Study (Section 5)
#
# Prerequisites:
#   pip install -e ".[all]"
#
# Usage:
#   ./reproduce_paper.sh           # Run all experiments
#   ./reproduce_paper.sh exp1      # Run specific experiment
#   ./reproduce_paper.sh --dry-run # Print configs without running

set -e

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
            exp[1-5]*)
                experiments+=("$1")
                shift
                ;;
            *)
                print_error "Unknown argument: $1"
                echo "Usage: $0 [--dry-run] [exp1] [exp2] [exp3] [exp4] [exp5]"
                exit 1
                ;;
        esac
    done
    
    # Default to all experiments if none specified
    if [[ ${#experiments[@]} -eq 0 ]]; then
        experiments=(
            "exp1_halueval_qa"
            "exp2_halueval_summ"
            "exp3_ragtruth"
            "exp4_baseline_comparison"
            "exp5_ablation"
        )
    else
        # Expand short names
        expanded=()
        for exp in "${experiments[@]}"; do
            case $exp in
                exp1) expanded+=("exp1_halueval_qa") ;;
                exp2) expanded+=("exp2_halueval_summ") ;;
                exp3) expanded+=("exp3_ragtruth") ;;
                exp4) expanded+=("exp4_baseline_comparison") ;;
                exp5) expanded+=("exp5_ablation") ;;
                *) expanded+=("$exp") ;;
            esac
        done
        experiments=("${expanded[@]}")
    fi
    
    check_dependencies
    
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
        echo "  - results/exp*/*.jsonl (per-sample scores)"
        echo "  - results/exp*/summary.json (metrics + CI)"
    fi
}

main "$@"
