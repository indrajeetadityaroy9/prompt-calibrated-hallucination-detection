#!/bin/bash
export HF_TOKEN="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"
cd /lambda/nfs/lambda-cloud-data/AG-SAR

echo "============================================================"
echo "AG-SAR GOLDEN RUN - NeurIPS 2025 Final Artifact"
echo "Started: $(date)"
echo "============================================================"

python -m experiments.main \
    --config experiments/configs/benchmarks/sota_final_golden_run.yaml \
    --seed 42 \
    --deterministic \
    2>&1 | tee results/final_submission/golden_run.log

echo "============================================================"
echo "GOLDEN RUN COMPLETE"
echo "Finished: $(date)"
echo "============================================================"
