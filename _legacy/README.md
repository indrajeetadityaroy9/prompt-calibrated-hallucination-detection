# Archived Experimental Code

This directory contains experimental features from AG-SAR v10.0-v13.0 that are not part of the published paper method (v8.0 SOTA).

## Contents

### modeling/
- `predictor.py` - JEPA Predictor (v11.0, static neural predictor)
- `online_predictor.py` - Online adaptation with Test-Time Training (deprecated)

### measures/
- `symbolic.py` - Entity overlap detection for RAG violations (v13.0 Hybrid Controller)

### calibration/
- `truth_vector.py` - Intrinsic hallucination detection via Truth Vector projection (v10.0)

### generation/
- `guided_decoding.py` - Uncertainty-aware token generation (separate feature)

### scripts/
- `train_jepa_predictor.py` - Train JEPA predictor on normal text corpus
- `test_online_jepa.py` - Test online JEPA test-time training
- `calibrate_truth_vector.py` - Calibrate Truth Vector from fact/counterfact pairs
- `test_hybrid_controller.py` - Test v13.0 hybrid controller integration
- `evaluate_universal_veto.py` - Evaluate universal veto engine
- `interactive_guided.py` - Interactive guided decoding demo
- `test_guided_trap.py` - Trap detection in guided decoding
- `test_guided_harder.py` - Stress test for guided decoding
- `evaluate_generative.py` - Evaluate on generative tasks

### configs/
- `benchmark_layer_drift.yaml` - Layer Drift validation benchmark
- `benchmark_truthfulqa.yaml` - TruthfulQA intrinsic hallucination benchmark
- `validate_intrinsic.yaml` - Truth Vector integration validation

## Why Archived

These features showed negative or inconclusive results during research:

| Feature | Version | Result | Reason |
|---------|---------|--------|--------|
| Layer Drift | v11.0 | AUROC 0.23 | Worse than random (0.5). Measures "thinking effort", not deception. |
| TTT (Test-Time Training) | v11.0 | Failed | "Semantic Resolution Limit" - cannot distinguish entities (Paris vs London) |
| Symbolic Veto | v13.0 | Promising | Works but not in paper scope - detection paper, not RAG paper |
| Truth Vector | v10.0 | Experimental | Requires calibration, not zero-shot |
| Guided Decoding | - | Separate | Generation is out of scope for detection paper |

## Published Method (v8.0 SOTA)

The published method uses only:
1. **Authority Flow** - Recursive prompt recharge tracking signal provenance
2. **Unified Gating** - Context-dependent switch between RAG and free-gen modes
3. **Semantic Dispersion** - Consistency of top-k predictions over raw confidence

See `src/ag_sar/` for the canonical implementation.

## Restoring Archived Code

If you need to restore any archived features:

```bash
# Example: Restore truth_vector.py
git mv _legacy/calibration/truth_vector.py src/ag_sar/calibration/
```

Then update the relevant `__init__.py` files to re-export the classes.
