# Legacy Research Code

This directory contains archived code for reproducing paper ablation studies.
**NOT part of the pip-installable library.**

## Contents
- `ablations/` - v5.0 LID, v6.0 Spectral methods (Table 3)
- `05_mechanism_ablation.yaml` - Ablation experiment config
- `test_triton_centrality.py` - Legacy kernel tests

## Important: Import Dependencies
These files contain relative imports from `src/ag_sar/` (e.g., `from ..config import ...`).
They are archived **for reading/copying**, not for direct execution from this location.

## To Reproduce Ablations
1. Copy files back to original locations:
   ```bash
   cp -r legacy_research/ablations/ src/ag_sar/measures/ablations/
   cp legacy_research/05_mechanism_ablation.yaml experiments/configs/
   ```
2. Run ablation experiment:
   ```bash
   python -m experiments.main --config experiments/configs/05_mechanism_ablation.yaml
   ```
3. Remove after reproduction:
   ```bash
   rm -rf src/ag_sar/measures/ablations/
   ```
