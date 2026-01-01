# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without requiring external semantic models.

**Key Innovation**: Uses O(N) matrix-free centrality computation via custom Triton kernel, eliminating O(N²) attention matrix materialization for infinite context scaling.

## Common Commands

```bash
# Install for development
pip install -e ".[dev]"

# Install with evaluation dependencies
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Run tests (skip GPU-only tests on CPU)
pytest tests/ -v --ignore=tests/test_triton_centrality.py

# Run a single test file
pytest tests/test_pipeline.py -v

# Run specific test class or method
pytest tests/test_pipeline.py::TestFullPipeline::test_compute_uncertainty_basic -v

# Run evaluation experiments
python scripts/run_eval.py --chapter 3     # Profiling only
python scripts/run_eval.py --chapter 1 2   # Mechanistic + predictive
python scripts/run_eval.py --all           # All experiments
python scripts/run_eval.py --quick         # Quick run with reduced samples
```

## Architecture

### Core Pipeline (`ag_sar/`)

The main entry point is `AGSAR` class in `ag_sar.py`. The pipeline flow:

1. **AttentionExtractor** (`attention_extractor.py`): Registers hooks on GPT-2's fused c_attn layer to capture Q, K, V vectors during forward pass. Key method `extract_semantic_qk()` returns stacked Q/K tensors (B, L*H, S, D) without O(N²) attention reconstruction.

2. **Triton Kernel** (`kernels/centrality_flash.py`): Custom Triton kernel computing `Out[b,h,i] = Σ_j softmax(Q·K^T/√d)[i,j] × v[j]` using Flash Attention-style online softmax. O(N) memory, causal masking built-in, float32 accumulators for numerical stability.

3. **Centrality** (`centrality.py`):
   - `matrix_free_power_iteration()`: Computes eigenvector centrality via Triton kernel without materializing attention matrices
   - `compute_sink_aware_centrality()`: Key formula `R(t_i) = C_eigen(t_i) × ||v_i||_2` - value norm weighting naturally filters attention sinks
   - Residual connection (0.5 weight) prevents centrality collapse on early tokens

4. **Uncertainty** (`uncertainty.py`): Computes Graph-Shifted Entropy (GSE): `GSE(T) = Σ_i H(t_i) × R̃(t_i)` where H is token entropy and R̃ is normalized relevance.

### Triton Kernels (`ag_sar/kernels/`)

- `centrality_flash.py`: Matrix-free attention-weighted centrality kernel
  - Input: Q (B,H,S,D), K (B,H,S,D), v (B,S)
  - Output: (B,H,S) per-head attention-weighted sum
  - Block sizes tuned for H100: BLOCK_M=64, BLOCK_N=64

### Configuration (`config.py`)

`AGSARConfig` dataclass controls all hyperparameters:
- `semantic_layers`: Number of final layers to use (default: 4) - structural selection
- `residual_weight`: Weight for self-attention residual in power iteration (0.5)
- `power_iteration_steps`: Max iterations for centrality (3)
- `hallucination_threshold`: GSE threshold for detection (0.7)
- `preferred_dtype`: Use `torch.bfloat16` on H100 (NEVER float16 with GPT-2 - causes NaN overflow)

### Evaluation Framework (`eval/`)

- `experiments/`: 7 experiments organized by thesis chapter (exp1-exp7)
- `baselines/`: Predictive Entropy and Original SAR implementations
- `metrics/`: AUROC, calibration (ECE), POS-tag correlation
- `profiling/`: Latency and throughput benchmarking (updated for Triton kernel timing)
- `visualizations/`: Plotting utilities for results

## Key Implementation Notes

- **O(N) Memory**: Matrix-free centrality via Triton kernel - no O(N²) attention matrices
- **Triton Required**: PyTorch >= 2.2, Triton >= 2.1 (no CPU fallback)
- **Precision**: Triton accumulators use float32 even with bfloat16 I/O for numerical stability
- **Structural Selection**: Uses last 4 semantic layers directly (no entropy-based head filtering)
- Power iteration uses 3 iterations (typically sufficient for convergence)
- The framework is optimized for NVIDIA H100 with TF32 and bfloat16
- Flash Attention 2 compatible - Q/K captured via hooks, attention never materialized
