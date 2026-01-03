# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AG-SAR (Attention-Graph Shifting Attention to Relevance) is a zero-latency uncertainty quantification framework for LLMs. It detects hallucinations by analyzing internal attention graph structure without external semantic models.

**Key Features:**
- Optimized for NVIDIA H100 with bfloat16 precision and TF32 acceleration
- Zero external latency: pure internal model analysis using attention patterns
- Supports GPT-2, Llama-3/3.1/3.2, Mistral, and Qwen architectures
- Core metrics: Graph-Shifted Entropy (GSE), Manifold-Consistent Spectral Surprisal (MC-SS)

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install all including eval dependencies
pip install -e ".[all]"

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_pipeline.py -v

# Run tests with coverage
pytest --cov=ag_sar tests/

# Run evaluation suite (chapters correspond to experiment groups)
python scripts/run_eval.py --chapter 3
python scripts/run_eval.py --all

# Calibrate truth heads for SGSS
python scripts/calibrate_truth_heads.py
```

## Architecture

### Core Package (`ag_sar/`)

```
AGSAR (ag_sar.py)              Main interface - orchestrates full pipeline
    │
    ├── AttentionExtractor     Hook-based Q/K/V extraction without O(N²) matrices
    │   (attention_extractor.py)   Supports Flash Attention, handles GPT-2/Llama/Mistral/Qwen
    │
    ├── Centrality Module      Matrix-free eigenvector centrality via Triton kernel
    │   (centrality.py)            O(N) memory, Flash Attention-style online softmax
    │   └── kernels/centrality_flash.py  (Custom Triton kernel)
    │
    ├── Uncertainty Module     GSE/MC-SS metric computation
    │   (uncertainty.py)           torch.compile accelerated hot paths
    │
    └── AGSARConfig            All hyperparameters as dataclass
        (config.py)
```

### Data Flow

```
prompt + response → tokenize → AttentionExtractor.extract_semantic_qk()
    → Q, K, value_norms, logits (from final N semantic layers)
    → compute_sink_aware_centrality() via matrix-free power iteration
    → compute_token_entropy(logits)
    → GSE: entropy * normalized_relevance
    → MC-SS: bounded_surprisal + λ(1 - centrality)
    → detect_hallucination(uncertainty > threshold)
```

### Evaluation Framework (`eval/`)

- `experiments/`: 13 experiments (AUROC, calibration, latency, ablation, etc.)
- `baselines/`: Comparison methods (semantic entropy, eigenscore, original SAR)
- `metrics/`: AUROC, ECE, POS correlation, ROUGE, Gini
- `datasets.py`: TruthfulQA, WikiText loaders
- `nli_labeler.py`: NLI-based ground truth labeling

## Critical Implementation Details

### Precision Requirements
- **BFloat16 required for GPT-2** - float16 causes NaN overflow
- TF32 enforced at import via `enable_tf32()` (~3x speedup on H100)
- `torch.compile` used for entropy/GSE hot paths

### Architecture-Specific Hooks
- **GPT-2**: Uses c_attn hook (fused QKV)
- **Llama/Mistral/Qwen**: Monkey-patches attention.forward for post-RoPE Q/K capture
- **GQA**: Auto-expands KV heads (8 KV-heads → 32 Q-heads for Llama-3-8B)

### Critical Parameters (in AGSARConfig)
- `residual_weight=0.5`: Prevents early-token collapse in causal attention
- `power_iteration_steps=3`: Converges in 2-3 iterations
- `semantic_layers=4`: Final layers contain semantic consolidation
- `preferred_dtype=torch.bfloat16`: Required for stable H100 inference

### Attention Sink Handling
- Value norm multiplication filters sinks: `R(t_i) = C_eigen(t_i) × ||v_i||_2`
- First N tokens masked as structural (sink token masking)
- Hebbian prior for prompt-grounded semantics (MC-SS mode)

## Key Abstractions

1. **Matrix-Free Centrality**: Custom Triton kernel computes eigenvector centrality in O(N) memory without materializing the N×N attention matrix

2. **Graph-Shifted Entropy (GSE)**: Weights token entropy by topological relevance, filtering attention sinks

3. **Manifold-Consistent Spectral Surprisal (MC-SS)**: Additive formulation that catches "Confident Lies" using Hebbian prior anchored to prompt tokens

## Configuration Files

- `configs/llama3.2_truth_heads.json`: Pre-calibrated Z-scored head weights for SGSS
- `pyproject.toml`: Project metadata, dependencies, pytest config

## v3.2 Implementation Status (January 2025)

### Completed Work

1. **SmolLM-135M Baseline Established**
   - Pass Rate: 3/5 (60%) on distractor tests
   - Authority Flow (λ=0) is the primary signal source
   - Use `lambda_roughness=0.0` for small models

2. **v3.2 Captures Added to AttentionExtractor**
   - `_value_states`: Full V vectors after GQA expansion (B, S, hidden_size)
   - `_attn_outputs`: h_attn before o_proj (B, S, hidden_size)
   - `_block_outputs`: h_final after full decoder block (B, S, hidden_size) - for MLP Divergence

3. **New Metrics Implemented**
   - `compute_spectral_roughness()`: Dirichlet Energy formulation
   - `compute_mlp_divergence()`: v3.2 metric measuring 1 - CosineSim(h_attn, h_block)

### Key Scientific Findings

| Metric | SmolLM-135M Signal | Direction | Notes |
|--------|-------------------|-----------|-------|
| Authority Flow (λ=0) | Δ = +0.014 | ✓ Correct | Primary driver |
| Dirichlet Energy | Δ ≈ 0 | ✗ No signal | Hallucinations are "smooth" on attention graph |
| MLP Divergence | Δ = +0.003 to +0.01 | ✓ Correct | Weak on small models |

**Critical Insight**: "Confident Lies are smooth on the attention graph"
- Dirichlet Energy failed because hallucinations attend consistently (just to wrong sources)
- Authority Flow works by tracking signal provenance (context vs. generated tokens)
- MLP Divergence shows promise - hallucinations require MLP to override attention

### Next Steps for Llama-3 (GPU Required)

1. **Establish Llama-3 Baseline**
   ```python
   config = AGSARConfig(
       semantic_layers=4,
       enable_authority_flow=True,
       enable_spectral_roughness=False,  # Start with λ=0
       lambda_roughness=0.0,
       uncertainty_metric='v31',
   )
   ```

2. **Test MLP Divergence on Llama-3-8B**
   - Larger models have more MLP capacity to override attention
   - Expected: Larger Δ between grounded and parametric responses
   - Use `compute_mlp_divergence(h_attn, h_block)` from `ag_sar.ops`

3. **Conditional Roughness Strategy**
   ```python
   # For models > 1B params, use MLP Divergence instead of Dirichlet Energy
   if model_params > 1_000_000_000:
       roughness = compute_mlp_divergence(h_attn, h_block)
   else:
       roughness = 0.0  # Authority Flow only for small models
   ```

4. **Run Full Evaluation Suite**
   ```bash
   python scripts/run_eval.py --chapter 3  # Distractor tests
   python scripts/run_eval.py --all        # Full benchmark
   ```

### Environment Setup (venv311)

The project uses Python 3.11 with pinned transformers:
```bash
# Create and activate venv
python3.11 -m venv venv311
source venv311/bin/activate

# Install with eval dependencies
pip install -e ".[all]"

# Verify transformers version (must be 4.40-4.44)
python -c "import transformers; print(transformers.__version__)"
```

**Critical**: transformers >= 4.45 has breaking changes for attention hooks. The version is pinned in pyproject.toml.
