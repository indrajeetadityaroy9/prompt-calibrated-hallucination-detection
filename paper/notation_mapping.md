# AG-SAR Notation Mapping

This document maps code parameter names to paper notation symbols for consistency.

## Core Parameters

| Code Parameter | Paper Symbol | Description | Default |
|----------------|--------------|-------------|---------|
| `prompt_authority` | $\alpha_p$ | Weight for prompt token authority in flow computation | 0.3 |
| `memory_weight` | $w_m$ | Balance between context (prompt) and parametric (memory) knowledge | 0.5 |
| `gate_temperature` | $\tau$ | Temperature for Agreement Gate softmax | 1.0 |
| `residual_weight` | $\beta$ | Residual connection strength in authority propagation | 0.5 |
| `dispersion_k` | $k$ | Number of top tokens for Semantic Dispersion | 5 |
| `dispersion_sensitivity` | $\gamma$ | Sensitivity scaling for dispersion score | 1.0 |
| `semantic_layers` | $L$ | Number of final layers for semantic analysis | 4 |
| `hallucination_threshold` | $\theta$ | Decision threshold for hallucination classification | 0.5 |

## Mechanism Components

| Code Name | Paper Name | Symbol |
|-----------|------------|--------|
| `authority_flow` | Authority Flow | $\mathcal{A}$ |
| `agreement_gate` | Agreement Gate | $\mathcal{G}$ |
| `semantic_dispersion` | Semantic Dispersion | $\mathcal{D}$ |
| `unified_gating` | Unified Gating | $\mathcal{U}$ |

## Score Computation

The final uncertainty score is computed as:

$$U = \mathcal{G}(\mathcal{A}, \mathcal{D})$$

Where:
- $\mathcal{A}$ = Authority Flow score (how much response attends to prompt)
- $\mathcal{D}$ = Semantic Dispersion score (vocabulary distribution concentration)
- $\mathcal{G}$ = Agreement Gate (combines signals using attention agreement)

## Ablation Notation

For ablation studies, we use subtraction notation:

| Ablation | Notation | Description |
|----------|----------|-------------|
| No Authority Flow | $-\mathcal{A}$ | Remove authority flow computation |
| No Agreement Gate | $-\mathcal{G}$ | Remove gating mechanism |
| No Semantic Dispersion | $-\mathcal{D}$ | Remove dispersion signal |
| No Unified Gating | $-\mathcal{U}$ | Disable unified gating |
| No Residual | $-\beta$ | Remove residual connections |

## LaTeX Macros

For paper consistency, define these macros in your preamble:

```latex
% Parameters
\newcommand{\promptauth}{\alpha_p}
\newcommand{\memweight}{w_m}
\newcommand{\gatetemp}{\tau}
\newcommand{\residweight}{\beta}
\newcommand{\dispersionk}{k}
\newcommand{\dispsens}{\gamma}
\newcommand{\semlayers}{L}
\newcommand{\hallthresh}{\theta}

% Components
\newcommand{\authflow}{\mathcal{A}}
\newcommand{\agreegate}{\mathcal{G}}
\newcommand{\semdisp}{\mathcal{D}}
\newcommand{\unifiedgate}{\mathcal{U}}

% Method name
\newcommand{\agsar}{\textsc{AG-SAR}}
```

## Usage in Code

The `paper/utils.py` module provides a `NOTATION_MAPPING` dictionary and
`get_paper_symbol()` function for programmatic access:

```python
from paper.utils import get_paper_symbol, NOTATION_MAPPING

# Get LaTeX symbol for a parameter
symbol = get_paper_symbol("memory_weight")  # Returns r"$w_m$"

# Full mapping dictionary
print(NOTATION_MAPPING)
```
