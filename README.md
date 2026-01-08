# AG-SAR: Attention-Graph Shifting Attention to Relevance

Single-pass inference-time uncertainty quantification for LLMs via internal attention graph analysis.

## Overview

AG-SAR detects hallucinations by analyzing the internal attention structure of language models, without requiring external semantic models or multiple forward passes. It achieves **O(N) memory** complexity through matrix-free eigenvector centrality computation.
