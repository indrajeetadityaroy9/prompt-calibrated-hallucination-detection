"""
Triton kernels for matrix-free centrality computation.

This module provides GPU-optimized kernels that compute attention-weighted
centrality without materializing O(N^2) attention matrices.
"""

from .centrality_flash import centrality_flash_fwd

__all__ = ["centrality_flash_fwd"]
