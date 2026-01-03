"""
Triton kernels for matrix-free centrality computation.

This module provides GPU-optimized kernels that compute attention-weighted
centrality without materializing O(N^2) attention matrices.

Note: Triton is only available on Linux with NVIDIA GPUs. On other platforms,
a fallback implementation is provided that raises an error if called.
"""

import sys

if sys.platform == "linux":
    from .centrality_flash import centrality_flash_fwd
else:
    def centrality_flash_fwd(*args, **kwargs):
        raise RuntimeError(
            "Triton kernels are only available on Linux with NVIDIA GPUs. "
            "The centrality_flash_fwd kernel cannot be used on this platform."
        )

__all__ = ["centrality_flash_fwd"]
