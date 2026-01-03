"""
Global pytest fixtures for AG-SAR test suite.

Provides common fixtures for both unit and integration tests.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ===== Device & Dtype =====

@pytest.fixture
def device():
    """Get compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype():
    """Get default dtype (bfloat16 on GPU, float32 on CPU)."""
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


# ===== Tensor Dimensions =====

@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 64


@pytest.fixture
def hidden_size():
    return 768


@pytest.fixture
def num_heads():
    return 12


@pytest.fixture
def head_dim(hidden_size, num_heads):
    return hidden_size // num_heads


# ===== Random Tensors =====

@pytest.fixture
def random_qk(batch_size, num_heads, seq_len, head_dim, device, dtype):
    """Generate random Q and K tensors."""
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    return Q, K


@pytest.fixture
def random_v(batch_size, seq_len, device, dtype):
    """Generate random value signal."""
    return torch.rand(batch_size, seq_len, device=device, dtype=dtype)


@pytest.fixture
def attention_mask(batch_size, seq_len, device):
    """Generate attention mask (all ones)."""
    return torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
