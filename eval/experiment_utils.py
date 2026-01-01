"""
Common utilities for AG-SAR experiments.

Provides shared patterns across all 7 evaluation experiments to reduce
code duplication and ensure consistent result formatting.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import json
import numpy as np
import torch


def tensor_to_list(tensor: Any) -> Optional[List]:
    """
    Convert tensor/array to Python list, handling all formats.

    Handles:
    - PyTorch tensors (CPU or GPU)
    - NumPy arrays
    - Objects with .tolist() method
    - Regular Python sequences

    Args:
        tensor: Input tensor, array, or sequence

    Returns:
        Python list, or None if input is None
    """
    if tensor is None:
        return None
    if hasattr(tensor, 'cpu'):
        arr = tensor.cpu().numpy()
    elif hasattr(tensor, 'numpy'):
        arr = tensor.numpy()
    elif hasattr(tensor, 'tolist'):
        return tensor.tolist()
    else:
        return list(tensor)
    # Flatten if batched (2D or higher)
    if len(arr.shape) > 1:
        arr = arr.flatten()
    return arr.tolist()


def safe_json_value(value: Any) -> Any:
    """
    Convert a value to JSON-serializable format.

    Args:
        value: Any Python value

    Returns:
        JSON-serializable version of the value
    """
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [safe_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: safe_json_value(v) for k, v in value.items()}
    # Check for numpy arrays BEFORE .item() since arrays have .item() but it fails for non-scalars
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    # For scalar tensors/arrays, try .item()
    if hasattr(value, 'item'):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


@dataclass
class ExperimentResult:
    """
    Standard experiment result container.

    Provides consistent structure for all AG-SAR experiments with
    automatic JSON serialization support.

    Attributes:
        experiment: Experiment name/identifier
        success: Whether experiment passed success criteria
        success_criteria: Description of success criteria
        num_samples: Number of samples processed
        metrics: Dictionary of computed metrics
    """
    experiment: str
    success: bool
    success_criteria: str
    num_samples: int
    metrics: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path
        """
        data = {
            'experiment': self.experiment,
            'success': self.success,
            'success_criteria': self.success_criteria,
            'num_samples': self.num_samples,
            'metrics': safe_json_value(self.metrics)
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Results saved to: {path}")

    def print_summary(self) -> None:
        """Print formatted experiment summary."""
        print(f"\n{'=' * 60}")
        print(f"RESULT: {'PASS' if self.success else 'FAIL'}")
        print(f"Criteria: {self.success_criteria}")
        for key, value in self.metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'=' * 60}")


def print_experiment_header(name: str, description: str = "") -> None:
    """
    Print formatted experiment header.

    Args:
        name: Experiment name
        description: Optional description
    """
    print("=" * 60)
    print(f"Experiment: {name}")
    if description:
        print(f"Description: {description}")
    print("=" * 60)


def print_progress(current: int, total: int, interval: int = 20) -> None:
    """
    Print progress update at specified intervals.

    Args:
        current: Current item index (0-based)
        total: Total number of items
        interval: Print every N items
    """
    if (current + 1) % interval == 0:
        print(f"  Processed {current + 1}/{total}")


def compute_summary_stats(
    values: List[float],
    prefix: str = ""
) -> Dict[str, float]:
    """
    Compute summary statistics for a list of values.

    Args:
        values: List of numeric values
        prefix: Optional prefix for result keys

    Returns:
        Dictionary with mean, std, min, max, median
    """
    if not values:
        return {}

    arr = np.array(values)
    stats = {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr))
    }

    if prefix:
        stats = {f"{prefix}_{k}": v for k, v in stats.items()}

    return stats


def format_table_row(
    label: str,
    values: List[Any],
    widths: List[int] = None,
    formats: List[str] = None
) -> str:
    """
    Format a table row with aligned columns.

    Args:
        label: Row label
        values: Column values
        widths: Column widths (default: 15 per column)
        formats: Format strings per column

    Returns:
        Formatted table row string
    """
    if widths is None:
        widths = [25] + [15] * len(values)
    if formats is None:
        formats = ['.3f'] * len(values)

    parts = [f"{label:<{widths[0]}}"]
    for i, val in enumerate(values):
        w = widths[i + 1] if i + 1 < len(widths) else 15
        fmt = formats[i] if i < len(formats) else '.3f'
        if isinstance(val, float):
            parts.append(f"{val:>{w}{fmt}}")
        else:
            parts.append(f"{val:>{w}}")

    return "".join(parts)


class MetricAccumulator:
    """
    Accumulator for collecting metrics across multiple samples.

    Provides running mean, std computation without storing all values.
    """

    def __init__(self):
        self.values: Dict[str, List[float]] = {}

    def add(self, key: str, value: float) -> None:
        """Add a single value for a metric."""
        if key not in self.values:
            self.values[key] = []
        self.values[key].append(value)

    def add_dict(self, metrics: Dict[str, float]) -> None:
        """Add multiple metrics from a dictionary."""
        for key, value in metrics.items():
            self.add(key, value)

    def get_mean(self, key: str) -> float:
        """Get mean for a metric."""
        if key not in self.values or not self.values[key]:
            return 0.0
        return float(np.mean(self.values[key]))

    def get_std(self, key: str) -> float:
        """Get standard deviation for a metric."""
        if key not in self.values or not self.values[key]:
            return 0.0
        return float(np.std(self.values[key]))

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        return {
            key: compute_summary_stats(values)
            for key, values in self.values.items()
        }

    def count(self, key: str = None) -> int:
        """Get count of values for a metric or minimum across all."""
        if key is not None:
            return len(self.values.get(key, []))
        if not self.values:
            return 0
        return min(len(v) for v in self.values.values())
