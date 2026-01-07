"""
AG-SAR Task-Specific Parameter Presets.

This module provides externalized calibration parameters for different task types.
Presets are stored as YAML files for easy modification and extension.

Available presets:
- qa: Question answering tasks (conservative aggregation)
- rag: Retrieval-augmented generation (trust context more)
- summarization: Long-form summarization (higher dispersion k)
- attribution: Source attribution (moderate conservative)
- default: Baseline parameters (no task-specific tuning)

Usage:
    from ag_sar.presets import load_preset, get_available_presets

    # Load a preset
    params = load_preset("qa")
    print(params["calibration_temperature"])  # 1.2

    # List available presets
    presets = get_available_presets()

    # Clear cache (if presets were modified)
    clear_preset_cache()
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import functools

# Presets directory (same directory as this file)
_PRESETS_DIR = Path(__file__).parent


# Cache for loaded presets
@functools.lru_cache(maxsize=16)
def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file (cached)."""
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_preset(name: str, fallback_to_default: bool = True) -> Dict[str, Any]:
    """
    Load a task-specific parameter preset.

    Args:
        name: Preset name (e.g., "qa", "rag", "summarization")
        fallback_to_default: If True, return "default" preset when name not found

    Returns:
        Dict with calibration parameters:
            - aggregation_method: str ("mean", "percentile_10", etc.)
            - calibration_temperature: float
            - dispersion_k: int
            - dispersion_sensitivity: float
            - parametric_weight: float

    Raises:
        FileNotFoundError: If preset not found and fallback_to_default is False

    Example:
        >>> params = load_preset("qa")
        >>> params["calibration_temperature"]
        1.2
    """
    preset_path = _PRESETS_DIR / f"{name}.yaml"

    if not preset_path.exists():
        if fallback_to_default:
            preset_path = _PRESETS_DIR / "default.yaml"
            if not preset_path.exists():
                # Return hardcoded defaults if file missing
                import warnings
                warnings.warn(
                    f"Preset '{name}' not found and default.yaml is missing from {_PRESETS_DIR}. "
                    "Using hardcoded defaults. This may indicate an incomplete installation.",
                    RuntimeWarning,
                )
                return {
                    "aggregation_method": "mean",
                    "calibration_temperature": 1.0,
                    "dispersion_k": 5,
                    "dispersion_sensitivity": 1.0,
                    "parametric_weight": 0.5,
                }
        else:
            raise FileNotFoundError(
                f"Preset '{name}' not found at {preset_path}. "
                f"Available: {get_available_presets()}"
            )

    return _load_yaml(preset_path)


def get_available_presets() -> List[str]:
    """
    Get list of available preset names.

    Returns:
        List of preset names (without .yaml extension)

    Example:
        >>> get_available_presets()
        ['attribution', 'default', 'qa', 'rag', 'summarization']
    """
    return sorted([
        p.stem for p in _PRESETS_DIR.glob("*.yaml")
        if not p.stem.startswith("_")
    ])


def clear_preset_cache() -> None:
    """
    Clear the preset loading cache.

    Call this if you've modified preset YAML files and want to reload them.
    """
    _load_yaml.cache_clear()


def get_preset_path(name: str) -> Optional[Path]:
    """
    Get the file path for a preset.

    Args:
        name: Preset name

    Returns:
        Path to preset file, or None if not found
    """
    preset_path = _PRESETS_DIR / f"{name}.yaml"
    return preset_path if preset_path.exists() else None


__all__ = [
    "load_preset",
    "get_available_presets",
    "clear_preset_cache",
    "get_preset_path",
]
