"""Gini coefficient computation for attention graph analysis."""

import numpy as np


def gini_coefficient(x: np.ndarray) -> float:
    """
    Compute Gini coefficient - measure of inequality/sharpness.

    The Gini coefficient measures how unequally distributed values are:
    - Range: 0 (perfect equality) to 1 (perfect inequality)
    - Higher Gini = more concentrated attention on fewer tokens (sharper focus)
    - Lower Gini = more diffuse attention across many tokens (scattered/fractured)

    In AG-SAR context:
    - Factual responses tend to have higher Gini (focused attention)
    - Hallucinations tend to have lower Gini (diffuse/scattered attention)

    Args:
        x: Input array of values (e.g., relevance scores)

    Returns:
        Gini coefficient between 0 and 1
    """
    x = np.abs(x).flatten()
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0

    # Sort values
    sorted_x = np.sort(x)
    n = len(x)

    # Gini formula: G = (n+1 - 2*sum(cumsum)/total) / n
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
