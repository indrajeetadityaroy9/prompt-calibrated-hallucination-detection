"""
Paper Generation Utilities for AG-SAR.

Provides helper functions for generating LaTeX tables and figures
from experimental results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def load_jsonl_results(path: Union[str, Path]) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def aggregate_results_by_method(results: List[Dict]) -> Dict[str, Dict]:
    """
    Aggregate results by method name.

    Returns:
        Dict mapping method_name -> {metric_name -> value}
    """
    by_method = {}
    for r in results:
        method = r.get('method', 'unknown')
        if method not in by_method:
            by_method[method] = {'scores': [], 'labels': []}
        by_method[method]['scores'].append(r.get('score', 0))
        by_method[method]['labels'].append(r.get('label', 0))
    return by_method


def format_metric(value: float, precision: int = 3, bold_threshold: Optional[float] = None) -> str:
    """
    Format a metric value for LaTeX.

    Args:
        value: The metric value
        precision: Decimal places
        bold_threshold: If provided, bold values >= threshold

    Returns:
        Formatted string, optionally with \\textbf{}
    """
    formatted = f"{value:.{precision}f}"
    if bold_threshold is not None and value >= bold_threshold:
        return f"\\textbf{{{formatted}}}"
    return formatted


def generate_latex_table(
    data: Dict[str, Dict[str, float]],
    metrics: List[str],
    method_order: Optional[List[str]] = None,
    caption: str = "",
    label: str = "",
    bold_best: bool = True,
) -> str:
    """
    Generate a LaTeX table from results data.

    Args:
        data: Dict mapping method_name -> {metric_name -> value}
        metrics: List of metric names to include as columns
        method_order: Optional ordering of methods (rows)
        caption: Table caption
        label: Table label for referencing
        bold_best: Whether to bold the best value in each column

    Returns:
        LaTeX table string
    """
    methods = method_order or sorted(data.keys())

    # Find best values for bolding
    best_values = {}
    if bold_best:
        for metric in metrics:
            values = [data[m].get(metric, 0) for m in methods if m in data]
            best_values[metric] = max(values) if values else 0

    # Build table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        if method not in data:
            continue
        row_values = []
        for metric in metrics:
            value = data[method].get(metric, 0)
            threshold = best_values.get(metric) if bold_best else None
            row_values.append(format_metric(value, bold_threshold=threshold))
        lines.append(f"{method} & " + " & ".join(row_values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def generate_latex_table_with_ci(
    data: Dict[str, Dict[str, tuple]],
    metrics: List[str],
    method_order: Optional[List[str]] = None,
    caption: str = "",
    label: str = "",
) -> str:
    """
    Generate LaTeX table with confidence intervals.

    Args:
        data: Dict mapping method_name -> {metric_name -> (mean, ci_low, ci_high)}
        metrics: List of metric names
        method_order: Optional ordering of methods
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string with CI format: mean_{-ci_low}^{+ci_high}
    """
    methods = method_order or sorted(data.keys())

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "Method & " + " & ".join(metrics) + " \\\\",
        "\\midrule",
    ]

    for method in methods:
        if method not in data:
            continue
        row_values = []
        for metric in metrics:
            if metric in data[method]:
                mean, ci_low, ci_high = data[method][metric]
                row_values.append(f"${mean:.3f}_{{-{ci_low:.3f}}}^{{+{ci_high:.3f}}}$")
            else:
                row_values.append("-")
        lines.append(f"{method} & " + " & ".join(row_values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


# Notation mapping for paper
NOTATION_MAPPING = {
    # Config parameter -> Paper symbol
    "prompt_authority": r"$\alpha_p$",
    "memory_weight": r"$w_m$",
    "gate_temperature": r"$\tau$",
    "dispersion_k": r"$k$",
    "semantic_layers": r"$L$",
    "residual_weight": r"$\beta$",
    "dispersion_sensitivity": r"$\gamma$",
    "hallucination_threshold": r"$\theta$",
}


def get_paper_symbol(param_name: str) -> str:
    """Get paper notation symbol for a config parameter."""
    return NOTATION_MAPPING.get(param_name, param_name)
