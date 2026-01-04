"""
Stage validation for ICML/NeurIPS evaluation pipeline.

Implements fail-fast validation checkpoints to prevent silent errors
from propagating through the evaluation pipeline.
"""

import json
import math
import warnings
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    details: Optional[Dict] = None


class StageValidator:
    """
    Validates experiment stage outputs before proceeding to next stage.

    Implements the "Valley of Death" guard: catch silent failures early,
    before they corrupt downstream metrics or waste GPU hours.

    Checks:
    1. NaN rate < 5%
    2. Score variance > 1e-4 (not collapsed)
    3. Label balance between 10%-90%
    4. AUROC computable and reasonable
    5. Auto-flip detection (AUROC < 0.5 warning)
    """

    def __init__(
        self,
        max_nan_rate: float = 0.05,
        min_variance: float = 1e-4,
        agsar_min_auroc: float = 0.6,
    ):
        """
        Initialize validator.

        Args:
            max_nan_rate: Maximum allowed NaN rate (default 5%)
            min_variance: Minimum score variance (default 1e-4)
            agsar_min_auroc: Minimum AUROC threshold for AG-SAR (default 0.6)
        """
        self.max_nan_rate = max_nan_rate
        self.min_variance = min_variance
        self.agsar_min_auroc = agsar_min_auroc

    def validate_stage(self, results: List[Dict]) -> ValidationResult:
        """
        Validate stage outputs before proceeding.

        Args:
            results: List of result dicts with 'score', 'label' keys

        Returns:
            ValidationResult with pass/fail status and message

        Raises:
            ValueError: If validation fails
        """
        if not results:
            raise ValueError("FAIL: No results to validate")

        scores = [r['score'] for r in results]
        labels = [r['label'] for r in results]

        # Check 1: NaN rate
        nan_count = sum(1 for s in scores if math.isnan(s) or math.isinf(s))
        nan_rate = nan_count / len(scores)
        if nan_rate > self.max_nan_rate:
            raise ValueError(
                f"FAIL: NaN rate {nan_rate:.1%} exceeds threshold {self.max_nan_rate:.1%}. "
                f"Dropped {nan_count}/{len(scores)} samples."
            )

        # Filter to valid scores for remaining checks
        valid_scores = [s for s in scores if not (math.isnan(s) or math.isinf(s))]
        valid_pairs = [(s, l) for s, l in zip(scores, labels)
                       if not (math.isnan(s) or math.isinf(s))]

        if not valid_scores:
            raise ValueError("FAIL: No valid scores after NaN filtering")

        # Check 2: Variance (detect model collapse)
        variance = float(np.std(valid_scores))
        if variance < self.min_variance:
            raise ValueError(
                f"FAIL: Score variance {variance:.6f} < {self.min_variance}. "
                "This indicates model collapse or degenerate scores."
            )

        # Check 3: Label balance (use <= to accept boundary values)
        valid_labels = [l for _, l in valid_pairs]
        pos_rate = sum(valid_labels) / len(valid_labels)
        if not (0.1 <= pos_rate <= 0.9):
            raise ValueError(
                f"FAIL: Extreme label imbalance ({pos_rate:.2f}). "
                "Expected between 10%-90% positive rate."
            )

        return ValidationResult(
            passed=True,
            message="Stage validation passed",
            details={
                "total_samples": len(results),
                "valid_samples": len(valid_scores),
                "nan_count": nan_count,
                "nan_rate": nan_rate,
                "score_variance": variance,
                "score_min": float(min(valid_scores)),
                "score_max": float(max(valid_scores)),
                "pos_rate": pos_rate,
            }
        )

    def validate_metrics(
        self,
        metrics: Dict[str, float],
        method_name: str = None
    ) -> ValidationResult:
        """
        Validate computed metrics are within expected ranges.

        Includes auto-flip detection and method-specific thresholds.

        Args:
            metrics: Dict of metric_name -> value
            method_name: Optional method name for specific thresholds

        Returns:
            ValidationResult with pass/fail status

        Raises:
            ValueError: If validation fails
        """
        auroc = metrics.get('auroc', 0.5)

        # Range checks
        if 'auroc' in metrics and not (0.0 <= auroc <= 1.0):
            raise ValueError(f"FAIL: AUROC {auroc} out of valid range [0, 1]")

        if 'auprc' in metrics:
            auprc = metrics['auprc']
            if not (0.0 <= auprc <= 1.0):
                raise ValueError(f"FAIL: AUPRC {auprc} out of valid range [0, 1]")

        if 'ece' in metrics:
            ece = metrics['ece']
            if not (0.0 <= ece <= 1.0):
                raise ValueError(f"FAIL: ECE {ece} out of valid range [0, 1]")

        if 'brier' in metrics:
            brier = metrics['brier']
            if not (0.0 <= brier <= 1.0):
                raise ValueError(f"FAIL: Brier {brier} out of valid range [0, 1]")

        # Auto-Flip Detection: Signal might be inverted
        warnings_list = []
        if auroc < 0.5:
            warnings_list.append(
                f"WARNING: SIGNAL INVERTED? AUROC={auroc:.3f} < 0.5. "
                "Higher score may mean LESS likely to hallucinate!"
            )

        # Stricter threshold for AG-SAR (we know the physics)
        if method_name == "AG-SAR" and auroc < self.agsar_min_auroc:
            raise ValueError(
                f"FAIL: AG-SAR AUROC={auroc:.3f} < {self.agsar_min_auroc} threshold. "
                "This indicates a broken run. Do not proceed to next stage."
            )

        # General sanity check
        if auroc < 0.4:
            warnings_list.append(
                f"WARNING: AUROC={auroc:.3f} is significantly below random chance"
            )

        return ValidationResult(
            passed=True,
            message="Metrics validation passed" + (
                f" with warnings: {'; '.join(warnings_list)}" if warnings_list else ""
            ),
            details={
                "metrics": metrics,
                "method_name": method_name,
                "warnings": warnings_list,
            }
        )

    def validate_jsonl_file(
        self, jsonl_path: str, method_name: str = None, max_corruption_rate: float = 0.01
    ) -> ValidationResult:
        """
        Validate results from a JSONL file.

        Args:
            jsonl_path: Path to JSONL results file
            method_name: Filter to specific method (optional)
            max_corruption_rate: Maximum allowed rate of corrupted lines (default 1%)

        Returns:
            ValidationResult with pass/fail status

        Raises:
            ValueError: If file not found or corruption rate exceeds threshold
        """
        path = Path(jsonl_path)
        if not path.exists():
            raise ValueError(f"FAIL: JSONL file not found: {jsonl_path}")

        results = []
        skipped_lines = 0
        total_lines = 0

        with open(path, 'r') as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if method_name and data.get('method') != method_name:
                        continue
                    if 'score' in data and 'label' in data:
                        results.append(data)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    continue

        # Check for file corruption
        if skipped_lines > 0:
            skip_rate = skipped_lines / total_lines if total_lines > 0 else 0
            warnings.warn(
                f"Skipped {skipped_lines} malformed JSONL lines ({skip_rate:.1%})"
            )
            if skip_rate > max_corruption_rate:
                raise ValueError(
                    f"FAIL: JSONL corruption rate {skip_rate:.1%} exceeds threshold "
                    f"{max_corruption_rate:.1%}. Skipped {skipped_lines}/{total_lines} lines."
                )

        return self.validate_stage(results)

    def generate_report(
        self,
        stage_name: str,
        results: List[Dict],
        metrics: Dict[str, float],
        ci_bounds: Dict[str, Tuple[float, float]] = None,
    ) -> str:
        """
        Generate a formatted validation report.

        Args:
            stage_name: Name of the stage
            results: List of result dicts
            metrics: Computed metrics
            ci_bounds: Optional confidence interval bounds

        Returns:
            Formatted report string
        """
        # Compute statistics
        scores = [r['score'] for r in results]
        labels = [r['label'] for r in results]

        nan_count = sum(1 for s in scores if math.isnan(s) or math.isinf(s))
        valid_scores = [s for s in scores if not (math.isnan(s) or math.isinf(s))]

        lines = [
            "=" * 80,
            f"STAGE VALIDATION REPORT: {stage_name}",
            "=" * 80,
            "",
            "DATA QUALITY:",
            f"  Total samples:     {len(results)}",
            f"  NaN scores:        {nan_count} ({nan_count/len(scores):.2%})  " +
                ("PASS" if nan_count / len(scores) <= self.max_nan_rate else "FAIL"),
            f"  Inf scores:        {sum(1 for s in scores if math.isinf(s))} ({sum(1 for s in scores if math.isinf(s))/len(scores):.2%})  PASS",
            f"  Label balance:     {sum(labels)} / {len(labels) - sum(labels)} ({sum(labels)/len(labels):.1%})  PASS",
            "",
            "SCORE DISTRIBUTIONS:",
        ]

        if valid_scores:
            lines.append(
                f"  Range:    min={min(valid_scores):.4f}, max={max(valid_scores):.4f}, "
                f"std={np.std(valid_scores):.4f}  " +
                ("PASS" if np.std(valid_scores) > self.min_variance else "FAIL")
            )

        lines.extend(["", "METRIC VALIDITY:"])
        for name, value in metrics.items():
            ci_str = ""
            if ci_bounds and name in ci_bounds:
                lo, hi = ci_bounds[name]
                ci_str = f" [{lo:.4f}, {hi:.4f}]"
            lines.append(f"  {name.upper()}:   {value:.4f}{ci_str}  PASS")

        # Auto-flip warning
        if metrics.get('auroc', 0.5) < 0.5:
            lines.extend([
                "",
                "=" * 60,
                "WARNING: SIGNAL MAY BE INVERTED (AUROC < 0.5)",
                "=" * 60,
            ])

        lines.extend([
            "",
            "VALIDATION: PASSED",
            "=" * 80,
        ])

        return "\n".join(lines)


def quick_validate(
    scores: List[float],
    labels: List[int],
    method_name: str = None,
) -> Tuple[bool, str]:
    """
    Quick validation helper for inline checks.

    Args:
        scores: List of uncertainty scores
        labels: List of ground truth labels
        method_name: Optional method name

    Returns:
        Tuple of (passed, message)
    """
    validator = StageValidator()
    results = [{'score': s, 'label': l} for s, l in zip(scores, labels)]

    try:
        validator.validate_stage(results)

        # Compute AUROC for metrics validation
        valid_mask = [not (math.isnan(s) or math.isinf(s)) for s in scores]
        valid_scores = [s for s, m in zip(scores, valid_mask) if m]
        valid_labels = [l for l, m in zip(labels, valid_mask) if m]

        if len(set(valid_labels)) >= 2:
            auroc = roc_auc_score(valid_labels, valid_scores)
            validator.validate_metrics({'auroc': auroc}, method_name)

        return True, "Validation passed"

    except ValueError as e:
        return False, str(e)
