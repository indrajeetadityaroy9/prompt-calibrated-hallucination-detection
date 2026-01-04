"""
JSONL streaming logger for experiment results.

Uses line-delimited JSON (.jsonl) for incremental writes,
preventing data loss on long-running experiments.

Key design decision: Write results incrementally rather than
accumulating in memory to prevent OOM on large evaluations.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, TextIO


class JSONLLogger:
    """
    Streaming JSONL logger for experiment results.

    Writes each result immediately to disk as a JSON line,
    preventing data loss if the experiment crashes.

    Usage:
        >>> logger = JSONLLogger("results/exp1")
        >>> logger.start_experiment(config)
        >>> for result in results:
        ...     logger.log_result(result)
        >>> logger.finalize(summary_metrics)
    """

    def __init__(self, output_dir: str):
        """
        Initialize logger.

        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._results_file: Optional[TextIO] = None
        self._results_path: Optional[Path] = None
        self._config: Optional[Dict] = None
        self._n_results = 0

    def start_experiment(self, config: Dict[str, Any]) -> Path:
        """
        Start a new experiment and create output files.

        Args:
            config: Experiment configuration (will be saved)

        Returns:
            Path to the results JSONL file
        """
        exp_name = config.get("experiment", {}).get("name", "experiment")
        self._results_path = self.output_dir / f"{exp_name}_{self.timestamp}.jsonl"

        # Save config separately
        config_path = self.output_dir / f"{exp_name}_{self.timestamp}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        # Open results file for streaming writes
        self._results_file = open(self._results_path, "w")
        self._config = config
        self._n_results = 0

        print(f"Logging to: {self._results_path}")

        return self._results_path

    def log_result(
        self,
        method_name: str,
        dataset_name: str,
        sample_idx: int,
        score: float,
        label: int,
        latency_ms: float,
        confidence: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a single result (one sample, one method).

        Writes immediately to disk for crash safety.

        Args:
            method_name: Name of the method
            dataset_name: Name of the dataset
            sample_idx: Sample index in the dataset
            score: Uncertainty score
            label: Ground truth label
            latency_ms: Computation time in ms
            confidence: Model confidence (optional)
            extra: Additional metadata (must be JSON-serializable)
        """
        if self._results_file is None:
            raise RuntimeError("Must call start_experiment() before logging")

        record = {
            "method": method_name,
            "dataset": dataset_name,
            "sample_idx": sample_idx,
            "score": score,
            "label": label,
            "latency_ms": latency_ms,
            "confidence": confidence,
        }

        if extra:
            record["extra"] = extra

        self._results_file.write(json.dumps(record) + "\n")
        self._results_file.flush()  # Ensure immediate write
        self._n_results += 1

    def log_method_summary(
        self,
        method_name: str,
        dataset_name: str,
        metrics: Dict[str, float],
        ci_bounds: Dict[str, tuple],
        n_samples: int,
        mean_latency_ms: float,
    ) -> None:
        """
        Log summary metrics for one method on one dataset.

        Args:
            method_name: Name of the method
            dataset_name: Name of the dataset
            metrics: Computed metrics (AUROC, AUPRC, etc.)
            ci_bounds: Confidence interval bounds
            n_samples: Number of samples evaluated
            mean_latency_ms: Mean latency per sample
        """
        if self._results_file is None:
            raise RuntimeError("Must call start_experiment() before logging")

        summary = {
            "_type": "summary",
            "method": method_name,
            "dataset": dataset_name,
            "metrics": metrics,
            "ci_bounds": {k: list(v) for k, v in ci_bounds.items()},
            "n_samples": n_samples,
            "mean_latency_ms": mean_latency_ms,
        }

        self._results_file.write(json.dumps(summary) + "\n")
        self._results_file.flush()

    def finalize(self, final_summary: Optional[Dict[str, Any]] = None) -> Path:
        """
        Finalize logging and close files.

        Args:
            final_summary: Optional final summary to write

        Returns:
            Path to the results file
        """
        if self._results_file is None:
            raise RuntimeError("No experiment in progress")

        if final_summary:
            final_summary["_type"] = "final"
            final_summary["total_results"] = self._n_results
            final_summary["timestamp"] = self.timestamp
            self._results_file.write(json.dumps(final_summary) + "\n")

        self._results_file.close()
        self._results_file = None

        print(f"Wrote {self._n_results} results to {self._results_path}")

        return self._results_path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._results_file is not None:
            self._results_file.close()
        return False


def load_jsonl_results(path: str) -> Dict[str, Any]:
    """
    Load results from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        Dict with 'results' (list of records) and 'summaries' (list of summaries)
    """
    results = []
    summaries = []

    with open(path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get("_type") == "summary":
                summaries.append(record)
            elif record.get("_type") == "final":
                continue  # Skip final summary in results
            else:
                results.append(record)

    return {"results": results, "summaries": summaries}
