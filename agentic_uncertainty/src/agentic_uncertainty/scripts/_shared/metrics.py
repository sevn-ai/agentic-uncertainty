"""Metrics computation utilities."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from agentic_uncertainty.data import load_ground_truth
from agentic_uncertainty.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from agentic_uncertainty.evaluation.discrimination import auprc_with_ci, auroc_with_ci

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Result of metrics computation for a single method."""

    method: str
    n_samples: int
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    auprc: float
    auprc_ci_lower: float
    auprc_ci_upper: float
    ece: float
    mce: float
    brier: float
    mean_prediction: float
    mean_label: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "n_samples": self.n_samples,
            "auroc": self.auroc,
            "auroc_ci_lower": self.auroc_ci_lower,
            "auroc_ci_upper": self.auroc_ci_upper,
            "auprc": self.auprc,
            "auprc_ci_lower": self.auprc_ci_lower,
            "auprc_ci_upper": self.auprc_ci_upper,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "mean_prediction": self.mean_prediction,
            "mean_label": self.mean_label,
        }


def compute_metrics_for_method(
    predictions: list[float],
    labels: list[bool],
    method_name: str,
) -> MetricsResult:
    """Compute metrics for a single elicitation method.

    Args:
        predictions: List of probability predictions (0-1).
        labels: List of binary ground truth labels.
        method_name: Name of the elicitation method.

    Returns:
        MetricsResult with computed metrics.
    """
    preds = np.array(predictions)
    labs = np.array(labels, dtype=int)

    # Compute metrics
    auroc_result = auroc_with_ci(preds, labs)
    auprc_result = auprc_with_ci(preds, labs)
    ece = expected_calibration_error(preds, labs)
    mce = maximum_calibration_error(preds, labs)
    brier = brier_score(preds, labs)

    return MetricsResult(
        method=method_name,
        n_samples=len(predictions),
        auroc=auroc_result.auroc,
        auroc_ci_lower=auroc_result.ci_lower,
        auroc_ci_upper=auroc_result.ci_upper,
        auprc=auprc_result.auprc,
        auprc_ci_lower=auprc_result.ci_lower,
        auprc_ci_upper=auprc_result.ci_upper,
        ece=ece,
        mce=mce,
        brier=brier,
        mean_prediction=float(preds.mean()),
        mean_label=float(labs.mean()),
    )


def compute_metrics(
    predictions_path: Path,
    ground_truth_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Compute metrics for all methods in a predictions file.

    Args:
        predictions_path: Path to predictions JSON file.
        ground_truth_path: Path to ground truth JSON file.
        output_dir: Optional output directory for metrics file.

    Returns:
        Dictionary with metrics for all methods.
    """
    # Load data
    with open(predictions_path) as f:
        predictions_data = json.load(f)

    ground_truth = load_ground_truth(ground_truth_path)

    # Get instance IDs from predictions
    instance_ids = predictions_data.get("instance_ids", [])
    methods = predictions_data.get("methods", [])
    results = predictions_data.get("results", {})

    # Match predictions with ground truth
    matched_ids = []
    matched_labels = []
    for iid in instance_ids:
        if iid in ground_truth:
            matched_ids.append(iid)
            matched_labels.append(ground_truth[iid])

    # Compute metrics for each method
    all_metrics = []
    for method in methods:
        method_results = results.get(method, {})
        method_preds = method_results.get("predictions", [])

        # Filter predictions to matched instances only
        matched_preds = []
        for i, iid in enumerate(instance_ids):
            if iid in ground_truth and i < len(method_preds):
                matched_preds.append(method_preds[i])

        if len(matched_preds) != len(matched_labels):
            continue

        metrics = compute_metrics_for_method(matched_preds, matched_labels, method)
        all_metrics.append(metrics.to_dict())

    # Build output
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "predictions_file": str(predictions_path),
        "ground_truth_file": str(ground_truth_path),
        "n_matched_instances": len(matched_ids),
        "n_resolved": sum(matched_labels),
        "n_unresolved": len(matched_labels) - sum(matched_labels),
        "metrics": all_metrics,
    }

    # Save if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "metrics.json"
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

    return output_data


def compute_standard_metrics(
    predictions: list[float | None] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
) -> dict[str, float]:
    """Compute standard calibration metrics.

    Computes AUROC (with CI), ECE, Brier score, and overconfidence.
    This is the common set of metrics used across all experiment scripts.

    Note: Predictions with None values are filtered out. If any predictions
    are filtered, a warning is logged with the count.

    Args:
        predictions: List or array of probability predictions (0-1).
            None values are filtered out along with their corresponding labels.
        labels: List or array of binary ground truth labels.

    Returns:
        Dictionary with:
        - n_samples: Number of valid samples (after filtering)
        - n_filtered: Number of samples filtered out due to None predictions
        - auroc, auroc_ci_lower, auroc_ci_upper: AUROC with confidence interval
        - ece: Expected calibration error
        - brier: Brier score
        - mean_prediction: Mean predicted probability
        - mean_label: Base rate (mean of labels)
        - overconfidence: mean_prediction - mean_label
    """
    # Filter out None predictions and their corresponding labels
    predictions_list = list(predictions)
    labels_list = list(labels)

    valid_pairs = [
        (p, l) for p, l in zip(predictions_list, labels_list)
        if p is not None
    ]

    n_total = len(predictions_list)
    n_filtered = n_total - len(valid_pairs)

    if n_filtered > 0:
        logger.warning(
            "Filtered out %d/%d predictions with missing confidence values.",
            n_filtered,
            n_total,
        )

    if not valid_pairs:
        logger.error("No valid predictions to compute metrics.")
        return {
            "n_samples": 0,
            "n_filtered": n_filtered,
            "auroc": float("nan"),
            "auroc_ci_lower": float("nan"),
            "auroc_ci_upper": float("nan"),
            "ece": float("nan"),
            "brier": float("nan"),
            "mean_prediction": float("nan"),
            "mean_label": float("nan"),
            "overconfidence": float("nan"),
        }

    preds = np.array([p for p, _ in valid_pairs])
    labs = np.array([l for _, l in valid_pairs], dtype=float)

    auroc_result = auroc_with_ci(preds, labs)

    return {
        "n_samples": len(valid_pairs),
        "n_filtered": n_filtered,
        "auroc": auroc_result.auroc,
        "auroc_ci_lower": auroc_result.ci_lower,
        "auroc_ci_upper": auroc_result.ci_upper,
        "ece": expected_calibration_error(preds, labs),
        "brier": brier_score(preds, labs),
        "mean_prediction": float(preds.mean()),
        "mean_label": float(labs.mean()),
        "overconfidence": float(preds.mean() - labs.mean()),
    }


def compute_false_commitment_rate(
    predictions: list[float] | np.ndarray,
    labels: list[bool] | list[int] | np.ndarray,
    threshold: float = 0.9,
) -> dict[str, float]:
    """Compute false commitment rate.

    False commitment occurs when the model is highly confident (above threshold)
    but the prediction is wrong (label is 0/False).

    Args:
        predictions: List or array of probability predictions (0-1).
        labels: List or array of binary ground truth labels.
        threshold: Confidence threshold for "high confidence" (default: 0.9).

    Returns:
        Dictionary with:
        - false_commitment_rate: Fraction of high-confidence predictions that were wrong
        - false_commitment_count: Number of false commitments
        - high_confidence_count: Number of high-confidence predictions
        - threshold: The threshold used
    """
    preds = np.array(predictions)
    labs = np.array(labels, dtype=float)

    high_conf_mask = preds > threshold
    false_commits = high_conf_mask & (labs == 0)

    high_conf_count = int(high_conf_mask.sum())
    false_commit_count = int(false_commits.sum())
    fcr = false_commit_count / high_conf_count if high_conf_count > 0 else 0.0

    return {
        "false_commitment_rate": fcr,
        "false_commitment_count": false_commit_count,
        "high_confidence_count": high_conf_count,
        "threshold": threshold,
    }
