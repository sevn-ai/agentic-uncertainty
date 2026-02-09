"""Calibration metrics: Expected Calibration Error (ECE) and Brier score."""

import numpy as np
from numpy.typing import ArrayLike


def expected_calibration_error(
    probabilities: ArrayLike,
    labels: ArrayLike,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted probabilities and actual
    outcomes, weighted by the number of samples in each bin.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1 for failure/success).
        num_bins: Number of bins for probability bucketing.

    Returns:
        ECE value (lower is better, 0 is perfectly calibrated).
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    if len(probabilities) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(probabilities, bin_boundaries[1:-1])

    ece = 0.0
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue

        bin_probs = probabilities[mask]
        bin_labels = labels[mask]

        avg_confidence = np.mean(bin_probs)
        avg_accuracy = np.mean(bin_labels)

        bin_weight = len(bin_probs) / len(probabilities)
        ece += bin_weight * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def maximum_calibration_error(
    probabilities: ArrayLike,
    labels: ArrayLike,
    num_bins: int = 10,
) -> float:
    """Compute Maximum Calibration Error (MCE).

    MCE is the maximum absolute difference between predicted probabilities
    and actual outcomes across all bins. Unlike ECE which averages across bins,
    MCE captures the worst-case calibration failure.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1 for failure/success).
        num_bins: Number of bins for probability bucketing.

    Returns:
        MCE value (lower is better, 0 is perfectly calibrated).
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    if len(probabilities) == 0:
        return 0.0

    # Create bins
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(probabilities, bin_boundaries[1:-1])

    max_error = 0.0
    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if not np.any(mask):
            continue

        bin_probs = probabilities[mask]
        bin_labels = labels[mask]

        avg_confidence = np.mean(bin_probs)
        avg_accuracy = np.mean(bin_labels)

        bin_error = np.abs(avg_accuracy - avg_confidence)
        max_error = max(max_error, bin_error)

    return float(max_error)


def brier_score(probabilities: ArrayLike, labels: ArrayLike) -> float:
    """Compute Brier score.

    Brier score is the mean squared error between predicted probabilities
    and binary outcomes. Lower is better.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1 for failure/success).

    Returns:
        Brier score (lower is better, 0 is perfect).
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    if len(probabilities) == 0:
        return 0.0

    return float(np.mean((probabilities - labels) ** 2))
