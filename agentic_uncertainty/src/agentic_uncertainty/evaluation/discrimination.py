"""Discrimination metrics: AUROC with bootstrap confidence intervals."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class AUROCResult:
    """AUROC result with confidence interval."""

    auroc: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def auroc_with_ci(
    probabilities: ArrayLike,
    labels: ArrayLike,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> AUROCResult:
    """Compute AUROC with bootstrap confidence interval.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1 for failure/success).
        confidence_level: Confidence level for the interval (e.g., 0.95).
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed for reproducibility.

    Returns:
        AUROCResult with point estimate and confidence interval.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    n_samples = len(probabilities)

    # Check for degenerate cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Can't compute AUROC with only one class
        return AUROCResult(
            auroc=0.5,
            ci_lower=0.5,
            ci_upper=0.5,
            confidence_level=confidence_level,
        )

    # Compute point estimate
    auroc = roc_auc_score(labels, probabilities)

    # Bootstrap for confidence interval
    rng = np.random.default_rng(random_state)
    bootstrap_aurocs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probabilities[indices]
        boot_labels = labels[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(boot_labels)) < 2:
            continue

        boot_auroc = roc_auc_score(boot_labels, boot_probs)
        bootstrap_aurocs.append(boot_auroc)

    if len(bootstrap_aurocs) == 0:
        # Fall back if all bootstrap samples were degenerate
        return AUROCResult(
            auroc=auroc,
            ci_lower=auroc,
            ci_upper=auroc,
            confidence_level=confidence_level,
        )

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aurocs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_aurocs, 100 * (1 - alpha / 2))

    return AUROCResult(
        auroc=auroc,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
    )


@dataclass
class AUPRCResult:
    """AUPRC result with confidence interval."""

    auprc: float
    ci_lower: float
    ci_upper: float
    confidence_level: float


def auprc_with_ci(
    probabilities: ArrayLike,
    labels: ArrayLike,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> AUPRCResult:
    """Compute AUPRC (Area Under Precision-Recall Curve) with bootstrap CI.

    AUPRC is a discrimination metric that is more informative than AUROC
    for imbalanced datasets. It measures the area under the precision-recall
    curve, giving more weight to correct positive predictions.

    Args:
        probabilities: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1 for failure/success).
        confidence_level: Confidence level for the interval (e.g., 0.95).
        n_bootstrap: Number of bootstrap samples.
        random_state: Random seed for reproducibility.

    Returns:
        AUPRCResult with point estimate and confidence interval.
    """
    probabilities = np.asarray(probabilities)
    labels = np.asarray(labels)

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")

    n_samples = len(probabilities)

    # Check for degenerate cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        # Can't compute AUPRC with only one class
        # Return base rate as fallback (proportion of positive class)
        base_rate = float(np.mean(labels)) if len(labels) > 0 else 0.5
        return AUPRCResult(
            auprc=base_rate,
            ci_lower=base_rate,
            ci_upper=base_rate,
            confidence_level=confidence_level,
        )

    # Compute point estimate
    auprc = average_precision_score(labels, probabilities)

    # Bootstrap for confidence interval
    rng = np.random.default_rng(random_state)
    bootstrap_auprcs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_probs = probabilities[indices]
        boot_labels = labels[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(boot_labels)) < 2:
            continue

        boot_auprc = average_precision_score(boot_labels, boot_probs)
        bootstrap_auprcs.append(boot_auprc)

    if len(bootstrap_auprcs) == 0:
        # Fall back if all bootstrap samples were degenerate
        return AUPRCResult(
            auprc=auprc,
            ci_lower=auprc,
            ci_upper=auprc,
            confidence_level=confidence_level,
        )

    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_auprcs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_auprcs, 100 * (1 - alpha / 2))

    return AUPRCResult(
        auprc=auprc,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
    )
