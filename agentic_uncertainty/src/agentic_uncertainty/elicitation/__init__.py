"""Uncertainty elicitation methods for the paper-oriented public release.

Supported methods:
- exploration_direct
- review_direct
- review_adversarial
- mid_execution_direct

Checkpoint methods are kept for compatibility with existing scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EstimationResult, UncertaintyEstimator
from .checkpoint import CheckpointElicitation, CheckpointPosthocElicitation
from .exploration import (
    EXPLORATION_METHODS,
    ExplorationDirectElicitation,
    ExplorationElicitation,
)
from .mid_execution import (
    MID_EXECUTION_METHODS,
    MidExecutionDirectElicitation,
    MidExecutionElicitation,
)
from .review import (
    REVIEW_METHODS,
    ReviewAdversarialElicitation,
    ReviewDirectElicitation,
    ReviewElicitation,
)

if TYPE_CHECKING:
    from agentic_uncertainty.config import Settings
    from agentic_uncertainty.providers import ModelClient


# Core methods for experiments
SPECIALIZED_METHODS = {
    "checkpoint": CheckpointElicitation,
    "checkpoint_posthoc": CheckpointPosthocElicitation,
    # Exploration-based (pre-execution with agent)
    "exploration_direct": ExplorationDirectElicitation,
    # Mid-execution (during-execution trajectory evaluation)
    "mid_execution_direct": MidExecutionDirectElicitation,
    # Review-based (post-execution with agent)
    "review_direct": ReviewDirectElicitation,
    "review_adversarial": ReviewAdversarialElicitation,
}

def get_estimator(
    method: str,
    client: ModelClient | None = None,
    settings: Settings | None = None,
    **kwargs,
) -> UncertaintyEstimator:
    """Get an uncertainty estimator by method name.

    Args:
        method: Name of the elicitation method.
        client: Optional ModelClient instance.
        settings: Optional Settings instance.
        **kwargs: Additional arguments passed to specialized estimators.

    Returns:
        An UncertaintyEstimator instance configured for the specified method.

    Raises:
        ValueError: If the method is not recognized.

    Examples:
        >>> estimator = get_estimator("exploration_direct")
    """
    # Check for specialized methods first
    if method in SPECIALIZED_METHODS:
        cls = SPECIALIZED_METHODS[method]
        return cls(client=client, settings=settings, **kwargs)

    raise ValueError(
        f"Unknown elicitation method: {method}. "
        f"Available methods: {list(SPECIALIZED_METHODS.keys())}"
    )


def list_methods() -> list[str]:
    """List all available elicitation methods.

    Returns:
        Sorted list of method names.
    """
    return sorted(list(SPECIALIZED_METHODS.keys()))


__all__ = [
    # Core classes
    "UncertaintyEstimator",
    "EstimationResult",
    # Factory functions
    "get_estimator",
    "list_methods",
    # Core estimators
    "CheckpointElicitation",
    "CheckpointPosthocElicitation",
    # Exploration-based
    "ExplorationElicitation",
    "ExplorationDirectElicitation",
    "EXPLORATION_METHODS",
    # Mid-execution-based
    "MidExecutionElicitation",
    "MidExecutionDirectElicitation",
    "MID_EXECUTION_METHODS",
    # Review-based
    "ReviewElicitation",
    "ReviewAdversarialElicitation",
    "ReviewDirectElicitation",
    "REVIEW_METHODS",
]
