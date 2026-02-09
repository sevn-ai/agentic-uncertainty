"""Control policies and cost tracking for uncertainty-aware agent execution."""

from dataclasses import dataclass, field
from typing import Literal


# --- Cost Tracking ---


@dataclass
class CostLog:
    """Cost tracking for a single instance."""

    instance_id: str
    costs: list[float] = field(default_factory=list)
    resolved: bool | None = None

    @property
    def total_cost(self) -> float:
        return sum(self.costs)

    @property
    def num_attempts(self) -> int:
        return len(self.costs)


def compute_efficiency_metrics(logs: dict[str, CostLog]) -> dict:
    """Compute efficiency metrics from cost logs.

    Args:
        logs: Dict mapping instance_id to CostLog.

    Returns:
        Dict with efficiency metrics.
    """
    if not logs:
        return {
            "num_instances": 0,
            "num_resolved": 0,
            "resolve_rate": 0.0,
            "total_cost": 0.0,
            "avg_cost_per_instance": 0.0,
            "avg_cost_per_resolved": float("inf"),
            "resolved_per_dollar": 0.0,
        }

    resolved = [log for log in logs.values() if log.resolved is True]
    total_cost = sum(log.total_cost for log in logs.values())
    resolved_cost = sum(log.total_cost for log in resolved)

    num_instances = len(logs)
    num_resolved = len(resolved)

    return {
        "num_instances": num_instances,
        "num_resolved": num_resolved,
        "resolve_rate": num_resolved / num_instances if num_instances > 0 else 0.0,
        "total_cost": total_cost,
        "avg_cost_per_instance": total_cost / num_instances if num_instances > 0 else 0.0,
        "avg_cost_per_resolved": resolved_cost / num_resolved if num_resolved > 0 else float("inf"),
        "resolved_per_dollar": num_resolved / total_cost if total_cost > 0 else 0.0,
    }


# --- Policies ---


def should_stop_early(
    confidence: float,
    threshold: float = 0.3,
) -> bool:
    """Early stop policy: abort when confidence is too low.

    Args:
        confidence: Current p(resolved) estimate.
        threshold: Minimum confidence to continue.

    Returns:
        True if should stop, False to continue.
    """
    return confidence < threshold


def should_restart(
    confidence: float,
    attempt: int,
    restart_threshold: float = 0.4,
    max_attempts: int = 3,
) -> bool:
    """Restart policy: abort current attempt and try again.

    Args:
        confidence: Current p(resolved) estimate.
        attempt: Current attempt number (1-indexed).
        restart_threshold: Confidence below which to consider restart.
        max_attempts: Maximum number of attempts allowed.

    Returns:
        True if should restart, False otherwise.
    """
    if attempt >= max_attempts:
        return False
    return confidence < restart_threshold


def decide_action(
    confidence: float,
    cost_so_far: float,
    remaining_budget: float,
    attempt: int = 1,
    policy: str = "baseline",
    **policy_params,
) -> str:
    """Decide action based on policy.

    Args:
        confidence: Current p(resolved) estimate.
        cost_so_far: Cost spent on this instance.
        remaining_budget: Remaining per-instance budget.
        attempt: Current attempt number.
        policy: Policy name (baseline, early_stop, restart).
        **policy_params: Additional policy parameters.

    Returns:
        Action string: "continue", "submit", "restart", or "abort".
    """
    if remaining_budget <= 0:
        return "submit"

    if policy == "baseline":
        return "continue"

    elif policy == "early_stop":
        threshold = policy_params.get("threshold", 0.3)
        if should_stop_early(confidence, threshold):
            return "abort"
        return "continue"

    elif policy == "restart":
        abort_threshold = policy_params.get("abort_threshold", 0.2)
        restart_threshold = policy_params.get("restart_threshold", 0.4)
        max_attempts = policy_params.get("max_attempts", 3)

        if confidence < abort_threshold and attempt >= max_attempts:
            return "abort"
        if should_restart(confidence, attempt, restart_threshold, max_attempts):
            return "restart"
        return "continue"

    else:
        return "continue"


# --- Portfolio Allocation ---


def allocate_budget(
    confidences: dict[str, float],
    total_budget: float,
    strategy: str = "uniform",
    min_budget: float = 0.5,
    max_budget: float = 6.0,
) -> dict[str, float]:
    """Allocate budget across instances based on confidence estimates.

    Args:
        confidences: Dict mapping instance_id to p(resolved) estimate.
        total_budget: Total budget to allocate.
        strategy: Allocation strategy (uniform, confidence_weighted, uncertainty_weighted).
        min_budget: Minimum budget per instance.
        max_budget: Maximum budget per instance.

    Returns:
        Dict mapping instance_id to allocated budget.
    """
    if not confidences:
        return {}

    n = len(confidences)

    if strategy == "uniform":
        budget_per = min(max_budget, total_budget / n)
        return {iid: max(min_budget, budget_per) for iid in confidences}

    elif strategy == "confidence_weighted":
        # Allocate more to higher-confidence instances
        total_conf = sum(confidences.values())
        if total_conf == 0:
            total_conf = n  # Fallback to uniform

        allocations = {}
        for iid, conf in confidences.items():
            weight = conf / total_conf
            budget = weight * total_budget
            budget = max(min_budget, min(max_budget, budget))
            allocations[iid] = budget
        return allocations

    elif strategy == "uncertainty_weighted":
        # Allocate more to mid-confidence (most uncertain) instances
        # Uncertainty peaks at 0.5
        uncertainties = {iid: 1 - abs(0.5 - conf) * 2 for iid, conf in confidences.items()}
        total_uncertainty = sum(uncertainties.values())
        if total_uncertainty == 0:
            total_uncertainty = n

        allocations = {}
        for iid in confidences:
            weight = uncertainties[iid] / total_uncertainty
            budget = weight * total_budget
            budget = max(min_budget, min(max_budget, budget))
            allocations[iid] = budget
        return allocations

    elif strategy == "greedy":
        # Allocate to highest-confidence first until budget exhausted
        sorted_items = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        allocations = {}
        remaining = total_budget

        for iid, _conf in sorted_items:
            if remaining >= min_budget:
                budget = min(max_budget, remaining)
                allocations[iid] = budget
                remaining -= budget
            else:
                allocations[iid] = 0.0

        return allocations

    else:
        # Default to uniform
        budget_per = min(max_budget, total_budget / n)
        return {iid: max(min_budget, budget_per) for iid in confidences}


def prioritize_instances(
    confidences: dict[str, float],
    strategy: str = "high_first",
) -> list[str]:
    """Order instances by priority for execution.

    Args:
        confidences: Dict mapping instance_id to p(resolved).
        strategy: Ordering strategy.

    Returns:
        List of instance_ids in priority order.
    """
    if strategy == "high_first":
        # Execute high-confidence instances first
        return sorted(confidences.keys(), key=lambda x: confidences[x], reverse=True)
    elif strategy == "low_first":
        # Execute low-confidence instances first (fail fast)
        return sorted(confidences.keys(), key=lambda x: confidences[x])
    elif strategy == "uncertain_first":
        # Execute most uncertain (closest to 0.5) first
        return sorted(confidences.keys(), key=lambda x: abs(0.5 - confidences[x]))
    else:
        return list(confidences.keys())


# --- Online Control Policy ---


@dataclass
class ControlPolicy:
    """Unified control policy for online uncertainty-based decisions.

    Handles both within-attempt (early abort) and between-attempt (retry) decisions.

    Args:
        early_abort_threshold: Abort attempt if confidence < this. None = never abort early.
        max_retries: Maximum number of retry attempts. 0 = no retries.
        retry_threshold: Only retry if final confidence > this. None = always retry on failure.
        checkpoints: Steps at which to check confidence for early abort.
            Can be absolute steps (int) or percentages of trajectory length (float 0-1).
    """

    early_abort_threshold: float | None = None
    max_retries: int = 0
    retry_threshold: float | None = None
    checkpoints: list[float | int] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])

    def get_checkpoint_steps(self, n_steps: int) -> list[int]:
        """Convert checkpoint specification to actual step numbers for a trajectory.

        Args:
            n_steps: Total number of steps in the trajectory.

        Returns:
            List of step numbers to checkpoint at.
        """
        steps = []
        for cp in self.checkpoints:
            if isinstance(cp, float) and 0 < cp <= 1:
                # Percentage-based: convert to step number
                step = max(1, int(cp * n_steps))
            else:
                # Absolute step number
                step = int(cp)
            if step <= n_steps and step not in steps:
                steps.append(step)
        return sorted(steps)

    def should_abort(self, confidence: float) -> bool:
        """Check if current attempt should be aborted early."""
        if self.early_abort_threshold is None:
            return False
        return confidence < self.early_abort_threshold

    def should_retry(self, confidence: float, attempt: int) -> bool:
        """Check if we should retry after failure/abort.

        Args:
            confidence: Final confidence from the attempt.
            attempt: Current attempt number (1-indexed).

        Returns:
            True if should retry, False otherwise.
        """
        if attempt > self.max_retries:
            return False
        if self.retry_threshold is None:
            # Blind retry: always retry if we have attempts left
            return True
        # Smart retry: only retry if confidence suggests recovery is possible
        return confidence >= self.retry_threshold


@dataclass
class OnlineRunResult:
    """Result of running an instance with online control."""

    instance_id: str
    resolved: bool
    total_cost: float
    num_attempts: int
    confidence_trace: list[tuple[int, float]]  # (step, confidence)
    attempt_outcomes: list[str]  # e.g., ["aborted@step10", "resolved"]
    final_step: int


# --- Predefined Policies ---

POLICIES: dict[str, ControlPolicy] = {
    # Baseline: single run to completion, no intervention
    "single_run": ControlPolicy(),
    # Early abort only: stop failing runs early, no retry
    "early_abort": ControlPolicy(
        early_abort_threshold=0.15,
    ),
    # Early abort with higher threshold (more aggressive)
    "early_abort_aggressive": ControlPolicy(
        early_abort_threshold=0.6,
    ),
    # Blind retry: always retry on failure (up to 2 retries), no early abort
    "blind_retry_1": ControlPolicy(
        max_retries=1,
    ),
    "blind_retry_2": ControlPolicy(
        max_retries=2,
    ),
    # Early abort + blind retry
    "early_abort_blind_retry_1": ControlPolicy(
        early_abort_threshold=0.15,
        max_retries=1,
    ),
    "early_abort_blind_retry_2": ControlPolicy(
        early_abort_threshold=0.15,
        max_retries=2,
    ),
    # Full uncertainty control: early abort + smart retry
    "full_uncertainty_1": ControlPolicy(
        early_abort_threshold=0.15,
        max_retries=1,
        retry_threshold=0.25,
    ),
    "full_uncertainty_2": ControlPolicy(
        early_abort_threshold=0.15,
        max_retries=2,
        retry_threshold=0.25,
    ),
}
