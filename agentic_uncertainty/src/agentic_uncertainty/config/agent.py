"""Agent configuration defaults.

Centralized configuration for exploration and review agents.
"""

from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for agent-based elicitation.

    Cost is tracked for metrics but not limited during execution.
    """

    step_limit: int = 30
    cost_limit: float = 1.0
    timeout: int = 1800  # 30 minutes (allows ~95 steps at 19s/step)
    step_timeout: int = 120  # Per-step timeout (command execution)
    environment_class: str = "modal"
    model: str = "gpt-5.2-codex"
    checkpoint_interval: int = 5  # Save checkpoint every N steps (0 to disable)
