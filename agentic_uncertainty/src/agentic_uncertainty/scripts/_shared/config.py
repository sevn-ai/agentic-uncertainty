"""Configuration dataclasses for experiment runners.

Provides typed configuration objects to replace the many-parameter function
signatures in the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRunConfig:
    """Base configuration for agent-based experiments.

    Contains common parameters shared between exploration and review agents.
    """

    methods: list[str] = field(default_factory=lambda: ["direct"])
    step_limit: int = 100
    timeout: int = 900  # 15 minutes
    step_timeout: int = 120  # 2 minutes per step


@dataclass
class ExplorationConfig(AgentRunConfig):
    """Configuration for exploration agent experiments.

    Exploration agents run pre-execution to estimate task difficulty
    by exploring the repository without making changes.
    """

    pass


@dataclass
class ReviewConfig(AgentRunConfig):
    """Configuration for review agent experiments.

    Review agents run post-execution to estimate patch correctness
    by examining the applied changes.
    """

    cost_limit: float = 1.0
    traj_dir: Path | None = None


@dataclass
class CheckpointConfig(AgentRunConfig):
    """Configuration for checkpoint agent experiments.

    Checkpoint agents track confidence throughout task execution,
    eliciting confidence every N steps to detect overconfidence patterns.
    """

    confidence_interval: int = 5  # Elicit confidence every N steps

    def __post_init__(self) -> None:
        if self.confidence_interval <= 0:
            raise ValueError("confidence_interval must be >= 1")


@dataclass
class CheckpointPosthocConfig:
    """Configuration for post-hoc checkpoint analysis.

    Analyzes existing trajectories by eliciting confidence at checkpoint
    steps via side-channel model calls.
    """

    confidence_interval: int = 5
    traj_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.confidence_interval <= 0:
            raise ValueError("confidence_interval must be >= 1")


@dataclass
class MidExecutionConfig(AgentRunConfig):
    """Configuration for mid-execution trajectory evaluation.

    Evaluates an agent's trajectory at x% completion by running an
    exploration agent with the partial trajectory as context.
    """

    progress_fraction: float = 0.5  # What fraction of trajectory to show (0.0-1.0)
    traj_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.progress_fraction <= 1.0:
            raise ValueError("progress_fraction must be in (0.0, 1.0]")


@dataclass
class ExperimentConfig:
    """Top-level configuration for unified experiments.

    Combines instance selection, environment settings, and agent-specific
    configurations into a single typed object.
    """

    # Instance selection
    instance_ids: list[str]

    # Environment and model
    model: str = "gpt-5.2-codex"
    model_class: str = ""  # mini-swe-agent model class (e.g. "anthropic"); auto-detected if empty
    environment_class: str = "modal"
    parallel: int = 1

    # Paths
    ground_truth_path: Path | None = None
    output_dir: Path = field(default_factory=lambda: Path("results"))
    cache_dir: Path | None = None

    # Agent configurations (None means agent won't run)
    exploration: ExplorationConfig | None = None
    review: ReviewConfig | None = None
    checkpoint: CheckpointConfig | None = None
    checkpoint_posthoc: CheckpointPosthocConfig | None = None
    mid_execution: "MidExecutionConfig | None" = None

    # Execution options
    sequential_agents: bool = False
    watch: bool = False  # Stream agent steps live to console

    @property
    def active_agents(self) -> list[str]:
        """Return list of agent names that are configured to run."""
        agents = []
        if self.exploration is not None:
            agents.append("exploration")
        if self.review is not None:
            agents.append("review")
        if self.checkpoint is not None:
            agents.append("checkpoint")
        if self.checkpoint_posthoc is not None:
            agents.append("checkpoint_posthoc")
        if self.mid_execution is not None:
            agents.append("mid_execution")
        return agents

    def get_agent_config(self, name: str) -> AgentRunConfig | CheckpointPosthocConfig | None:
        """Get the configuration for a specific agent."""
        if name == "exploration":
            return self.exploration
        elif name == "review":
            return self.review
        elif name == "checkpoint":
            return self.checkpoint
        elif name == "checkpoint_posthoc":
            return self.checkpoint_posthoc
        elif name == "mid_execution":
            return self.mid_execution
        return None
