"""Agent that reviews patches by exploring the repo with the patch applied.

Used for post-execution confidence elicitation where the agent can:
1. See the proposed patch
2. Explore the repository (with patch applied)
3. Provide a confidence estimate that the bug is solved
"""

from minisweagent import Environment, Model
from minisweagent.agents.exploration import (
    ExplorationAgent,
    ExplorationAgentConfig,
)


class ReviewAgentConfig(ExplorationAgentConfig):
    """Configuration for review agent.

    Inherits all fields from ExplorationAgentConfig.
    Uses the same step limit; cost is tracked but not limited.
    """
    pass  # Uses inherited step_limit=100


class ReviewAgent(ExplorationAgent):
    """Agent that reviews patches by exploring the repo.

    Used for post-execution confidence elicitation. The agent:
    1. Receives the problem statement and proposed patch
    2. Explores the repository (with patch already applied)
    3. Provides a confidence estimate that the patch correctly solves the bug

    Inherits all read-only command validation from ExplorationAgent.
    The key difference is that it has access to the patch context and
    can see the modified files in the repo.
    """

    def __init__(
        self,
        model: Model,
        env: Environment,
        *,
        config_class: type = ReviewAgentConfig,
        **kwargs,
    ):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.patch: str | None = None

    def run(self, task: str, *, patch: str = "", **kwargs) -> tuple[str, str]:
        """Run review exploration with patch context.

        Args:
            task: The problem statement / issue description.
            patch: The proposed patch diff to review.
            **kwargs: Additional template variables.

        Returns:
            Tuple of (exit_status, final_output).
        """
        self.patch = patch
        # Make patch available in templates
        kwargs["patch"] = patch
        return super().run(task, **kwargs)

    def get_result(self) -> dict:
        """Get review result including confidence and patch."""
        result = super().get_result()
        result["patch"] = self.patch
        return result
