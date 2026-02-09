"""Exploration-based uncertainty estimation.

Runs a mini-swe-agent in exploration mode to gather information about the
repository before providing a confidence estimate. The agent uses read-only
commands (cat, grep, ls, find, git log, etc.) and cannot modify files.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import yaml

from agentic_uncertainty.config import AgentConfig
from agentic_uncertainty.data import Task
from agentic_uncertainty.elicitation.agent_base import AgentElicitation, AgentResult, StepCallback
from agentic_uncertainty.elicitation.base import EstimationResult
from agentic_uncertainty.elicitation.retry import RETRYABLE_EXCEPTIONS, retry_async
from agentic_uncertainty.scripts._shared.shutdown import register_environment, unregister_environment

if TYPE_CHECKING:
    from agentic_uncertainty.scripts._shared.cache import ResultCache

logger = logging.getLogger(__name__)

# Available exploration methods (method -> config name)
EXPLORATION_METHODS = {"direct": "exploration_direct"}


class ExplorationElicitation(AgentElicitation):
    """Run exploration agent to estimate task success probability."""

    METHODS = EXPLORATION_METHODS
    CONFIG_NAME = "exploration"
    AGENT_CLASS = "exploration"
    DEFAULT_CONFIG = AgentConfig()  # Uses class defaults (step_limit=30, cost_limit=0.5)

    def _load_agent_config(self) -> dict:
        """Load agent configuration from method-specific YAML."""
        try:
            from minisweagent.config import get_config_path

            config_name = EXPLORATION_METHODS[self.method]
            config_path = get_config_path(f"extra/{config_name}")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception:
            # Fallback to default config structure
            config = {
                "agent": {},
                "environment": {
                    "env": {"PAGER": "cat", "LESS": "-R"},
                },
                "model": {},
            }

        # Apply per-step timeout to environment
        config.setdefault("environment", {})
        config["environment"]["timeout"] = self.step_timeout

        return config

    def _get_elicitation_prompt(self, **kwargs) -> str:
        """Exploration prompts are fully defined in the YAML configs."""
        return ""

    async def estimate(
        self,
        task: Task,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> EstimationResult:
        """Estimate success probability by running exploration agent.

        Args:
            task: The task to estimate.
            cache: Optional cache for checkpointing. If provided, checkpoints
                will be saved on timeout/error and can be resumed.
            step_callback: Optional callback for live step streaming.
        """
        result = await self._run_exploration(task, cache=cache, step_callback=step_callback)

        if result.confidence is None:
            logger.warning(
                "Exploration elicitation for task %s did not return a valid confidence.",
                task.instance_id,
            )

        # Include checkpoint info in metadata
        metadata = {
            "method": f"exploration_{self.method}",
            "n_steps": result.n_steps,
            "exploration_cost": result.cost,
            "exit_status": result.exit_status,
            "exploration_history": result.history,
            "messages": result.messages,  # Full message history for inspector
        }
        if result.checkpoint_saved:
            metadata["checkpoint_saved"] = True
        if result.resumed_from_checkpoint:
            metadata["resumed_from_checkpoint"] = True

        return EstimationResult(
            probability=result.confidence,
            raw_response=result.final_output,
            metadata=metadata,
        )

    async def _run_exploration(
        self,
        task: Task,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run the exploration agent on the task with retries."""
        return await self._run_exploration_with_retry(task, cache=cache, step_callback=step_callback)

    @retry_async(logger)
    async def _run_exploration_with_retry(
        self,
        task: Task,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run exploration with automatic retries on transient failures."""
        env = None

        # Check for existing checkpoint to resume from
        initial_messages = None
        if cache:
            checkpoint = cache.load_checkpoint(
                method=f"exploration_{self.method}",
                instance_id=task.instance_id,
            )
            if checkpoint:
                initial_messages = checkpoint.get("messages")
                logger.info(
                    "Found checkpoint for %s: %d messages, %d steps",
                    task.instance_id,
                    len(initial_messages or []),
                    checkpoint.get("n_steps", 0),
                )

        # Create checkpoint callback if cache is provided
        def checkpoint_callback(messages, history, n_steps, cost):
            if cache:
                cache.save_checkpoint(
                    method=f"exploration_{self.method}",
                    instance_id=task.instance_id,
                    messages=messages,
                    exploration_history=history,
                    n_steps=n_steps,
                    cost=cost,
                )

        try:
            logger.info(f"Starting environment for {task.instance_id}")
            env = await asyncio.to_thread(self._get_swebench_environment, task)
            register_environment(env)  # Track for graceful shutdown
            logger.info(f"Environment ready for {task.instance_id}")

            # Find the repository location and set it as the working directory
            repo_dir = await asyncio.to_thread(self._find_repo_dir_sync, env)
            if repo_dir:
                self._set_env_working_dir(env, repo_dir)

            result = await self._run_agent(
                env,
                task.problem_statement,
                checkpoint_callback=checkpoint_callback if cache else None,
                step_callback=step_callback,
                initial_messages=initial_messages,
            )

            # If successful, delete checkpoint
            if cache and result.confidence is not None and result.exit_status not in ("Timeout", "Error"):
                cache.delete_checkpoint(
                    method=f"exploration_{self.method}",
                    instance_id=task.instance_id,
                )

            # If the result indicates a timeout AND we have no confidence, raise to retry
            # But if we got a valid fallback confidence, don't waste it by retrying
            if result.exit_status == "Timeout" and result.confidence is None:
                raise asyncio.TimeoutError(f"Agent timed out: {result.final_output}")

            return result
        except RETRYABLE_EXCEPTIONS:
            # Re-raise retryable exceptions to trigger retry
            raise
        except Exception as e:
            logger.error(f"Error in exploration for {task.instance_id}: {e}")
            return self._create_error_result(exit_status=type(e).__name__, message=str(e))
        finally:
            if env is not None:
                unregister_environment(env)  # Remove from tracking before cleanup
                if hasattr(env, "stop"):
                    try:
                        env.stop()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up environment: {cleanup_error}")


def ExplorationDirectElicitation(**kwargs) -> ExplorationElicitation:
    """Create exploration elicitation with direct method."""
    return ExplorationElicitation(method="direct", **kwargs)
