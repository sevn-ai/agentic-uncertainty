"""Mid-execution trajectory evaluation.

Evaluates an agent's trajectory at x% completion by:
1. Loading the trajectory
2. Extracting messages up to x% of steps
3. Running exploration agent with trajectory context

The evaluating agent can explore the repository (at base commit) and
uses the partial trajectory as context to estimate success probability.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import yaml

from agentic_uncertainty.config import AgentConfig
from agentic_uncertainty.elicitation.agent_base import (
    AgentElicitation,
    AgentResult,
    StepCallback,
)
from agentic_uncertainty.elicitation.base import EstimationResult
from agentic_uncertainty.elicitation.retry import RETRYABLE_EXCEPTIONS, retry_async
from agentic_uncertainty.scripts._shared.shutdown import register_environment, unregister_environment

if TYPE_CHECKING:
    from agentic_uncertainty.scripts._shared.cache import ResultCache

logger = logging.getLogger(__name__)

# Available mid-execution methods (method -> config name)
MID_EXECUTION_METHODS = {"direct": "mid_execution_direct"}


class MidExecutionElicitation(AgentElicitation):
    """Evaluate trajectory at x% completion.

    Runs an exploration agent that can examine the repository while having
    access to the partial trajectory context showing what the original
    agent has done so far.
    """

    METHODS = MID_EXECUTION_METHODS
    CONFIG_NAME = "mid_execution"
    AGENT_CLASS = "exploration"  # Reuse ExplorationAgent
    DEFAULT_CONFIG = AgentConfig()

    def __init__(
        self,
        progress_fraction: float = 0.5,
        method: str = "direct",
        **kwargs,
    ):
        """Initialize mid-execution elicitation.

        Args:
            progress_fraction: What fraction of trajectory to show (0.0-1.0).
            method: Elicitation method name.
            **kwargs: Additional arguments passed to AgentElicitation.
        """
        super().__init__(method=method, **kwargs)
        if not 0.0 < progress_fraction <= 1.0:
            raise ValueError("progress_fraction must be in (0.0, 1.0]")
        self.progress_fraction = progress_fraction

    def _load_agent_config(self) -> dict:
        """Load agent configuration from method-specific YAML."""
        try:
            from minisweagent.config import get_config_path

            config_name = MID_EXECUTION_METHODS[self.method]
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
        """Mid-execution prompts are fully defined in the YAML configs."""
        return ""

    async def estimate(self, task):
        """Not used - use estimate_trajectory instead."""
        raise NotImplementedError("Use estimate_trajectory() for mid-execution elicitation")

    async def estimate_trajectory(
        self,
        problem_statement: str,
        trajectory: dict,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> EstimationResult:
        """Estimate success probability at mid-execution point.

        Args:
            problem_statement: The task description.
            trajectory: Parsed trajectory dict with "messages" key.
            repo: Repository in format "owner/repo".
            base_commit: Base commit hash.
            instance_id: Optional instance ID for caching.
            cache: Optional cache for checkpointing.
            step_callback: Optional callback for live step streaming.

        Returns:
            EstimationResult with probability and metadata.
        """
        # 1. Calculate checkpoint step
        messages = trajectory.get("messages", [])
        assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        total_steps = len(assistant_indices)

        if total_steps == 0:
            return EstimationResult(
                probability=None,
                raw_response="No agent turns found in trajectory",
                metadata={"error": "No agent turns", "progress_fraction": self.progress_fraction},
            )

        checkpoint_step = max(1, int(total_steps * self.progress_fraction))

        # 2. Format partial trajectory
        # Get messages up to and including the checkpoint_step-th assistant message
        if checkpoint_step > 0:
            checkpoint_msg_idx = assistant_indices[checkpoint_step - 1] + 1
        else:
            checkpoint_msg_idx = 0
        partial_messages = messages[:checkpoint_msg_idx]
        trajectory_context = self._format_trajectory_context(partial_messages)

        # 3. Run exploration agent with trajectory context
        result = await self._run_mid_execution(
            problem_statement=problem_statement,
            trajectory_context=trajectory_context,
            n_steps=checkpoint_step,
            total_steps=total_steps,
            progress_percent=int(self.progress_fraction * 100),
            repo=repo,
            base_commit=base_commit,
            instance_id=instance_id,
            cache=cache,
            step_callback=step_callback,
        )

        if result.confidence is None:
            logger.warning(
                "Mid-execution elicitation for instance %s did not return a valid confidence.",
                instance_id,
            )

        # Include checkpoint info in metadata
        metadata = {
            "method": f"mid_execution_{self.method}",
            "progress_fraction": self.progress_fraction,
            "progress_percent": int(self.progress_fraction * 100),
            "checkpoint_step": checkpoint_step,
            "total_trajectory_steps": total_steps,
            "n_steps": result.n_steps,
            "mid_execution_cost": result.cost,
            "exit_status": result.exit_status,
            "exploration_history": result.history,
            "messages": result.messages,
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

    def _format_trajectory_context(
        self,
        messages: list[dict],
        max_chars: int = 15000,
    ) -> str:
        """Format messages as readable trajectory context.

        Args:
            messages: Messages to format (up to checkpoint).
            max_chars: Maximum characters (truncates from beginning if exceeded).

        Returns:
            Formatted string representation of the trajectory.
        """
        parts = []
        step = 0

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "assistant":
                step += 1
                # Truncate long assistant content
                if len(content) > 2000:
                    content = content[:2000] + "\n... (truncated)"
                parts.append(f"--- Step {step} (Agent) ---\n{content}")
            elif role == "user":
                # Check if this looks like command output (has returncode marker)
                if "<returncode>" in content or "returncode" in content.lower():
                    # Command output - truncate heavily
                    if len(content) > 1000:
                        content = content[:1000] + "\n... (truncated)"
                    parts.append(f"--- Output ---\n{content}")
                # Skip the initial task message (first user message is usually the task)
                elif step == 0 and len(parts) == 0:
                    continue
                else:
                    # Other user messages (shouldn't be common)
                    if len(content) > 500:
                        content = content[:500] + "\n... (truncated)"
                    parts.append(f"--- User ---\n{content}")

        result = "\n\n".join(parts)

        # Truncate from beginning if too long (keep recent context)
        if len(result) > max_chars:
            result = "... [earlier steps truncated] ...\n\n" + result[-max_chars:]

        return result

    async def _run_mid_execution(
        self,
        problem_statement: str,
        trajectory_context: str,
        n_steps: int,
        total_steps: int,
        progress_percent: int,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run the mid-execution evaluation agent with retries."""
        return await self._run_mid_execution_with_retry(
            problem_statement=problem_statement,
            trajectory_context=trajectory_context,
            n_steps=n_steps,
            total_steps=total_steps,
            progress_percent=progress_percent,
            repo=repo,
            base_commit=base_commit,
            instance_id=instance_id,
            cache=cache,
            step_callback=step_callback,
        )

    @retry_async(logger)
    async def _run_mid_execution_with_retry(
        self,
        problem_statement: str,
        trajectory_context: str,
        n_steps: int,
        total_steps: int,
        progress_percent: int,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run mid-execution evaluation with automatic retries on transient failures."""
        from agentic_uncertainty.elicitation.agent_base import get_swebench_pro_image_name

        env = None

        # Build instance dict for Docker image lookup
        clean_id = instance_id.replace("instance_", "") if instance_id else f"{repo.replace('/', '__')}__{base_commit[:8]}"
        full_id = instance_id or f"instance_{repo.replace('/', '__')}__{base_commit[:8]}"

        # Check for existing checkpoint to resume from
        initial_messages = None
        if cache and instance_id:
            checkpoint = cache.load_checkpoint(
                method=f"mid_execution_{self.method}",
                instance_id=instance_id,
            )
            if checkpoint:
                initial_messages = checkpoint.get("messages")
                logger.info(
                    "Found checkpoint for %s: %d messages, %d steps",
                    instance_id,
                    len(initial_messages or []),
                    checkpoint.get("n_steps", 0),
                )

        # Create checkpoint callback if cache is provided
        def checkpoint_callback(messages, history, n_steps_done, cost):
            if cache and instance_id:
                cache.save_checkpoint(
                    method=f"mid_execution_{self.method}",
                    instance_id=instance_id,
                    messages=messages,
                    exploration_history=history,
                    n_steps=n_steps_done,
                    cost=cost,
                )

        try:
            instance = {
                "instance_id": clean_id,
                "image_name": get_swebench_pro_image_name(full_id, repo),
                "repo": repo,
                "base_commit": base_commit,
                "problem_statement": problem_statement,
                "hints_text": "",
                "patch": "",
                "test_patch": "",
                "version": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }

            logger.info(f"Starting environment for mid-execution evaluation: {clean_id}")
            env = await asyncio.to_thread(self._get_environment_for_instance, instance)
            register_environment(env)
            logger.info(f"Environment ready for {clean_id}")

            # Find the repository location and set it as the working directory
            repo_dir = await asyncio.to_thread(self._find_repo_dir_sync, env)
            if repo_dir:
                self._set_env_working_dir(env, repo_dir)

            # Run the agent with trajectory context as extra template vars
            result = await self._run_agent_with_trajectory_context(
                env,
                problem_statement,
                trajectory_context=trajectory_context,
                n_steps=n_steps,
                total_steps=total_steps,
                progress_percent=progress_percent,
                checkpoint_callback=checkpoint_callback if cache and instance_id else None,
                step_callback=step_callback,
                initial_messages=initial_messages,
            )

            # If successful, delete checkpoint
            if cache and instance_id and result.confidence is not None and result.exit_status not in ("Timeout", "Error"):
                cache.delete_checkpoint(
                    method=f"mid_execution_{self.method}",
                    instance_id=instance_id,
                )

            # If the result indicates a timeout or transient error, raise to trigger retry
            if result.exit_status == "Timeout":
                raise asyncio.TimeoutError(f"Agent timed out: {result.final_output}")

            return result

        except RETRYABLE_EXCEPTIONS:
            # Re-raise retryable exceptions to trigger retry
            raise
        except Exception as e:
            logger.error(f"Error in mid-execution for {instance_id}: {e}")
            return self._create_error_result(exit_status=type(e).__name__, message=str(e))
        finally:
            if env is not None:
                unregister_environment(env)
                if hasattr(env, "stop"):
                    try:
                        env.stop()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up environment: {cleanup_error}")

    async def _run_agent_with_trajectory_context(
        self,
        env,
        problem_statement: str,
        trajectory_context: str,
        n_steps: int,
        total_steps: int,
        progress_percent: int,
        checkpoint_callback=None,
        step_callback: StepCallback | None = None,
        initial_messages: list[dict] | None = None,
    ) -> AgentResult:
        """Run a mini-swe-agent with trajectory context in template vars.

        Similar to _run_agent but adds trajectory-specific template variables.
        """
        try:
            from minisweagent.agents import get_agent_class
            from minisweagent.models import get_model
        except ImportError as e:
            return self._create_error_result(
                exit_status="ImportError",
                message=f"mini-swe-agent not available: {e}",
            )

        config = self._load_agent_config()

        # Override limits from instance config
        agent_config = config.get("agent", {}).copy()
        agent_config["step_limit"] = self.step_limit
        agent_config["cost_limit"] = 0  # Cost is tracked but not limited

        # Create model and agent
        model_config = config.get("model", {}).copy()
        if self.model_class:
            model_config["model_class"] = self.model_class
        # Auto-detect Gemini models and set the appropriate model class
        if "gemini" in self.model.lower():
            model_config["model_class"] = "minisweagent.models.gemini.GeminiModel"
        model = get_model(self.model, model_config)
        agent_class = get_agent_class(self.AGENT_CLASS)
        agent = agent_class(model, env, step_callback=step_callback, **agent_config)

        # Set extra template vars including trajectory context
        agent.extra_template_vars = {
            "trajectory_context": trajectory_context,
            "n_steps": n_steps,
            "total_steps": total_steps,
            "progress_percent": progress_percent,
        }

        # Resume from checkpoint if initial_messages provided
        resumed_from_checkpoint = False
        if initial_messages:
            try:
                agent.messages = initial_messages
                resumed_from_checkpoint = True
                logger.info(
                    "Resuming agent from checkpoint with %d messages",
                    len(initial_messages),
                )
            except Exception as e:
                logger.warning("Failed to restore checkpoint messages: %s", e)

        # Track whether checkpoint was saved on failure
        checkpoint_saved = False

        # Run with timeout
        try:
            exit_status, final_output = await asyncio.wait_for(
                asyncio.to_thread(agent.run, problem_statement),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            exit_status = "Timeout"
            final_output = f"Mid-execution evaluation timed out after {self.timeout}s"

            # Save checkpoint on timeout
            if checkpoint_callback:
                try:
                    messages = getattr(agent, "messages", None)
                    history = getattr(agent, "exploration_history", [])
                    checkpoint_callback(messages, history, len(history), model.cost)
                    checkpoint_saved = True
                    logger.info(
                        "Saved checkpoint on timeout: %d steps, cost=$%.2f",
                        len(history),
                        model.cost,
                    )
                except Exception as e:
                    logger.warning("Failed to save checkpoint: %s", e)

        except Exception as e:
            exit_status = "Error"
            final_output = str(e)

            # Save checkpoint on error
            if checkpoint_callback:
                try:
                    messages = getattr(agent, "messages", None)
                    history = getattr(agent, "exploration_history", [])
                    checkpoint_callback(messages, history, len(history), model.cost)
                    checkpoint_saved = True
                    logger.info(
                        "Saved checkpoint on error: %d steps, cost=$%.2f",
                        len(history),
                        model.cost,
                    )
                except Exception as cp_err:
                    logger.warning("Failed to save checkpoint: %s", cp_err)

        # Get confidence, with fallback extraction from history
        confidence = agent.final_confidence
        history = agent.exploration_history
        messages = getattr(agent, "messages", None)

        if confidence is None and history:
            # Try to extract confidence from agent's text output
            extracted = self._extract_confidence_from_history(history, final_output)
            if extracted is not None:
                confidence = extracted
                logger.info(
                    "Using fallback-extracted confidence: %.2f (found in text)",
                    confidence,
                )
            else:
                # Ask the model directly for a confidence estimate
                elicited = self._elicit_final_confidence(model, history, problem_statement)
                if elicited is not None:
                    confidence = elicited
                    logger.info(
                        "Using elicited fallback confidence: %.2f (agent did not submit formally)",
                        confidence,
                    )

        return AgentResult(
            confidence=confidence,
            n_steps=len(history),
            cost=model.cost,
            history=history,
            exit_status=exit_status,
            final_output=final_output,
            messages=messages,
            checkpoint_saved=checkpoint_saved,
            resumed_from_checkpoint=resumed_from_checkpoint,
        )


def MidExecutionDirectElicitation(**kwargs) -> MidExecutionElicitation:
    """Create mid-execution elicitation with direct method."""
    return MidExecutionElicitation(method="direct", **kwargs)
