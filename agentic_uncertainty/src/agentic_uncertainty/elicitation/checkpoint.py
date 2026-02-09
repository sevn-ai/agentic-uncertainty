"""Checkpoint-based uncertainty estimation.

Two approaches for tracking confidence throughout task execution:
1. Live in-band: CheckpointElicitation runs an agent that elicits confidence every K steps
2. Post-hoc: CheckpointPosthocElicitation analyzes existing trajectories via side-channel calls
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from agentic_uncertainty.config import AgentConfig
from agentic_uncertainty.elicitation.agent_base import AgentElicitation, AgentResult
from agentic_uncertainty.elicitation.base import EstimationResult, UncertaintyEstimator
from agentic_uncertainty.elicitation.retry import retry_async
from agentic_uncertainty.scripts._shared.shutdown import register_environment, unregister_environment

if TYPE_CHECKING:
    from pathlib import Path

    from agentic_uncertainty.data import Task

logger = logging.getLogger(__name__)

CHECKPOINT_POSTHOC_PROMPT_TEMPLATE = """## Checkpoint Probability Estimation

You are analyzing the progress of an AI coding agent working on a software engineering task.

### Task Description
{problem_statement}

### Agent Progress (up to step {step})
{context}

---

Based on the agent's progress so far at step {step}, estimate the probability (0-100) that this agent will **successfully resolve** the task.

Output your estimate in this exact format:
[p_success:NUMBER]

Replace NUMBER with an integer from 0 to 100:
- 0 = certainly will fail
- 50 = equal chance of success or failure
- 100 = certainly will succeed
"""


def parse_confidence(content: str) -> float | None:
    """Extract numeric probability/confidence from response text.

    Shared utility supporting multiple formats (checked in order):
    - Bracket format: "[p_success:75]" -> 0.75 (preferred)
    - Legacy bracket: "[confidence:75]" -> 0.75
    - XML format: "<confidence>75</confidence>" -> 0.75
    - Percentage: "75%" -> 0.75
    - Plain number: "75" -> 0.75 (if > 1, treated as percentage)

    Returns:
        Probability as float in [0, 1], or None if not found.
    """
    # Bracket format: [p_success:75] (preferred)
    p_success_match = re.search(r"\[p_success:\s*(\d+(?:\.\d+)?)\s*%?\s*\]", content, re.IGNORECASE)
    if p_success_match:
        value = float(p_success_match.group(1))
        return value / 100.0 if value > 1 else value

    # Legacy bracket format: [confidence:75]
    bracket_match = re.search(r"\[confidence:\s*(\d+(?:\.\d+)?)\s*%?\s*\]", content, re.IGNORECASE)
    if bracket_match:
        value = float(bracket_match.group(1))
        return value / 100.0 if value > 1 else value

    # XML format: <confidence>75</confidence>
    xml_match = re.search(r"<confidence>\s*(\d+(?:\.\d+)?)\s*%?\s*</confidence>", content, re.IGNORECASE)
    if xml_match:
        value = float(xml_match.group(1))
        return value / 100.0 if value > 1 else value

    # Percentage pattern: 75%
    percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", content)
    if percent_match:
        return float(percent_match.group(1)) / 100.0

    # Standalone number
    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", content)
    if number_match:
        value = float(number_match.group(1))
        if value > 1:
            return min(value / 100.0, 1.0)
        return value

    return None


class CheckpointElicitation(AgentElicitation):
    """Live in-band checkpoint elicitation using CheckpointAgent.

    Runs a mini-swe-agent that periodically elicits confidence during task
    execution, injecting confidence questions every K steps. This provides
    a confidence trace showing how the agent's confidence evolves.
    """

    METHODS = {"direct": "checkpoint_direct.md"}
    CONFIG_NAME = "default"  # Uses default agent config
    AGENT_CLASS = "checkpoint"  # Uses CheckpointAgent
    DEFAULT_CONFIG = AgentConfig()

    def __init__(
        self,
        confidence_interval: int = 5,
        client=None,  # Accepted but not used (agent manages its own model)
        **kwargs,
    ):
        """Initialize checkpoint elicitation.

        Args:
            confidence_interval: Elicit confidence every N steps.
            client: Ignored (agent manages its own model).
            **kwargs: Additional arguments passed to AgentElicitation.
        """
        # Set default method if not provided
        if "method" not in kwargs:
            kwargs["method"] = "direct"
        if confidence_interval <= 0:
            raise ValueError("confidence_interval must be >= 1")
        super().__init__(**kwargs)
        self.confidence_interval = confidence_interval

    async def estimate(self, task: Task) -> EstimationResult:
        """Run checkpoint agent and return confidence trace.

        Args:
            task: The task to estimate.

        Returns:
            EstimationResult with confidence trace in metadata.
        """
        result = await self._run_checkpoint(task)

        if result.confidence is None:
            logger.warning(
                "Checkpoint elicitation for task %s did not return final confidence.",
                task.instance_id,
            )

        metadata = {
            "method": "checkpoint_direct",
            "n_steps": result.n_steps,
            "checkpoint_cost": result.cost,
            "exit_status": result.exit_status,
            "confidence_trace": getattr(result, "confidence_trace", []),
            "exploration_history": result.history,
            "messages": result.messages,
        }

        return EstimationResult(
            probability=result.confidence,
            raw_response=result.final_output,
            metadata=metadata,
        )

    @retry_async(logger)
    async def _run_checkpoint(self, task: Task) -> AgentResult:
        """Run checkpoint agent with retries."""
        env = None
        try:
            logger.info(f"Starting environment for {task.instance_id}")
            env = await asyncio.to_thread(self._get_swebench_environment, task)
            register_environment(env)
            logger.info(f"Environment ready for {task.instance_id}")

            # Find repo and set working directory
            repo_dir = await asyncio.to_thread(self._find_repo_dir_sync, env)
            if repo_dir:
                self._set_env_working_dir(env, repo_dir)

            result = await self._run_checkpoint_agent(env, task.problem_statement)
            return result
        except Exception as e:
            logger.error(f"Error in checkpoint for {task.instance_id}: {e}")
            return self._create_error_result(exit_status=type(e).__name__, message=str(e))
        finally:
            if env is not None:
                unregister_environment(env)
                if hasattr(env, "stop"):
                    try:
                        env.stop()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up environment: {cleanup_error}")

    async def _run_checkpoint_agent(self, env, problem_statement: str) -> AgentResult:
        """Run the checkpoint agent."""
        try:
            from minisweagent.agents import get_agent_class
            from minisweagent.models import get_model
        except ImportError as e:
            return self._create_error_result(
                exit_status="ImportError",
                message=f"mini-swe-agent not available: {e}",
            )

        config = self._load_agent_config()

        # Configure agent with checkpoint interval
        agent_config = config.get("agent", {}).copy()
        agent_config["step_limit"] = self.step_limit
        agent_config["cost_limit"] = 0
        agent_config["confidence_interval"] = self.confidence_interval

        # Create model and agent
        model_config = config.get("model", {}).copy()
        model = get_model(self.model, model_config)
        agent_class = get_agent_class(self.AGENT_CLASS)
        agent = agent_class(model, env, **agent_config)

        # Run with timeout
        try:
            exit_status, final_output = await asyncio.wait_for(
                asyncio.to_thread(agent.run, problem_statement),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            exit_status = "Timeout"
            final_output = f"Checkpoint agent timed out after {self.timeout}s"
        except Exception as e:
            exit_status = "Error"
            final_output = str(e)

        # Extract results from agent
        result = agent.get_result()
        confidence_trace = result.get("confidence_trace", [])

        # Create result with confidence trace
        agent_result = AgentResult(
            confidence=result.get("confidence"),
            n_steps=result.get("n_steps", 0),
            cost=model.cost,
            history=result.get("exploration_history", []),
            exit_status=exit_status,
            final_output=final_output,
            messages=getattr(agent, "messages", None),
        )
        # Attach confidence trace as extra attribute
        agent_result.confidence_trace = confidence_trace

        return agent_result


class CheckpointPosthocElicitation(UncertaintyEstimator):
    """Post-hoc checkpoint analysis of existing trajectories.

    Analyzes saved trajectories by eliciting confidence at checkpoint steps
    (K, 2K, 3K, ...) via side-channel model calls. This allows studying
    confidence evolution without re-running agents.
    """

    def __init__(
        self,
        confidence_interval: int = 5,
        **kwargs,
    ):
        """Initialize post-hoc checkpoint elicitation.

        Args:
            confidence_interval: Steps between confidence checkpoints.
            **kwargs: Arguments passed to UncertaintyEstimator.
        """
        if confidence_interval <= 0:
            raise ValueError("confidence_interval must be >= 1")
        super().__init__(**kwargs)
        self.confidence_interval = confidence_interval

    async def estimate(self, task: Task) -> EstimationResult:
        """Not directly usable - use estimate_trajectory instead."""
        raise NotImplementedError(
            "CheckpointPosthocElicitation requires a trajectory. "
            "Use estimate_trajectory(trajectory_path, resolved) instead."
        )

    async def estimate_trajectory(
        self,
        trajectory_path: str | Path,
        resolved: bool,
    ) -> EstimationResult:
        """Process a trajectory and elicit confidence at checkpoints.

        Args:
            trajectory_path: Path to trajectory JSON file.
            resolved: Whether the task was actually resolved (ground truth).

        Returns:
            EstimationResult with confidence_trace in metadata.
        """
        # Load trajectory
        with open(trajectory_path) as f:
            traj = json.load(f)

        messages = traj.get("messages", [])
        if not messages:
            return EstimationResult(
                probability=None,
                raw_response="Empty trajectory",
                metadata={"error": "Empty trajectory", "resolved": resolved},
            )

        # Extract problem statement from first user message
        problem_statement = self._extract_problem_statement(messages)

        # Find agent turns (assistant messages)
        agent_turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]

        if not agent_turns:
            return EstimationResult(
                probability=None,
                raw_response="No agent turns found",
                metadata={"error": "No agent turns", "resolved": resolved},
            )

        # Elicit confidence at checkpoint steps
        confidence_trace = []
        for step in range(self.confidence_interval, len(agent_turns) + 1, self.confidence_interval):
            # Get context up to this step
            context_end = agent_turns[step - 1] + 1
            context = messages[:context_end]

            conf = await self._elicit_confidence_at_step(context, step, problem_statement)
            confidence_trace.append({"step": step, "confidence": conf})

        # Elicit final confidence at the end (avoid duplicate if already checkpointed)
        final_step = len(agent_turns)
        if confidence_trace and confidence_trace[-1]["step"] == final_step:
            final_conf = confidence_trace[-1]["confidence"]
        else:
            final_conf = await self._elicit_confidence_at_step(messages, final_step, problem_statement)

        return EstimationResult(
            probability=final_conf,
            raw_response=f"Confidence trace with {len(confidence_trace)} checkpoints",
            metadata={
                "method": "checkpoint_posthoc",
                "confidence_trace": confidence_trace,
                "final_confidence": final_conf,
                "final_step": final_step,
                "resolved": resolved,
                "total_agent_turns": len(agent_turns),
            },
        )

    def _extract_problem_statement(self, messages: list[dict]) -> str:
        """Extract the problem statement from the trajectory messages.

        Looks for the task description in the first user message, typically
        within <pr_description> tags or similar markers.

        Args:
            messages: All messages from the trajectory.

        Returns:
            The extracted problem statement, or a fallback if not found.
        """
        # Find first user message (usually contains the task)
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")

                # Try to extract from <pr_description> tags
                import re
                pr_match = re.search(
                    r"<pr_description>\s*(.*?)\s*</pr_description>",
                    content,
                    re.DOTALL,
                )
                if pr_match:
                    return pr_match.group(1).strip()

                # Try to extract from <task> tags
                task_match = re.search(
                    r"<task>\s*(.*?)\s*</task>",
                    content,
                    re.DOTALL,
                )
                if task_match:
                    return task_match.group(1).strip()

                # Try to extract from <problem_statement> tags
                ps_match = re.search(
                    r"<problem_statement>\s*(.*?)\s*</problem_statement>",
                    content,
                    re.DOTALL,
                )
                if ps_match:
                    return ps_match.group(1).strip()

                # If no tags found but content is reasonable length, use it
                if len(content) < 5000:
                    return content

                # Truncate if too long
                return content[:3000] + "\n... (truncated)"

        return "(Problem statement not found)"

    async def _elicit_confidence_at_step(
        self,
        context_messages: list[dict],
        step: int,
        problem_statement: str,
        max_retries: int = 3,
    ) -> float | None:
        """Elicit confidence given trajectory context up to a certain step.

        Args:
            context_messages: Messages up to the checkpoint.
            step: Current step number.
            problem_statement: The task description extracted from the trajectory.
            max_retries: Maximum number of retries on failure.

        Returns:
            Confidence value in [0, 1], or None if parsing failed after all retries.
        """
        # Build context summary (skip system message and first user message since
        # problem statement is now shown separately)
        context_text = self._format_context(context_messages, skip_initial=True)

        prompt = CHECKPOINT_POSTHOC_PROMPT_TEMPLATE.format(
            step=step,
            context=context_text,
            problem_statement=problem_statement,
        )

        # Retry loop for robustness against API errors and parsing failures
        last_error = None
        for attempt in range(max_retries):
            try:
                # Query model
                response = await self._call_model_async(prompt, temperature=0.0)

                # Parse confidence
                confidence = parse_confidence(response)
                if confidence is not None:
                    return confidence

                # Parsing failed - log and retry
                logger.warning(
                    f"Step {step} attempt {attempt + 1}: failed to parse confidence from response"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Step {step} attempt {attempt + 1} failed: {e}"
                )

            # Brief delay before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        if last_error:
            logger.error(f"Step {step}: all {max_retries} attempts failed, last error: {last_error}")
        else:
            logger.error(f"Step {step}: all {max_retries} attempts failed to parse confidence")
        return None

    def _format_context(self, messages: list[dict], skip_initial: bool = False) -> str:
        """Format messages into a context string for the prompt.

        Args:
            messages: Messages to format.
            skip_initial: If True, skip the system message and first user message
                (since problem statement is shown separately).

        Returns:
            Formatted context string.
        """
        parts = []
        start_idx = 0

        if skip_initial:
            # Skip system message and first user message
            for i, msg in enumerate(messages):
                if msg.get("role") == "user":
                    start_idx = i + 1
                    break

        for i, msg in enumerate(messages[start_idx:], start=start_idx):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 3000:
                content = content[:3000] + "\n... (truncated)"
            parts.append(f"[{role.upper()}]\n{content}")
        return "\n\n".join(parts)
