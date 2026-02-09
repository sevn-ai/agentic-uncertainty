"""Base class for agent-based uncertainty estimation.

Shared functionality for exploration and review agents using
mini-swe-agent's Docker infrastructure for SWE-bench environments.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import yaml

from agentic_uncertainty.config import AgentConfig, Settings, get_settings
from agentic_uncertainty.elicitation.base import UncertaintyEstimator

if TYPE_CHECKING:
    from agentic_uncertainty.data import Task

logger = logging.getLogger(__name__)


# Type for checkpoint callback: receives (messages, history, n_steps, cost)
CheckpointCallback = Callable[[list[dict] | None, list[dict], int, float], None]

# Type for step callback: receives (step_number, output_dict with action and output)
StepCallback = Callable[[int, dict], None]


@dataclass
class AgentResult:
    """Result from running an agent (exploration or review)."""

    confidence: float | None
    n_steps: int
    cost: float
    history: list[dict]
    exit_status: str
    final_output: str
    messages: list[dict] | None = None  # Full message history for inspector
    checkpoint_saved: bool = False  # Whether a checkpoint was saved on failure
    resumed_from_checkpoint: bool = False  # Whether this run resumed from checkpoint


def get_swebench_pro_image_name(instance_id: str, repo: str) -> str:
    """Get Docker Hub image name for a SWE-bench Pro instance.

    Uses the same logic as helper_code/image_uri.py for generating image URIs
    that match the pre-built images on Docker Hub (jefzda/sweap-images).

    Args:
        instance_id: The instance ID (e.g., instance_owner__repo-commit-version)
        repo: Repository in format "owner/repo"

    Returns:
        Docker image URI (e.g., jefzda/sweap-images:ansible.ansible-ansible__ansible-commit)
    """
    repo_base, repo_name_only = repo.lower().split("/")
    hsh = instance_id.replace("instance_", "")

    # Special cases for element-hq repos (to match pre-built images)
    if (
        instance_id
        == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan"
    ):
        repo_name_only = "element-web"
    elif "element-hq" in repo.lower() and "element-web" in repo.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    # All other repos: strip -vnan suffix
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    # Docker tag limit is 128 characters
    if len(tag) > 128:
        tag = tag[:128]

    return f"jefzda/sweap-images:{tag}"


def task_to_swebench_instance(task: Task) -> dict:
    """Convert an agentic_uncertainty Task to a mini-swe-agent instance dict.

    Args:
        task: Task object from agentic_uncertainty.

    Returns:
        Dictionary compatible with mini-swe-agent's get_sb_environment().
    """
    # Strip 'instance_' prefix from SWE-bench Pro instance_ids
    instance_id = task.instance_id
    if instance_id.startswith("instance_"):
        instance_id = instance_id[len("instance_") :]

    # Get Docker image name using same logic as trajectory generation
    image_name = get_swebench_pro_image_name(task.instance_id, task.repo)

    return {
        "instance_id": instance_id,
        "image_name": image_name,
        "repo": task.repo,
        "base_commit": task.base_commit,
        "problem_statement": task.problem_statement,
        "hints_text": task.hints_text,
        "patch": task.patch,
        "test_patch": task.test_patch,
        "version": task.version,
        "FAIL_TO_PASS": task.fail_to_pass,
        "PASS_TO_PASS": task.pass_to_pass,
    }


class AgentElicitation(UncertaintyEstimator):
    """Base class for agent-based elicitation (exploration and review).

    Uses mini-swe-agent's Docker infrastructure to run agents in
    pre-built SWE-bench environments with repositories already set up.
    """

    # Subclasses should override these
    METHODS: dict[str, str] = {}
    CONFIG_NAME: str = ""
    AGENT_CLASS: str = ""  # mini-swe-agent class name (e.g., "exploration", "review")
    DEFAULT_CONFIG: AgentConfig = AgentConfig()

    def __init__(
        self,
        method: str = "direct",
        step_limit: int | None = None,
        cost_limit: float | None = None,
        timeout: int | None = None,
        step_timeout: int | None = None,
        checkpoint_interval: int | None = None,
        environment_class: str | None = None,
        model: str | None = None,
        model_class: str = "",
        settings: Settings | None = None,
    ):
        """Initialize the agent elicitation.

        Args:
            method: Elicitation method name.
            step_limit: Maximum agent steps (uses class default if None).
            cost_limit: Maximum cost in $ (uses class default if None).
            timeout: Total timeout in seconds (uses class default if None).
            step_timeout: Per-step timeout in seconds (uses class default if None).
            checkpoint_interval: Save checkpoint every N steps (0 to disable).
            environment_class: Environment type ("docker", "singularity", "modal").
            model: Model name for the agent (uses class default if None).
            model_class: mini-swe-agent model class (e.g. "anthropic"). Auto-detected if empty.
            settings: Optional settings override.
        """
        # Don't call super().__init__ since we don't need the model client
        # The agent manages its own model
        self.settings = settings if settings is not None else get_settings()
        self.method = method
        self.step_limit = (
            step_limit if step_limit is not None else self.DEFAULT_CONFIG.step_limit
        )
        self.cost_limit = (
            cost_limit if cost_limit is not None else self.DEFAULT_CONFIG.cost_limit
        )
        self.timeout = timeout if timeout is not None else self.DEFAULT_CONFIG.timeout
        self.step_timeout = (
            step_timeout
            if step_timeout is not None
            else self.DEFAULT_CONFIG.step_timeout
        )
        self.checkpoint_interval = (
            checkpoint_interval
            if checkpoint_interval is not None
            else self.DEFAULT_CONFIG.checkpoint_interval
        )
        self.environment_class = (
            environment_class
            if environment_class is not None
            else self.DEFAULT_CONFIG.environment_class
        )
        self.model = model if model is not None else self.DEFAULT_CONFIG.model
        self.model_class = model_class

        if method not in self.METHODS:
            raise ValueError(
                f"Unknown {self.__class__.__name__} method: {method}. "
                f"Available: {list(self.METHODS.keys())}"
            )

    def _get_swebench_environment(self, task: Task):
        """Create a SWE-bench environment for a Task object."""
        instance = task_to_swebench_instance(task)
        return self._get_environment_for_instance(instance)

    def _get_environment_for_instance(self, instance: dict):
        """Create a SWE-bench environment for an instance dict.

        Uses mini-swe-agent's infrastructure to create an environment
        with the repository already cloned and set up.

        Args:
            instance: Instance dict with instance_id, image_name, repo, etc.

        Returns:
            Environment object from mini-swe-agent.
        """
        from minisweagent.run.extra.swebench import get_sb_environment

        config = self._load_agent_config()
        config.setdefault("environment", {})[
            "environment_class"
        ] = self.environment_class
        return get_sb_environment(config, instance)

    def _find_repo_dir_sync(self, env) -> str | None:
        """Find the git repository location in the container (synchronous).

        Some Docker images have the repo at /testbed, others at /app.
        Returns the repo path or None if not found.
        """
        # Use a short timeout for these discovery commands to avoid hanging
        cmd_timeout = 30  # 30 seconds should be plenty for simple commands

        try:
            # Try git rev-parse first (works if we're already in a git repo)
            result = env.execute(
                "git rev-parse --show-toplevel 2>/dev/null", timeout=cmd_timeout
            )
            repo_dir = result.get("output", "").strip()
            if repo_dir and repo_dir.startswith("/"):
                return repo_dir

            # Search common locations
            result = env.execute(
                'for d in /testbed /app /repo /workspace; do [ -d "$d/.git" ] && echo $d && break; done',
                timeout=cmd_timeout,
            )
            repo_dir = result.get("output", "").strip()
            if repo_dir and repo_dir.startswith("/"):
                return repo_dir

            # Last resort: find any .git directory
            result = env.execute(
                "find / -maxdepth 3 -type d -name '.git' 2>/dev/null | head -1 | xargs dirname 2>/dev/null",
                timeout=cmd_timeout,
            )
            repo_dir = result.get("output", "").strip()
            if repo_dir and repo_dir.startswith("/"):
                return repo_dir

        except Exception as e:
            logger.warning(f"Error finding repository: {e}")

        return None

    def _set_env_working_dir(self, env, repo_dir: str) -> None:
        """Set the environment's working directory to the repository."""
        if hasattr(env, "config") and hasattr(env.config, "cwd"):
            logger.info(f"Setting working directory to: {repo_dir}")
            env.config.cwd = repo_dir

    def _load_agent_config(self) -> dict:
        """Load agent configuration from YAML file.

        Returns:
            Configuration dictionary with agent, environment, and model sections.
        """
        try:
            from minisweagent.config import get_config_path

            config_path = get_config_path(f"extra/{self.CONFIG_NAME}")
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
        """Get the elicitation prompt for the configured method.

        Args:
            **kwargs: Context variables to substitute in the template.
        """
        template = self.METHODS[self.method]
        if isinstance(template, str) and template.endswith(".md"):
            raise RuntimeError(
                "Prompt-template elicitation methods were removed in the paper-only public release."
            )
        return str(template)

    def _create_error_result(self, exit_status: str, message: str) -> AgentResult:
        """Create an error AgentResult with no confidence (None).

        Error results have confidence=None to indicate that no valid confidence
        was obtained. Callers should filter these results out and log a warning.
        """
        logger.warning(
            "Agent encountered an error (%s): %s. "
            "This result will have no confidence value and should be filtered out.",
            exit_status,
            message,
        )
        return AgentResult(
            confidence=None,
            n_steps=0,
            cost=0.0,
            history=[],
            exit_status=exit_status,
            final_output=message,
        )

    def _extract_confidence_from_history(
        self, history: list[dict], final_output: str
    ) -> float | None:
        """Extract confidence from agent history when formal submission failed.

        This is a fallback for when the agent explored and reasoned about
        confidence but failed to use the exact submission command.

        Looks for patterns like:
        - <confidence>75</confidence>
        - confidence: 75
        - 75% confidence
        - my confidence is 75

        Returns:
            Confidence as float 0-1, or None if no valid confidence found.
        """
        # Combine all text for searching
        all_text = final_output + "\n"
        for entry in history:
            if "thought" in entry:
                all_text += str(entry["thought"]) + "\n"
            if "output" in entry:
                all_text += str(entry["output"]) + "\n"

        # Patterns to match probability/confidence values (0-100 scale)
        patterns = [
            r"\[p_success:\s*(\d+)\s*\]",  # Bracket format: [p_success:75] (preferred)
            r"\[confidence:\s*(\d+)\s*\]",  # Legacy bracket: [confidence:75]
            r"<confidence>\s*(\d+)\s*</confidence>",  # XML tag format
            r"confidence[:\s]+(\d+)\s*(?:%|percent)?",  # confidence: 75 or confidence 75%
            r"(\d+)\s*(?:%|percent)\s*confiden",  # 75% confident
            r"estimate[:\s]+(\d+)",  # estimate: 75
            r"rating[:\s]+(\d+)",  # rating: 75
        ]

        for pattern in patterns:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                try:
                    value = int(match.group(1))
                    if 0 <= value <= 100:
                        logger.info(
                            "Extracted confidence %d from history (pattern: %s)",
                            value,
                            pattern,
                        )
                        return value / 100.0
                except (ValueError, IndexError):
                    continue

        return None

    def _elicit_final_confidence(
        self, model, history: list[dict], problem_statement: str
    ) -> float | None:
        """Ask the model for a confidence estimate based on exploration history.

        This is a fallback when the agent didn't submit a confidence value.
        We summarize what the agent explored and ask for a confidence.

        Returns:
            Confidence as float 0-1, or None if elicitation failed.
        """
        if not history:
            return None

        # Build a summary of exploration (include all steps, truncate long outputs)
        exploration_summary = []
        for i, entry in enumerate(history, 1):
            cmd = entry.get("command", "")
            output = entry.get("output", "")
            if len(output) > 2000:
                output = output[:2000] + "\n... (truncated)"
            exploration_summary.append(f"Step {i}: {cmd}\n{output}")

        prompt = f"""Based on the following exploration of a repository to solve a bug, provide a confidence estimate.

TASK:
{problem_statement}

EXPLORATION:
{chr(10).join(exploration_summary)}

Based on this exploration, estimate the probability (0-100) that this task can be successfully solved.
Reply with ONLY a number between 0 and 100."""

        try:
            response = model.query([{"role": "user", "content": prompt}])
            content = response.get("content", "")

            # Parse number from response
            matches = re.findall(r"\b(\d{1,3})\b", content)
            for match in matches:
                value = int(match)
                if 0 <= value <= 100:
                    logger.info("Elicited final confidence: %d", value)
                    return value / 100.0
        except Exception as e:
            logger.warning("Failed to elicit final confidence: %s", e)

        return None

    async def _run_agent(
        self,
        env,
        problem_statement: str,
        checkpoint_callback: CheckpointCallback | None = None,
        step_callback: StepCallback | None = None,
        initial_messages: list[dict] | None = None,
        **run_kwargs,
    ) -> AgentResult:
        """Run a mini-swe-agent and return structured result.

        This is the shared agent execution logic used by both exploration
        and review elicitation. Subclasses specify AGENT_CLASS and pass
        any extra kwargs (e.g., patch= for review).

        Args:
            env: The mini-swe-agent environment (Docker/Modal/etc).
            problem_statement: The task description.
            checkpoint_callback: Optional callback to save checkpoints.
                Called with (messages, history, n_steps, cost) on timeout/error.
            step_callback: Optional callback for live step streaming.
                Called with (step_number, output_dict) after each agent step.
            initial_messages: Optional message history to resume from checkpoint.
            **run_kwargs: Extra arguments passed to agent.run().

        Returns:
            AgentResult with confidence, history, cost, etc.
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
        elicitation_prompt = self._get_elicitation_prompt(**run_kwargs)

        # Override limits from instance config
        # Note: step_limit limits API calls (not observable steps), so set higher to
        # account for retries on format errors. cost_limit=0 means no cost limit.
        agent_config = config.get("agent", {}).copy()
        agent_config["step_limit"] = self.step_limit
        agent_config["cost_limit"] = 0  # Cost is tracked but not limited

        # Create model and agent
        logger.info(f"Creating model: {self.model}")
        model_config = config.get("model", {}).copy()
        if self.model_class:
            model_config["model_class"] = self.model_class
        # Auto-detect Gemini models and set the appropriate model class
        if "gemini" in self.model.lower():
            model_config["model_class"] = "minisweagent.models.gemini.GeminiModel"
        model = get_model(self.model, model_config)
        logger.info(f"Model created, getting agent class: {self.AGENT_CLASS}")
        agent_class = get_agent_class(self.AGENT_CLASS)
        logger.info("Creating agent...")
        agent = agent_class(model, env, step_callback=step_callback, **agent_config)
        logger.info("Agent created")
        # Note: step_limit is already in agent_config (passed to agent constructor),
        # so it's available in templates via self.config.model_dump()
        agent.extra_template_vars = {
            "elicitation_prompt": elicitation_prompt,
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
        logger.info(f"Starting agent.run() with timeout={self.timeout}s...")
        try:
            exit_status, final_output = await asyncio.wait_for(
                asyncio.to_thread(agent.run, problem_statement, **run_kwargs),
                timeout=self.timeout,
            )
            logger.info(f"Agent.run() completed with status: {exit_status}")
        except asyncio.TimeoutError:
            exit_status = "Timeout"
            final_output = (
                f"{self.AGENT_CLASS.capitalize()} timed out after {self.timeout}s"
            )

            # Save checkpoint on timeout so work isn't lost
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
        # Capture full message history for inspector-compatible trajectory export
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
                elicited = self._elicit_final_confidence(
                    model, history, problem_statement
                )
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
