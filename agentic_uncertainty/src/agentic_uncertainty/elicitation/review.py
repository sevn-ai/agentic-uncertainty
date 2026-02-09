"""Review-based uncertainty estimation.

Runs a mini-swe-agent in review mode to evaluate a patch after generation.
The agent explores the repository (with patch applied) using read-only
commands and provides a confidence estimate that the bug is correctly solved.
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
    get_swebench_pro_image_name,
)
from agentic_uncertainty.elicitation.base import EstimationResult
from agentic_uncertainty.elicitation.retry import RETRYABLE_EXCEPTIONS, retry_async
from agentic_uncertainty.scripts._shared.shutdown import register_environment, unregister_environment

if TYPE_CHECKING:
    from agentic_uncertainty.scripts._shared.cache import ResultCache

logger = logging.getLogger(__name__)

# Available review methods (method -> config name)
REVIEW_METHODS = {"direct": "review_direct", "adversarial": "review_adversarial"}


class ReviewElicitation(AgentElicitation):
    """Run review agent to estimate patch correctness probability."""

    METHODS = REVIEW_METHODS
    CONFIG_NAME = "review"
    AGENT_CLASS = "review"
    DEFAULT_CONFIG = AgentConfig()

    def _load_agent_config(self) -> dict:
        """Load agent configuration from method-specific YAML."""
        try:
            from minisweagent.config import get_config_path

            config_name = REVIEW_METHODS[self.method]
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
        """Review prompts are fully defined in the YAML configs."""
        return ""

    async def estimate(self, task):
        """Not used - use estimate_patch instead."""
        raise NotImplementedError("Use estimate_patch() for review elicitation")

    async def estimate_patch(
        self,
        problem_statement: str,
        patch: str,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> EstimationResult:
        """Estimate patch correctness by running review agent.

        Args:
            problem_statement: The task description.
            patch: The git diff patch to review.
            repo: Repository in format "owner/repo".
            base_commit: Base commit hash.
            instance_id: Optional instance ID for caching.
            cache: Optional cache for checkpointing.
            step_callback: Optional callback for live step streaming.
        """
        result = await self._run_review(
            problem_statement=problem_statement,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            instance_id=instance_id,
            cache=cache,
            step_callback=step_callback,
        )

        if result.confidence is None:
            logger.warning(
                "Review elicitation for instance %s did not return a valid confidence.",
                instance_id,
            )

        # Include checkpoint info in metadata
        metadata = {
            "method": f"review_{self.method}",
            "n_steps": result.n_steps,
            "review_cost": result.cost,
            "exit_status": result.exit_status,
            "review_history": result.history,
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

    async def _run_review(
        self,
        problem_statement: str,
        patch: str,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run the review agent with patch applied, with retries."""
        return await self._run_review_with_retry(
            problem_statement=problem_statement,
            patch=patch,
            repo=repo,
            base_commit=base_commit,
            instance_id=instance_id,
            cache=cache,
            step_callback=step_callback,
        )

    @retry_async(logger)
    async def _run_review_with_retry(
        self,
        problem_statement: str,
        patch: str,
        repo: str,
        base_commit: str,
        instance_id: str | None = None,
        cache: ResultCache | None = None,
        step_callback: StepCallback | None = None,
    ) -> AgentResult:
        """Run review with automatic retries on transient failures."""
        env = None

        # Build instance dict for Docker image lookup
        clean_id = instance_id.replace("instance_", "") if instance_id else f"{repo.replace('/', '__')}__{base_commit[:8]}"
        full_id = instance_id or f"instance_{repo.replace('/', '__')}__{base_commit[:8]}"

        # Check for existing checkpoint to resume from
        initial_messages = None
        if cache and instance_id:
            checkpoint = cache.load_checkpoint(
                method=f"review_{self.method}",
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
        def checkpoint_callback(messages, history, n_steps, cost):
            if cache and instance_id:
                cache.save_checkpoint(
                    method=f"review_{self.method}",
                    instance_id=instance_id,
                    messages=messages,
                    exploration_history=history,
                    n_steps=n_steps,
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
                "patch": patch,
                "test_patch": "",
                "version": "",
                "FAIL_TO_PASS": "[]",
                "PASS_TO_PASS": "[]",
            }

            logger.info(f"Starting environment for review: {clean_id}")
            env = await asyncio.to_thread(self._get_environment_for_instance, instance)
            register_environment(env)  # Track for graceful shutdown
            logger.info(f"Environment ready for {clean_id}")

            # Find the repository location and set it as the working directory
            repo_dir = await asyncio.to_thread(self._find_repo_dir_sync, env)
            if repo_dir:
                self._set_env_working_dir(env, repo_dir)

            # Apply the patch inside the container
            patch_error = await self._apply_patch_in_container(env, patch, repo_dir=repo_dir)

            # Run the agent (base class handles model/agent creation)
            result = await self._run_agent(
                env,
                problem_statement,
                patch=patch,
                checkpoint_callback=checkpoint_callback if cache and instance_id else None,
                step_callback=step_callback,
                initial_messages=initial_messages,
            )

            # Add patch application warning if needed
            if patch_error:
                result = AgentResult(
                    confidence=result.confidence,
                    n_steps=result.n_steps,
                    cost=result.cost,
                    history=result.history,
                    exit_status=result.exit_status,
                    final_output=f"[Patch apply warning: {patch_error}]\n\n{result.final_output}",
                    messages=result.messages,
                    checkpoint_saved=result.checkpoint_saved,
                    resumed_from_checkpoint=result.resumed_from_checkpoint,
                )

            # If successful, delete checkpoint
            if cache and instance_id and result.confidence is not None and result.exit_status not in ("Timeout", "Error"):
                cache.delete_checkpoint(
                    method=f"review_{self.method}",
                    instance_id=instance_id,
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
            logger.error(f"Error in review for {instance_id}: {e}")
            return self._create_error_result(exit_status=type(e).__name__, message=str(e))
        finally:
            if env is not None:
                unregister_environment(env)  # Remove from tracking before cleanup
                if hasattr(env, "stop"):
                    try:
                        env.stop()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up environment: {cleanup_error}")

    async def _apply_patch_in_container(
        self,
        env,
        patch: str,
        repo_dir: str | None = None,
    ) -> str | None:
        """Apply patch inside the container. Returns error message or None.

        Args:
            env: The container environment.
            patch: The git diff patch to apply.
            repo_dir: Path to the git repository in the container.
        """
        if not patch.strip():
            return None

        # Default to common repo locations
        repo_dir = repo_dir or "/app"
        # Use a timeout for each command to avoid hanging
        cmd_timeout = 60

        try:
            import base64

            # Helper to run execute with Python-level timeout
            async def execute_with_timeout(cmd, timeout):
                return await asyncio.wait_for(
                    asyncio.to_thread(env.execute, cmd, timeout=timeout),
                    timeout=timeout + 10,  # Extra buffer for asyncio overhead
                )

            # Clean any uncommitted changes in the repo
            # Note: The container is pre-configured at the correct pre-fix state
            # The base_commit in instance_id is the SOLUTION commit, not the starting state
            logger.info(f"Running git checkout in {repo_dir}...")
            clean_cmd = f"cd {repo_dir} && git checkout -- . 2>&1"
            await execute_with_timeout(clean_cmd, cmd_timeout)
            logger.info("Git checkout completed")

            # Encode patch as base64 to safely transfer to container
            patch_b64 = base64.b64encode(patch.encode()).decode()
            patch_file = "/tmp/review_patch.diff"
            b64_file = "/tmp/review_patch.b64"

            # Write base64 in chunks to avoid ARG_MAX (max ~60k per command)
            chunk_size = 50000
            chunks = [patch_b64[i:i + chunk_size] for i in range(0, len(patch_b64), chunk_size)]

            # First chunk overwrites, rest append
            for i, chunk in enumerate(chunks):
                op = ">" if i == 0 else ">>"
                write_cmd = f"printf '%s' '{chunk}' {op} {b64_file}"
                result = await execute_with_timeout(write_cmd, cmd_timeout)
                if result.get("returncode", 1) != 0:
                    return f"Failed to write patch chunk {i}: {result.get('output', '')}"

            # Decode base64 to patch file
            decode_cmd = f"base64 -d < {b64_file} > {patch_file}"
            result = await execute_with_timeout(decode_cmd, cmd_timeout)
            if result.get("returncode", 1) != 0:
                return f"Failed to decode patch: {result.get('output', '')}"

            # Apply the patch from file (cd to repo_dir first)
            apply_cmd = f"cd {repo_dir} && git apply {patch_file}"
            result = await execute_with_timeout(apply_cmd, cmd_timeout)
            if result.get("returncode", 1) != 0:
                # Try 3-way merge for conflicts
                apply_cmd_3way = f"cd {repo_dir} && git apply --3way {patch_file}"
                result = await execute_with_timeout(apply_cmd_3way, cmd_timeout)
                if result.get("returncode", 1) != 0:
                    return result.get("output", "Patch apply failed")
        except Exception as e:
            return f"Patch application error: {e}"
        return None


def ReviewDirectElicitation(**kwargs) -> ReviewElicitation:
    """Create review elicitation with direct method."""
    return ReviewElicitation(method="direct", **kwargs)


def ReviewAdversarialElicitation(**kwargs) -> ReviewElicitation:
    """Create review elicitation with adversarial method."""
    return ReviewElicitation(method="adversarial", **kwargs)
