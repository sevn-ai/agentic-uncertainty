"""Modal environment for mini-swe-agent.

This environment runs commands in a Modal sandbox instead of local Docker.
Requires `modal` to be installed and configured (modal token set).
"""

import logging
import os
from typing import Any

from pydantic import BaseModel

try:
    import modal
except ImportError:
    modal = None


class ModalEnvironmentConfig(BaseModel):
    """Configuration for Modal environment."""

    image: str
    """Docker image to use (from Docker Hub or Modal registry)."""
    cwd: str = "/testbed"
    """Working directory in which to execute commands."""
    env: dict[str, str] = {}
    """Environment variables to set in the sandbox."""
    timeout: int = 30
    """Timeout for executing commands."""
    sandbox_timeout: int = 7200
    """Max duration to keep sandbox running (seconds)."""
    app_name: str = "minisweagent"
    """Name of the Modal app."""
    cpu: float = 1.0
    """Number of CPUs to allocate."""
    memory: int = 4096
    """Memory in MB to allocate."""


class ModalEnvironment:
    """Execute bash commands in a Modal sandbox.

    This provides the same interface as DockerEnvironment but uses
    Modal's serverless infrastructure instead of local Docker.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        **kwargs,
    ):
        """Initialize Modal environment.

        See `ModalEnvironmentConfig` for keyword arguments.
        """
        if modal is None:
            raise RuntimeError(
                "Modal is not installed. Run: uv add modal\n"
                "Then configure: modal token set"
            )

        self.logger = logger or logging.getLogger("minisweagent.environment.modal")
        self.config = ModalEnvironmentConfig(**kwargs)
        self._sandbox: modal.Sandbox | None = None
        self._app: modal.App | None = None
        self._start_sandbox()

    def get_template_vars(self) -> dict[str, Any]:
        """Return config as template variables."""
        return self.config.model_dump()

    def _get_modal_image(self) -> "modal.Image":
        """Build Modal image from config."""
        image_spec = self.config.image

        # Check if Docker credentials are available
        if os.environ.get("DOCKER_USERNAME") and os.environ.get("DOCKER_PASSWORD"):
            secret = modal.Secret.from_dict({
                "DOCKER_USERNAME": os.environ["DOCKER_USERNAME"],
                "DOCKER_PASSWORD": os.environ["DOCKER_PASSWORD"],
            })
            secrets = [secret]
            self.logger.debug("Using Docker credentials for image pull")
        else:
            secrets = None

        # Reset entrypoint - SWE-bench Pro images have /bin/bash as entrypoint
        # which interferes with how we start the sandbox
        return modal.Image.from_registry(image_spec, secrets=secrets).entrypoint([])

    def _start_sandbox(self):
        """Start the Modal sandbox."""
        self.logger.info(f"Starting Modal sandbox with image {self.config.image}")

        # Get or create Modal app
        self._app = modal.App.lookup(self.config.app_name, create_if_missing=True)

        # Build image (with entrypoint reset for SWE-bench Pro compatibility)
        image = self._get_modal_image()

        # Create sandbox with tail to keep it alive
        self._sandbox = modal.Sandbox.create(
            "tail", "-f", "/dev/null",
            image=image,
            timeout=self.config.sandbox_timeout,
            app=self._app,
            cpu=self.config.cpu,
            memory=self.config.memory,
            workdir=self.config.cwd,
        )

        self.logger.info(f"Modal sandbox started: {self._sandbox.object_id}")
        self._consecutive_errors = 0

    def execute(self, command: str, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the Modal sandbox.

        Args:
            command: Bash command to execute.
            cwd: Working directory (optional, uses config.cwd by default).
            timeout: Command timeout in seconds.

        Returns:
            Dict with 'output' and 'returncode' keys.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started")

        cwd = cwd or self.config.cwd
        timeout = timeout or self.config.timeout

        # Build command with environment variables
        env_exports = []
        for key, value in self.config.env.items():
            # Escape value for shell
            escaped_value = value.replace("'", "'\"'\"'")
            env_exports.append(f"export {key}='{escaped_value}'")

        # Combine env exports with command
        if env_exports:
            full_command = " && ".join(env_exports) + " && " + command
        else:
            full_command = command

        # Execute in sandbox with timeout
        try:
            process = self._sandbox.exec(
                "bash",
                "-lc",
                f"cd {cwd} && {full_command}",
                timeout=timeout,
            )

            # Wait for completion (timeout is enforced by exec)
            process.wait()

            # Read stdout and stderr
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            returncode = process.returncode

            # Combine stdout and stderr like Docker environment does
            output = stdout
            if stderr:
                output += "\n" + stderr if output else stderr

            # Reset error counter on success
            self._consecutive_errors = 0
            return {"output": output, "returncode": returncode}

        except Exception as e:
            self._consecutive_errors += 1
            self.logger.error(f"Error executing command: {e}")

            # If we get too many consecutive errors, the sandbox is likely broken
            if self._consecutive_errors >= 5:
                raise RuntimeError(
                    f"Modal sandbox appears broken after {self._consecutive_errors} consecutive errors. "
                    f"Last error: {e}"
                )

            return {"output": str(e), "returncode": 1}

    def cleanup(self):
        """Terminate the Modal sandbox."""
        if getattr(self, "_sandbox", None) is not None:
            try:
                self._sandbox.terminate()
                self.logger.info("Modal sandbox terminated")
            except Exception as e:
                self.logger.warning(f"Error terminating sandbox: {e}")
            self._sandbox = None

    def __del__(self):
        """Cleanup sandbox when object is destroyed."""
        self.cleanup()
