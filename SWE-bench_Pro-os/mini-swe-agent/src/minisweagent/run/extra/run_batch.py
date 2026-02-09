#!/usr/bin/env python3

"""Run mini-SWE-agent on instances in batch mode.

This script provides a robust batch processing system for running mini-swe-agent
on multiple instances, with features like:
- Multiple instance sources (file, HuggingFace, SWE-bench)
- Parallel processing with configurable workers
- Trajectory validation and resume capability
- Progress tracking and reporting
"""

import concurrent.futures
import contextlib
import getpass
import json
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock

import typer
import yaml
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.batch_instances import (
    BatchInstance,
    load_instances_from_file,
    load_instances_from_huggingface,
    load_swebench_instances,
)
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger, remove_file_handler

_HELP_TEXT = """Run mini-SWE-agent on instances in batch mode.

[cyan][bold]=== BASIC USAGE ===[/bold][/cyan]

Run on instances from a file:
  [green]mini-extra run-batch --instances-path /path/to/instances.json -o output_dir[/green]

Run on SWE-bench lite:
  [green]mini-extra run-batch --source swebench --subset lite --split dev -o output_dir[/green]

Run with multiple workers:
  [green]mini-extra run-batch --instances-path /path/to/instances.json -o output_dir -w 4[/green]

[cyan][bold]=== INSTANCE SOURCES ===[/bold][/cyan]

- [bold]file[/bold]: Load from JSON/JSONL file (--instances-path)
- [bold]swebench[/bold]: Load from SWE-bench dataset (--subset, --split)
- [bold]huggingface[/bold]: Load from HuggingFace dataset (--dataset-name, --split)
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

_OUTPUT_FILE_LOCK = Lock()
_LOG_HANDLERS: dict[str, list[str]] = {}
_LOG_LOCK = Lock()


@dataclass
class RunBatchConfig:
    """Configuration for a batch run."""

    instances_path: Path | None = None
    source: str = "file"
    subset: str = "lite"
    split: str = "dev"
    dataset_name: str = ""
    filter_spec: str = ".*"
    slice_spec: str = ""
    shuffle: bool = False
    output_dir: Path = field(default_factory=lambda: Path("output"))
    workers: int = 1
    model: str | None = None
    model_class: str | None = None
    model_api_base: str | None = None
    model_api_key: str | None = None
    model_temperature: float | None = None
    model_top_p: float | None = None
    per_instance_call_limit: int = 0
    per_instance_cost_limit: float = 0.0
    total_cost_limit: float = 0.0
    redo_existing: bool = False
    raise_exceptions: bool = False
    config_path: Path = field(
        default_factory=lambda: builtin_config_dir / "extra" / "swebench.yaml"
    )
    environment_class: str | None = None
    deployment_type: str | None = None
    deployment_install_pipx: bool = False
    deployment_startup_timeout: int = 600
    random_delay_multiplier: float = 0.3
    no_live: bool = False


class ProgressTrackingAgent(DefaultAgent):
    """Wrapper around DefaultAgent that provides progress updates."""

    def __init__(
        self,
        *args,
        progress_manager: RunBatchProgressManager,
        instance_id: str = "",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.progress_manager: RunBatchProgressManager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        """Override step to provide progress updates."""
        self.progress_manager.update_instance_status(
            self.instance_id,
            f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})",
        )
        return super().step()


def get_environment_for_instance(config: dict, instance: BatchInstance) -> Environment:
    """Create an environment for a specific instance."""
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get(
        "environment_class", "modal")

    if instance.image_name:
        env_class = env_config["environment_class"]
        if env_class == "docker":
            env_config["image"] = instance.image_name
        elif env_class == "singularity":
            env_config["image"] = "docker://" + instance.image_name
        elif env_class in ["modal", "swerex_modal"]:
            env_config["image"] = instance.image_name
        elif env_class == "swerex_docker":
            env_config["image"] = instance.image_name

    env = get_environment(env_config)

    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(
            instance_id=instance.instance_id,
            problem_statement=instance.problem_statement,
            **instance.extra_fields,
        )
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")

    return env


def update_preds_file(
    output_path: Path, instance_id: str, model_name: str, result: str
):
    """Update the predictions JSON file with results from a single instance."""
    with _OUTPUT_FILE_LOCK:
        output_data = {}
        if output_path.exists():
            output_data = json.loads(output_path.read_text())
        output_data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": result,
        }
        output_path.write_text(json.dumps(output_data, indent=2))


def remove_from_preds_file(output_path: Path, instance_id: str):
    """Remove an instance from the predictions file."""
    if not output_path.exists():
        return
    with _OUTPUT_FILE_LOCK:
        output_data = json.loads(output_path.read_text())
        if instance_id in output_data:
            del output_data[instance_id]
            output_path.write_text(json.dumps(output_data, indent=2))


def should_skip_instance(
    output_dir: Path, instance_id: str, redo_existing: bool
) -> bool | str:
    """Check if we should skip this instance.

    Returns False if instance should be processed.
    Returns exit_status string if instance should be skipped.
    """
    if redo_existing:
        return False

    traj_path = output_dir / instance_id / f"{instance_id}.traj.json"
    if not traj_path.exists():
        return False

    content = traj_path.read_text()
    if not content.strip():
        logger.warning(f"Found empty trajectory: {traj_path}. Removing.")
        traj_path.unlink()
        return False

    try:
        data = json.loads(content)
        exit_status = data.get("info", {}).get("exit_status", None)
        if exit_status == "early_exit" or exit_status is None:
            logger.warning(
                f"Found trajectory with no/invalid exit status: {traj_path}. Removing."
            )
            traj_path.unlink()
            return False
        # Valid trajectory exists
        logger.info(f"⏭️  Skipping existing trajectory: {traj_path}")
        return exit_status
    except Exception as e:
        logger.error(
            f"Failed to check existing trajectory: {traj_path}: {e}. Removing."
        )
        traj_path.unlink()
        return False


def add_instance_log_handlers(
    output_dir: Path, instance_id: str, multi_worker: bool = False
):
    """Add per-instance log file handlers."""
    with _LOG_LOCK:
        instance_dir = output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        handler_ids = []
        for level in ["info", "debug"]:
            handler_id = f"{instance_id}-{level}"
            filter_name = instance_id if multi_worker else ""
            add_file_handler(
                instance_dir / f"{instance_id}.{level}.log",
                filter=filter_name,
                level=level,
                id_=handler_id,
            )
            handler_ids.append(handler_id)

        _LOG_HANDLERS[instance_id] = handler_ids


def remove_instance_log_handlers(instance_id: str):
    """Remove per-instance log file handlers."""
    with _LOG_LOCK:
        if instance_id in _LOG_HANDLERS:
            for handler_id in _LOG_HANDLERS[instance_id]:
                remove_file_handler(handler_id)
            del _LOG_HANDLERS[instance_id]


def save_config_files(output_dir: Path, run_config: RunBatchConfig, agent_config: dict):
    """Save configuration files for reproducibility."""
    # Save run configuration
    (output_dir / "run_batch.config.yaml").write_text(
        yaml.dump(asdict(run_config), indent=2)
    )

    # Save agent configuration
    (output_dir / "agent.config.yaml").write_text(yaml.dump(agent_config, indent=2))


def save_instance_config(output_dir: Path, instance: BatchInstance, agent_config: dict):
    """Save per-instance configuration for replay."""
    instance_dir = output_dir / instance.instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    instance_config = {
        "instance": asdict(instance),
        "agent": agent_config,
    }

    (instance_dir / f"{instance.instance_id}.config.yaml").write_text(
        yaml.dump(instance_config, indent=2)
    )


def save_pred_file(
    output_path: Path, model_name_or_path: str, instance_id: str, model_patch: str
):
    """Save the predictions file for a single instance."""
    output_data = {
        "model_name_or_path": model_name_or_path,
        "instance_id": instance_id,
        "model_patch": model_patch,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


# Errors that should trigger a retry
RETRYABLE_ERRORS = (
    "RemoteError",
    "InternalServerError",
    "ServerError",
    "ServiceUnavailable",
    "ConnectionError",
    "TimeoutError",
    "APIError",
    "RateLimitError",
)


def process_instance(
    instance: BatchInstance,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_config: RunBatchConfig,
    max_retries: int = 3,
) -> None:
    """Process a single instance with retry logic for transient errors."""
    instance_id = instance.instance_id
    instance_dir = output_dir / instance_id

    # Add per-instance logging
    add_instance_log_handlers(
        output_dir, instance_id, multi_worker=run_config.workers > 1
    )

    # Random delay to avoid thundering herd
    if progress_manager.n_completed < run_config.workers:
        time.sleep(
            random.random()
            * run_config.random_delay_multiplier
            * (run_config.workers - 1)
        )

    progress_manager.on_instance_start(instance_id)

    # Check if we should skip this instance
    if skip_status := should_skip_instance(
        output_dir, instance_id, run_config.redo_existing
    ):
        progress_manager.on_instance_end(
            instance_id, f"skipped ({skip_status})")
        remove_instance_log_handlers(instance_id)
        return

    # Clean up any inconsistent state
    remove_from_preds_file(output_dir / "preds.json", instance_id)
    (instance_dir / f"{instance_id}.traj.json").unlink(missing_ok=True)

    # Save instance config for replay
    save_instance_config(output_dir, instance, config)

    task = instance.problem_statement
    agent = None
    extra_info = None
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            # Recreate model for each attempt to avoid stale state
            model = get_model(config=config.get("model", {}))

            if attempt > 0:
                # Exponential backoff: 2^attempt seconds (2, 4, 8, ...), max 60s
                backoff = min(2 ** attempt, 60)
                logger.info(f"Retry {attempt}/{max_retries} for {instance_id} after {backoff}s backoff")
                progress_manager.update_instance_status(
                    instance_id, f"Retry {attempt}/{max_retries} (waiting {backoff}s)"
                )
                time.sleep(backoff)

            progress_manager.update_instance_status(
                instance_id, "Starting environment")

            env = get_environment_for_instance(config, instance)
            agent = ProgressTrackingAgent(
                model,
                env,
                progress_manager=progress_manager,
                instance_id=instance_id,
                **config.get("agent", {}),
            )
            exit_status, result = agent.run(task, problem_statement=task)
            # When agent submits successfully, result contains the git diff (patch)
            patch = result if exit_status == "Submitted" else ""
            logger.info(
                f"Exit status: {exit_status}, Result: {result}, Patch: {patch}")
            break  # Success, exit retry loop

        except KeyboardInterrupt:
            logger.info(f"Keyboard interrupt for instance {instance_id}")
            exit_status, result, patch = "KeyboardInterrupt", "User interrupted", ""
            extra_info = {"traceback": traceback.format_exc()}
            if run_config.raise_exceptions:
                raise
            break  # Don't retry on keyboard interrupt

        except Exception as e:
            error_type = type(e).__name__
            last_error = e
            logger.error(
                f"Error processing instance {instance_id} (attempt {attempt + 1}/{max_retries + 1}): {e}",
                exc_info=True
            )

            # Check if this is a retryable error and we have retries left
            if error_type in RETRYABLE_ERRORS and attempt < max_retries:
                logger.info(f"Retryable error {error_type}, will retry...")
                # Clean up environment before retry
                if agent and hasattr(agent, "environment"):
                    try:
                        agent.environment.stop()
                    except Exception:
                        pass
                continue  # Retry

            # Non-retryable error or out of retries
            exit_status, result, patch = error_type, str(e), ""
            extra_info = {
                "traceback": traceback.format_exc(),
                "retry_attempts": attempt + 1,
            }
            if run_config.raise_exceptions:
                raise
            break

    # Handle case where all retries failed
    if last_error and 'exit_status' not in dir():
        exit_status, result, patch = type(last_error).__name__, str(last_error), ""
        extra_info = {
            "traceback": traceback.format_exc(),
            "retry_attempts": max_retries + 1,
        }

    # Save results (always runs, like a finally block)
    save_traj(
        agent,
        instance_dir / f"{instance_id}.traj.json",
        exit_status=exit_status,
        result=patch,
        extra_info=extra_info,
        instance_id=instance_id,
        print_fct=logger.info,
    )
    save_pred_file(
        output_dir / f"{instance_id}/{instance_id}.pred",
        output_dir.name,
        instance_id,
        patch,
    )
    if agent and hasattr(agent, "model"):
        update_preds_file(
            output_dir / "preds.json",
            instance_id,
            agent.model.config.model_name,
            patch,
        )
    progress_manager.on_instance_end(instance_id, exit_status)
    remove_instance_log_handlers(instance_id)


def load_instances(run_config: RunBatchConfig) -> list[BatchInstance]:
    """Load instances based on the run configuration."""
    if run_config.source == "file":
        if not run_config.instances_path:
            raise ValueError("--instances-path is required when --source=file")
        return load_instances_from_file(
            run_config.instances_path,
            filter_spec=run_config.filter_spec,
            slice_spec=run_config.slice_spec,
            shuffle=run_config.shuffle,
        )
    elif run_config.source == "swebench":
        return load_swebench_instances(
            subset=run_config.subset,
            split=run_config.split,
            filter_spec=run_config.filter_spec,
            slice_spec=run_config.slice_spec,
            shuffle=run_config.shuffle,
        )
    elif run_config.source == "huggingface":
        if not run_config.dataset_name:
            raise ValueError(
                "--dataset-name is required when --source=huggingface")
        return load_instances_from_huggingface(
            dataset_name=run_config.dataset_name,
            split=run_config.split,
            filter_spec=run_config.filter_spec,
            slice_spec=run_config.slice_spec,
            shuffle=run_config.shuffle,
        )
    else:
        raise ValueError(f"Unknown source: {run_config.source}")


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    instances_path: Path = typer.Option(None, "--instances-path", help="Path to instances file (JSON/JSONL)", rich_help_panel="Instance Loading"),
    source: str = typer.Option("file", "--source", help="Instance source: file, swebench, or huggingface", rich_help_panel="Instance Loading"),
    subset: str = typer.Option("lite", "--subset", help="SWE-bench subset (lite, verified, full, etc.)", rich_help_panel="Instance Loading"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Instance Loading"),
    dataset_name: str = typer.Option("", "--dataset-name", help="HuggingFace dataset name", rich_help_panel="Instance Loading"),
    filter_spec: str = typer.Option(".*", "--filter", help="Filter instance IDs by regex", rich_help_panel="Instance Filtering"),
    slice_spec: str = typer.Option("", "--slice", help="Slice instances (e.g., '0:5' for first 5)", rich_help_panel="Instance Filtering"),
    shuffle: bool = typer.Option(False, "--shuffle/--no-shuffle", help="Shuffle instances (fixed seed)", rich_help_panel="Instance Filtering"),
    output: str = typer.Option("", "-o", "--output", "--output-dir", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", "--num-workers", help="Number of parallel workers", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model name", rich_help_panel="Model"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class", rich_help_panel="Model"),
    model_api_base: str | None = typer.Option(None, "--model-api-base", help="Model API base URL", rich_help_panel="Model"),
    model_api_key: str | None = typer.Option(None, "--model-api-key", help="Model API key", rich_help_panel="Model"),
    model_temperature: float | None = typer.Option(None, "--model-temperature", help="Model temperature", rich_help_panel="Model"),
    model_top_p: float | None = typer.Option(None, "--model-top-p", help="Model top-p", rich_help_panel="Model"),
    per_instance_call_limit: int = typer.Option(0, "--per-instance-call-limit", help="Max API calls per instance (0=unlimited)", rich_help_panel="Model Limits"),
    per_instance_cost_limit: float = typer.Option(0.0, "--per-instance-cost-limit", help="Max cost per instance (0=unlimited)", rich_help_panel="Model Limits"),
    total_cost_limit: float = typer.Option(0.0, "--total-cost-limit", help="Max total cost (0=unlimited)", rich_help_panel="Model Limits"),
    redo_existing: bool = typer.Option(False, "--redo-existing/--no-redo-existing", help="Redo existing instances", rich_help_panel="Advanced"),
    raise_exceptions: bool = typer.Option(False, "--raise-exceptions/--no-raise-exceptions", help="Raise exceptions instead of continuing", rich_help_panel="Advanced"),
    config_spec: Path = typer.Option(builtin_config_dir / "extra" / "swebench.yaml", "--config", help="Path to agent config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type (docker, singularity, local)", rich_help_panel="Environment"),
    deployment_type: str | None = typer.Option(None, "--deployment-type", help="Deployment type (modal, docker, etc.)", rich_help_panel="Environment"),
    deployment_install_pipx: bool = typer.Option(False, "--deployment-install-pipx/--no-deployment-install-pipx", help="Install pipx in deployment", rich_help_panel="Environment"),
    deployment_startup_timeout: int = typer.Option(600, "--deployment-startup-timeout", help="Deployment startup timeout (seconds)", rich_help_panel="Environment"),
    random_delay_multiplier: float = typer.Option(0.3, "--random-delay-multiplier", help="Random startup delay multiplier", rich_help_panel="Advanced"),
    no_live: bool = typer.Option(False, "--no-live/--live", help="Disable live progress display (avoids threading deadlocks)", rich_help_panel="Advanced"),
) -> None:
    # fmt: on

    # Create run configuration
    run_config = RunBatchConfig(
        instances_path=instances_path,
        source=source,
        subset=subset,
        split=split,
        dataset_name=dataset_name,
        filter_spec=filter_spec,
        slice_spec=slice_spec,
        shuffle=shuffle,
        output_dir=Path(output) if output else Path("output") /
        getpass.getuser() / f"run_{int(time.time())}",
        workers=workers,
        model=model,
        model_class=model_class,
        model_api_base=model_api_base,
        model_api_key=model_api_key,
        model_temperature=model_temperature,
        model_top_p=model_top_p,
        per_instance_call_limit=per_instance_call_limit,
        per_instance_cost_limit=per_instance_cost_limit,
        total_cost_limit=total_cost_limit,
        redo_existing=redo_existing,
        raise_exceptions=raise_exceptions,
        config_path=config_spec,
        environment_class=environment_class,
        deployment_type=deployment_type,
        deployment_install_pipx=deployment_install_pipx,
        deployment_startup_timeout=deployment_startup_timeout,
        random_delay_multiplier=random_delay_multiplier,
        no_live=no_live,
    )

    # Setup output directory
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {run_config.output_dir}")

    # Add main log file handler
    add_file_handler(run_config.output_dir / "run_batch.log", id_="main")

    # Load agent configuration
    config_path = get_config_path(run_config.config_path)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())

    # Override config with CLI options - Model settings
    if run_config.model is not None:
        config.setdefault("model", {})["model_name"] = run_config.model
    if run_config.model_class is not None:
        config.setdefault("model", {})["model_class"] = run_config.model_class

    # For LiteLLM model, parameters go into model_kwargs
    model_kwargs = config.setdefault(
        "model", {}).setdefault("model_kwargs", {})
    if run_config.model_api_base is not None:
        model_kwargs["api_base"] = run_config.model_api_base
    if run_config.model_api_key is not None:
        model_kwargs["api_key"] = run_config.model_api_key
    if run_config.model_temperature is not None:
        model_kwargs["temperature"] = run_config.model_temperature
    if run_config.model_top_p is not None:
        model_kwargs["top_p"] = run_config.model_top_p

    # Model limits
    if run_config.per_instance_call_limit > 0:
        config.setdefault("agent", {})[
            "step_limit"] = run_config.per_instance_call_limit
    if run_config.per_instance_cost_limit > 0:
        config.setdefault("agent", {})[
            "cost_limit"] = run_config.per_instance_cost_limit
    if run_config.total_cost_limit > 0:
        config.setdefault("model", {})[
            "total_cost_limit"] = run_config.total_cost_limit

    # Environment settings
    # Handle deployment-specific parameters (mainly for Modal)
    # This must come BEFORE environment_class to allow --deployment-type to override
    if run_config.deployment_type is not None:
        # If deployment_type is "modal", set environment_class to modal (override any existing value)
        if run_config.deployment_type.lower() == "modal":
            config.setdefault("environment", {})
            # Direct assignment to override
            config["environment"]["environment_class"] = "modal"

    # Only set environment_class from CLI if explicitly provided AND deployment_type is not set
    if run_config.environment_class is not None and run_config.deployment_type is None:
        config.setdefault("environment", {})
        config["environment"]["environment_class"] = run_config.environment_class

    # Pass Modal-specific parameters if using Modal environment
    env_class = config.get("environment", {}).get("environment_class", "")
    if "modal" in env_class.lower():
        if run_config.deployment_install_pipx:
            config.setdefault("environment", {})[
                "install_pipx"] = run_config.deployment_install_pipx
        if run_config.deployment_startup_timeout != 600:
            config.setdefault("environment", {})[
                "startup_timeout"] = run_config.deployment_startup_timeout

    # Save configuration files
    save_config_files(run_config.output_dir, run_config, config)

    # Load instances
    logger.info(f"Loading instances from {run_config.source}...")
    instances = load_instances(run_config)
    logger.info(f"Loaded {len(instances)} instances")

    if not instances:
        logger.error("No instances to process!")
        return

    # Create progress manager
    progress_manager = RunBatchProgressManager(
        len(instances),
        run_config.output_dir / "exit_statuses.yaml"
    )

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        """Process completed futures."""
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                instance_id = futures[future]
                logger.error(
                    f"Error in future for instance {instance_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(instance_id, e)

    # Run instances
    # Use nullcontext if --no-live is set to avoid deadlocks with threading
    live_context = contextlib.nullcontext() if run_config.no_live else Live(progress_manager.render_group, refresh_per_second=4)
    with live_context:
        with concurrent.futures.ThreadPoolExecutor(max_workers=run_config.workers) as executor:
            futures = {
                executor.submit(
                    process_instance,
                    instance,
                    run_config.output_dir,
                    config,
                    progress_manager,
                    run_config,
                ): instance.instance_id
                for instance in instances
            }
            try:
                process_futures(futures)
            except KeyboardInterrupt:
                logger.info(
                    "Cancelling pending jobs. Press ^C again to exit immediately.")
                for future in futures:
                    if not future.running() and not future.done():
                        future.cancel()
                process_futures(futures)

    logger.info("Batch run complete!")
    logger.info(f"Results saved to {run_config.output_dir}")


if __name__ == "__main__":
    app()
