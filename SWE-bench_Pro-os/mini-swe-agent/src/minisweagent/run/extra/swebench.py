#!/usr/bin/env python3

"""Run mini-SWE-agent on SWE-bench instances in batch mode."""
# Read this first: https://mini-swe-agent.com/latest/usage/swebench/  (usage docs)

import concurrent.futures
import json
import logging
import os
import random
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import typer
import yaml
from datasets import load_dataset
from jinja2 import StrictUndefined, Template
from rich.live import Live

from minisweagent import Environment
from minisweagent.agents import get_agent_class
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.extra.utils.batch_progress import RunBatchProgressManager
from minisweagent.run.utils.save import save_traj
from minisweagent.utils.log import add_file_handler, logger


# Thread-safe structured run log
_RUN_LOG_LOCK = threading.Lock()
_RUN_LOG: dict = {"instances": {}, "start_time": None, "end_time": None, "config": None}


def _log_instance_event(
    instance_id: str,
    event: str,
    *,
    run_idx: int = 0,
    details: dict | None = None,
    error: str | None = None,
    traceback_str: str | None = None,
) -> None:
    """Log a structured event for an instance to the run log."""
    key = f"{instance_id}_run_{run_idx}"
    with _RUN_LOG_LOCK:
        if key not in _RUN_LOG["instances"]:
            _RUN_LOG["instances"][key] = {
                "instance_id": instance_id,
                "run_idx": run_idx,
                "events": [],
                "status": "pending",
            }

        event_data = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
        }
        if details:
            event_data["details"] = details
        if error:
            event_data["error"] = error
        if traceback_str:
            event_data["traceback"] = traceback_str

        _RUN_LOG["instances"][key]["events"].append(event_data)

        # Update status based on event
        if event == "started":
            _RUN_LOG["instances"][key]["status"] = "running"
        elif event == "completed":
            _RUN_LOG["instances"][key]["status"] = "completed"
            _RUN_LOG["instances"][key]["exit_status"] = details.get("exit_status") if details else None
        elif event == "failed":
            _RUN_LOG["instances"][key]["status"] = "failed"
            _RUN_LOG["instances"][key]["error"] = error


def _save_run_log(output_dir: Path) -> None:
    """Save the structured run log to a JSON file."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with _RUN_LOG_LOCK:
        _RUN_LOG["end_time"] = datetime.now().isoformat()
        log_path = logs_dir / "run_log.json"
        log_path.write_text(json.dumps(_RUN_LOG, indent=2))


def _generate_summary_report(output_dir: Path) -> None:
    """Generate a human-readable summary report of the run."""
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with _RUN_LOG_LOCK:
        instances = _RUN_LOG["instances"]

    # Count by status
    completed = [k for k, v in instances.items() if v["status"] == "completed"]
    failed = [k for k, v in instances.items() if v["status"] == "failed"]

    # Group completed by exit_status
    exit_statuses: dict[str, list] = {}
    for key in completed:
        status = instances[key].get("exit_status", "Unknown")
        exit_statuses.setdefault(status, []).append(key)

    # Group failed by error type
    error_types: dict[str, list] = {}
    for key in failed:
        error = instances[key].get("error", "Unknown error")
        # Extract error type (first line or class name)
        error_type = error.split("\n")[0][:100] if error else "Unknown"
        error_types.setdefault(error_type, []).append(key)

    # Get model name for report title
    model_name = _RUN_LOG.get("config", {}).get("model", "unknown_model") or "unknown_model"

    # Write report to logs directory
    report_path = logs_dir / "run_summary.md"
    with open(report_path, "w") as f:
        f.write(f"# Mini-SWE-Agent Run Summary\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Start time:** {_RUN_LOG.get('start_time', 'N/A')}\n")
        f.write(f"**End time:** {_RUN_LOG.get('end_time', 'N/A')}\n\n")

        f.write("## Overview\n\n")
        f.write(f"- **Total instances:** {len(instances)}\n")
        f.write(f"- **Completed:** {len(completed)}\n")
        f.write(f"- **Failed:** {len(failed)}\n\n")

        if exit_statuses:
            f.write("## Completed Instances by Exit Status\n\n")
            for status, keys in sorted(exit_statuses.items(), key=lambda x: -len(x[1])):
                f.write(f"### {status} ({len(keys)})\n\n")
                for key in keys:
                    f.write(f"- `{key}`\n")
                f.write("\n")

        if error_types:
            f.write("## Failed Instances by Error Type\n\n")
            for error_type, keys in sorted(error_types.items(), key=lambda x: -len(x[1])):
                f.write(f"### {error_type} ({len(keys)})\n\n")
                for key in keys:
                    inst = instances[key]
                    f.write(f"- `{key}`\n")
                    # Include first few lines of traceback if available
                    events = inst.get("events", [])
                    for ev in reversed(events):
                        if ev.get("traceback"):
                            tb_lines = ev["traceback"].split("\n")[:10]
                            f.write("  ```\n")
                            for line in tb_lines:
                                f.write(f"  {line}\n")
                            if len(ev["traceback"].split("\n")) > 10:
                                f.write("  ...\n")
                            f.write("  ```\n")
                            break
                f.write("\n")

    logger.info(f"Summary report saved to {report_path}")

_HELP_TEXT = """Run mini-SWE-agent on SWEBench instances.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/swebench/[/bold green]
[/not dim]
"""

app = typer.Typer(rich_markup_mode="rich", add_completion=False)

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
    "smith": "SWE-bench/SWE-smith",
    "_test": "klieret/swe-bench-dummy-test-dataset",
}


_OUTPUT_FILE_LOCK = threading.Lock()


def make_progress_tracking_agent(base_class: type) -> type:
    """Create a progress-tracking subclass of any agent class."""

    class ProgressTrackingAgent(base_class):
        """Wrapper that adds progress tracking to any agent class."""

        def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
            super().__init__(*args, **kwargs)
            self.progress_manager: RunBatchProgressManager = progress_manager
            self.instance_id = instance_id

        def step(self) -> dict:
            """Override step to provide progress updates."""
            self.progress_manager.update_instance_status(
                self.instance_id, f"Step {self.model.n_calls + 1:3d} (${self.model.cost:.2f})"
            )
            return super().step()

    ProgressTrackingAgent.__name__ = f"ProgressTracking{base_class.__name__}"
    return ProgressTrackingAgent


# Default for backward compatibility
ProgressTrackingAgent = make_progress_tracking_agent(DefaultAgent)


def get_swebench_docker_image_name(instance: dict, dockerhub_username: str | None = None) -> str:
    """Get the image name for a SWEBench instance.

    Args:
        instance: SWEBench instance dict.
        dockerhub_username: Docker Hub username to use. If None, uses SWEBENCH_DOCKERHUB_USERNAME
                           env var, or defaults to 'swebench'.
    """
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        # Allow override via env var or parameter
        username = dockerhub_username or os.environ.get("SWEBENCH_DOCKERHUB_USERNAME", "swebench")
        image_name = f"docker.io/{username}/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
    return image_name


def get_sb_environment(config: dict, instance: dict) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ["docker", "swerex_modal"]:
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = "docker://" + image_name
    elif env_config["environment_class"] == "modal":
        env_config["image"] = image_name
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(
            **instance
        )
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def update_preds_file(
    output_path: Path, instance_id: str, model_name: str, result: str
):
    """Update the output JSON file with results from a single instance."""
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


def process_instance(
    instance: dict,
    output_dir: Path,
    config: dict,
    progress_manager: RunBatchProgressManager,
    run_idx: int = 0,
    max_retries: int = 3,
) -> None:
    """Process a single SWEBench instance with retry logic.

    Args:
        instance: SWEBench instance dict with problem_statement, instance_id, etc.
        output_dir: Directory to save trajectories.
        config: Agent/model configuration.
        progress_manager: Progress tracking manager.
        run_idx: Run index for multi-run experiments (0, 1, 2, ...).
        max_retries: Maximum number of retry attempts on failure (default 3).
    """
    instance_id = instance["instance_id"]
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    # Set up per-instance log file in logs/instances/ directory
    instance_logs_dir = output_dir / "logs" / "instances"
    instance_logs_dir.mkdir(parents=True, exist_ok=True)
    instance_log_path = instance_logs_dir / f"{instance_id}_run_{run_idx}.log"
    instance_log_handler = logging.FileHandler(instance_log_path, encoding="utf-8")
    instance_log_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    instance_log_handler.setLevel(logging.DEBUG)
    logger.addHandler(instance_log_handler)

    # Trajectory filename includes run index for multi-run support
    traj_filename = f"{instance_id}_run_{run_idx}.traj.json"
    traj_path = instance_dir / traj_filename

    # Only remove if we're about to overwrite
    traj_path.unlink(missing_ok=True)

    task = instance["problem_statement"]

    run_label = f"[run {run_idx}]" if run_idx > 0 else ""
    progress_manager.on_instance_start(f"{instance_id}{run_label}")

    # Log instance start
    _log_instance_event(instance_id, "started", run_idx=run_idx, details={
        "problem_statement_length": len(task),
        "repo": instance.get("repo", "unknown"),
    })
    logger.info(f"Starting instance {instance_id} run {run_idx}")

    # Retry loop for transient failures
    last_exception = None
    try:
        for attempt in range(max_retries + 1):
            model = get_model(config=config.get("model", {}))
            agent = None
            extra_info = None
            env = None

            if attempt > 0:
                # Exponential backoff: 2^attempt seconds (2, 4, 8, ...)
                backoff = min(2 ** attempt, 30)
                logger.info(f"Retry {attempt}/{max_retries} for {instance_id} after {backoff}s backoff")
                _log_instance_event(instance_id, "retry", run_idx=run_idx, details={
                    "attempt": attempt,
                    "max_retries": max_retries,
                    "backoff_seconds": backoff,
                })
                progress_manager.update_instance_status(
                    f"{instance_id}{run_label}", f"Retry {attempt}/{max_retries} (waiting {backoff}s)"
                )
                time.sleep(backoff)

            progress_manager.update_instance_status(f"{instance_id}{run_label}", "Pulling/starting docker")
            _log_instance_event(instance_id, "environment_setup", run_idx=run_idx, details={
                "attempt": attempt,
            })

            try:
                env = get_sb_environment(config, instance)
                logger.info(f"Environment started for {instance_id}")

                # Get agent class from config and wrap with progress tracking
                agent_config = config.get("agent", {}).copy()
                agent_class_name = agent_config.pop("agent_class", None)
                base_agent_class = get_agent_class(agent_class_name)
                agent_class = make_progress_tracking_agent(base_agent_class)
                agent = agent_class(
                    model,
                    env,
                    progress_manager=progress_manager,
                    instance_id=f"{instance_id}{run_label}",
                    **agent_config,
                )

                _log_instance_event(instance_id, "agent_run_start", run_idx=run_idx)
                exit_status, result = agent.run(task)
                # When agent submits successfully, result contains the git diff (patch)
                patch = result if exit_status == "Submitted" else ""

                # Success - save and return
                logger.info(f"Instance {instance_id} completed with exit_status={exit_status}")
                if env and hasattr(env, "stop"):
                    env.stop()
                save_traj(
                    agent,
                    traj_path,
                    exit_status=exit_status,
                    result=patch,
                    extra_info=extra_info,
                    instance_id=instance_id,
                    print_fct=logger.info,
                )
                if run_idx == 0:
                    update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, patch)
                progress_manager.on_instance_end(f"{instance_id}{run_label}", exit_status)

                _log_instance_event(instance_id, "completed", run_idx=run_idx, details={
                    "exit_status": exit_status,
                    "patch_length": len(patch),
                    "model_calls": getattr(model, "n_calls", 0),
                    "model_cost": getattr(model, "cost", 0),
                })
                return  # Success, exit retry loop

            except KeyboardInterrupt:
                # Don't retry on keyboard interrupt
                _log_instance_event(instance_id, "interrupted", run_idx=run_idx)
                raise
            except Exception as e:
                last_exception = e
                tb_str = traceback.format_exc()
                logger.error(
                    f"Error processing instance {instance_id} run {run_idx} (attempt {attempt + 1}/{max_retries + 1}): {e}",
                    exc_info=True
                )
                _log_instance_event(instance_id, "error", run_idx=run_idx,
                    error=str(e),
                    traceback_str=tb_str,
                    details={"attempt": attempt + 1, "max_retries": max_retries + 1}
                )

                # Clean up environment before retry
                if env and hasattr(env, "stop"):
                    try:
                        env.stop()
                    except Exception as cleanup_error:
                        logger.warning(f"Error cleaning up environment: {cleanup_error}")

                # If this was the last attempt, save the failure
                if attempt == max_retries:
                    exit_status, result, patch = type(e).__name__, str(e), ""
                    extra_info = {
                        "traceback": tb_str,
                        "retry_attempts": attempt + 1,
                    }
                    save_traj(
                        agent,
                        traj_path,
                        exit_status=exit_status,
                        result=patch,
                        extra_info=extra_info,
                        instance_id=instance_id,
                        print_fct=logger.info,
                    )
                    if run_idx == 0:
                        update_preds_file(output_dir / "preds.json", instance_id, model.config.model_name, patch)
                    progress_manager.on_instance_end(f"{instance_id}{run_label}", exit_status)

                    _log_instance_event(instance_id, "failed", run_idx=run_idx,
                        error=str(e),
                        traceback_str=tb_str,
                        details={"exit_status": exit_status, "total_attempts": attempt + 1}
                    )
                # Otherwise continue to next retry attempt
    finally:
        # Clean up instance-specific log handler
        logger.removeHandler(instance_log_handler)
        instance_log_handler.close()


def filter_instances(
    instances: list[dict],
    *,
    filter_spec: str,
    slice_spec: str = "",
    shuffle: bool = False,
) -> list[dict]:
    """Filter and slice a list of SWEBench instances."""
    if shuffle:
        instances = sorted(instances.copy(), key=lambda x: x["instance_id"])
        random.seed(42)
        random.shuffle(instances)
    before_filter = len(instances)
    instances = [
        instance
        for instance in instances
        if re.match(filter_spec, instance["instance_id"])
    ]
    if (after_filter := len(instances)) != before_filter:
        logger.info(f"Instance filter: {before_filter} -> {after_filter} instances")
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
        if (after_slice := len(instances)) != before_filter:
            logger.info(f"Instance slice: {before_filter} -> {after_slice} instances")
    return instances


# fmt: off
@app.command(help=_HELP_TEXT)
def main(
    subset: str = typer.Option("lite", "--subset", help="SWEBench subset to use or path to a dataset", rich_help_panel="Data selection"),
    split: str = typer.Option("dev", "--split", help="Dataset split", rich_help_panel="Data selection"),
    slice_spec: str = typer.Option("", "--slice", help="Slice specification (e.g., '0:5' for first 5 instances)", rich_help_panel="Data selection"),
    filter_spec: str = typer.Option("", "--filter", help="Filter instance IDs by regex", rich_help_panel="Data selection"),
    shuffle: bool = typer.Option(False, "--shuffle", help="Shuffle instances", rich_help_panel="Data selection"),
    output: str = typer.Option("", "-o", "--output", help="Output directory", rich_help_panel="Basic"),
    workers: int = typer.Option(1, "-w", "--workers", help="Number of worker threads for parallel processing", rich_help_panel="Basic"),
    num_runs: int = typer.Option(1, "--num-runs", help="Number of independent runs per instance (for retry experiments)", rich_help_panel="Basic"),
    model: str | None = typer.Option(None, "-m", "--model", help="Model to use", rich_help_panel="Basic"),
    model_class: str | None = typer.Option(None, "--model-class", help="Model class to use (e.g., 'anthropic' or 'minisweagent.models.anthropic.AnthropicModel')", rich_help_panel="Advanced"),
    agent_class: str | None = typer.Option(None, "--agent-class", help="Agent class to use (e.g., 'confidence' for online confidence elicitation)", rich_help_panel="Advanced"),
    redo_existing: bool = typer.Option(False, "--redo-existing", help="Redo existing instances", rich_help_panel="Data selection"),
    config_spec: Path = typer.Option( builtin_config_dir / "extra" / "swebench.yaml", "-c", "--config", help="Path to a config file", rich_help_panel="Basic"),
    environment_class: str | None = typer.Option( None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logs_path = output_path / "logs"
    logs_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")
    add_file_handler(logs_path / "minisweagent.log")

    # Initialize structured run log
    global _RUN_LOG
    _RUN_LOG = {
        "instances": {},
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "config": {
            "subset": subset,
            "split": split,
            "model": model,
            "model_class": model_class,
            "environment_class": environment_class,
            "workers": workers,
            "num_runs": num_runs,
        },
    }

    dataset_path = DATASET_MAPPING.get(subset, subset)
    logger.info(f"Loading dataset {dataset_path}, split {split}...")
    instances = list(load_dataset(dataset_path, split=split))

    instances = filter_instances(instances, filter_spec=filter_spec, slice_spec=slice_spec, shuffle=shuffle)
    if not redo_existing and (output_path / "preds.json").exists():
        existing_instances = list(json.loads((output_path / "preds.json").read_text()).keys())
        logger.info(f"Skipping {len(existing_instances)} existing instances")
        instances = [instance for instance in instances if instance["instance_id"] not in existing_instances]
    logger.info(f"Running on {len(instances)} instances...")

    config_path = get_config_path(config_spec)
    logger.info(f"Loading agent config from '{config_path}'")
    config = yaml.safe_load(config_path.read_text())
    if environment_class is not None:
        config.setdefault("environment", {})["environment_class"] = environment_class
    if model is not None:
        config.setdefault("model", {})["model_name"] = model
    if model_class is not None:
        config.setdefault("model", {})["model_class"] = model_class
    if agent_class is not None:
        config.setdefault("agent", {})["agent_class"] = agent_class

    # Total jobs = instances × num_runs
    total_jobs = len(instances) * num_runs
    if num_runs > 1:
        logger.info(f"Running {num_runs} runs per instance ({total_jobs} total jobs)")

    progress_manager = RunBatchProgressManager(total_jobs, logs_path / f"exit_statuses_{time.time()}.yaml")

    def process_futures(futures: dict[concurrent.futures.Future, str]):
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except concurrent.futures.CancelledError:
                pass
            except Exception as e:
                job_id = futures[future]
                logger.error(f"Error in future for {job_id}: {e}", exc_info=True)
                progress_manager.on_uncaught_exception(job_id, e)

    try:
        with Live(progress_manager.render_group, refresh_per_second=4):
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {}
                for instance in instances:
                    for run_idx in range(num_runs):
                        job_id = f"{instance['instance_id']}_run_{run_idx}"
                        future = executor.submit(
                            process_instance, instance, output_path, config, progress_manager, run_idx
                        )
                        futures[future] = job_id
                try:
                    process_futures(futures)
                except KeyboardInterrupt:
                    logger.info("Cancelling all pending jobs. Press ^C again to exit immediately.")
                    for future in futures:
                        if not future.running() and not future.done():
                            future.cancel()
                    process_futures(futures)
    finally:
        # Always save run log and summary, even on error
        _save_run_log(output_path)
        _generate_summary_report(output_path)
        logger.info(f"Logs saved to {output_path / 'logs'}")


if __name__ == "__main__":
    app()
