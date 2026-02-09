"""Unified experiment runner for exploration and review agents.

This module provides a single CLI entry point for running both pre-execution
(exploration) and post-execution (review) uncertainty estimation experiments
with consistent instance sampling.

The key advantage over separate scripts is that instances are sampled ONCE
at the start and the same instances are used for all requested agents,
ensuring consistent comparison.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import signal
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


# ============================================================================
# Logging Configuration
# ============================================================================

def configure_logging(verbose: bool = False) -> None:
    """Configure logging with Rich handler and suppress noisy warnings.
    
    Args:
        verbose: If True, show DEBUG level logs. Otherwise INFO only.
    """
    # Suppress noisy Pydantic serialization warnings from litellm
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*PydanticSerializationUnexpectedValue.*")
    
    # Suppress litellm verbose logging
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    # Configure root logger with Rich handler
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=verbose,
        )],
        force=True,  # Override any existing configuration
    )
    
    # Set our logger to show important messages
    logging.getLogger("agentic_uncertainty").setLevel(level)
    logging.getLogger("minisweagent").setLevel(logging.WARNING)


# Module logger
logger = logging.getLogger(__name__)

from agentic_uncertainty.config import get_settings
from agentic_uncertainty.data import SWEBenchProLoader, load_ground_truth
from agentic_uncertainty.data.trajectories import get_submission, load_trajectories
from agentic_uncertainty.elicitation import (
    EXPLORATION_METHODS,
    ExplorationElicitation,
    MID_EXECUTION_METHODS,
    MidExecutionElicitation,
    REVIEW_METHODS,
    ReviewElicitation,
    CheckpointElicitation,
    CheckpointPosthocElicitation,
)
from agentic_uncertainty.scripts._shared import (
    BaseAgentRunner,
    ExperimentConfig,
    ExplorationConfig,
    ReviewConfig,
    CheckpointConfig,
    CheckpointPosthocConfig,
    MidExecutionConfig,
    ResultCache,
    add_agent_selection_args,
    add_cache_args,
    add_checkpoint_args,
    add_checkpoint_posthoc_args,
    add_mid_execution_args,
    add_environment_args,
    add_exploration_args,
    add_output_args,
    add_parallel_args,
    add_review_args,
    add_sampling_args,
    atexit_cleanup,
    cleanup_all_environments,
    compute_standard_metrics,
    get_active_environment_count,
    is_shutdown_requested,
    match_trajectories_to_ground_truth,
    normalize_instance_id,
    print_metrics_table,
    print_summary_statistics,
    register_agent,
    request_shutdown,
    run_estimators_on_tasks,
    sample_items,
    save_experiment_results,
)

# Register atexit handler as fallback for cleanup
atexit.register(atexit_cleanup)


console = Console()


# ============================================================================
# Patch Truncation
# ============================================================================

# Maximum patch size in bytes before truncation (100KB ≈ 25K tokens)
MAX_PATCH_SIZE_BYTES = 100 * 1024


def truncate_patch(patch: str, max_bytes: int = MAX_PATCH_SIZE_BYTES) -> str:
    """Truncate a patch to fit within the maximum size limit.

    Truncates at file boundaries when possible to keep complete diffs.
    Adds a note at the end indicating truncation.

    Args:
        patch: The git diff patch string.
        max_bytes: Maximum size in bytes.

    Returns:
        The truncated patch, or original if within limit.
    """
    if len(patch.encode('utf-8')) <= max_bytes:
        return patch

    # Reserve space for truncation message
    truncation_msg = "\n\n[... patch truncated due to size limit ...]\n"
    available_bytes = max_bytes - len(truncation_msg.encode('utf-8'))

    # Try to truncate at file boundaries (diff --git lines)
    lines = patch.split('\n')
    truncated_lines = []
    current_size = 0
    last_file_boundary = 0

    for i, line in enumerate(lines):
        line_bytes = len((line + '\n').encode('utf-8'))
        if current_size + line_bytes > available_bytes:
            # Truncate at the last file boundary if possible
            if last_file_boundary > 0:
                truncated_lines = lines[:last_file_boundary]
            else:
                truncated_lines = lines[:i]
            break

        current_size += line_bytes
        truncated_lines.append(line)

        # Track file boundaries
        if line.startswith('diff --git'):
            last_file_boundary = i

    result = '\n'.join(truncated_lines) + truncation_msg
    logger.warning(
        f"Patch truncated from {len(patch)/1024:.1f}KB to {len(result)/1024:.1f}KB"
    )
    return result


# ============================================================================
# Timing and Progress Tracking
# ============================================================================


@dataclass
class TimingStats:
    """Track timing for experiment phases."""

    phases: dict[str, float] = field(default_factory=dict)
    _start_times: dict[str, float] = field(default_factory=dict)

    def start(self, phase: str) -> None:
        """Start timing a phase."""
        self._start_times[phase] = time.time()

    def stop(self, phase: str) -> float:
        """Stop timing a phase and return duration."""
        if phase not in self._start_times:
            return 0.0
        duration = time.time() - self._start_times[phase]
        self.phases[phase] = duration
        return duration

    def get(self, phase: str) -> float:
        """Get duration for a phase."""
        return self.phases.get(phase, 0.0)

    def total(self) -> float:
        """Get total time across all phases."""
        return sum(self.phases.values())


@dataclass
class ProgressStats:
    """Track progress for an agent."""

    total: int = 0
    completed: int = 0
    cached: int = 0
    failed: int = 0

    @property
    def remaining(self) -> int:
        return self.total - self.completed - self.cached - self.failed


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ============================================================================
# Display Functions
# ============================================================================


def print_config_table(config: ExperimentConfig, console: Console) -> None:
    """Print a formatted configuration summary table."""
    table = Table(title="Experiment Configuration", show_header=False, border_style="blue")
    table.add_column("Setting", style="cyan", width=20)
    table.add_column("Value", style="white")

    # General settings
    table.add_row("Model", config.model)
    table.add_row("Environment", config.environment_class)
    table.add_row("Instances", str(len(config.instance_ids)))
    table.add_row("Parallel", str(config.parallel))
    table.add_row("Agents", ", ".join(config.active_agents))

    # Paths
    table.add_row("Output", str(config.output_dir))
    if config.cache_dir:
        table.add_row("Cache", str(config.cache_dir))
    if config.ground_truth_path:
        table.add_row("Ground Truth", str(config.ground_truth_path))

    # Exploration settings
    if config.exploration:
        table.add_row("", "")  # Separator
        table.add_row("[bold]Exploration[/bold]", "")
        table.add_row("  Methods", ", ".join(config.exploration.methods))
        table.add_row("  Step Limit", str(config.exploration.step_limit))
        table.add_row("  Timeout", f"{config.exploration.timeout}s")

    # Review settings
    if config.review:
        table.add_row("", "")  # Separator
        table.add_row("[bold]Review[/bold]", "")
        table.add_row("  Methods", ", ".join(config.review.methods))
        table.add_row("  Step Limit", str(config.review.step_limit))
        table.add_row("  Cost Limit", f"${config.review.cost_limit:.2f}")
        table.add_row("  Timeout", f"{config.review.timeout}s")
        if config.review.traj_dir:
            table.add_row("  Trajectories", str(config.review.traj_dir))

    console.print()
    console.print(table)
    console.print()


def print_timing_summary(timing: TimingStats, console: Console) -> None:
    """Print a timing summary table."""
    if not timing.phases:
        return

    table = Table(title="Timing Summary", show_header=True, border_style="green")
    table.add_column("Phase", style="cyan")
    table.add_column("Duration", style="white", justify="right")

    for phase, duration in timing.phases.items():
        table.add_row(phase, format_duration(duration))

    table.add_row("", "")  # Separator
    table.add_row("[bold]Total[/bold]", f"[bold]{format_duration(timing.total())}[/bold]")

    console.print()
    console.print(table)


def create_progress_display(agents: list[str]) -> tuple[Progress, dict[str, int]]:
    """Create a Rich Progress display for tracking agent progress."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )

    task_ids = {}
    for agent in agents:
        task_ids[agent] = progress.add_task(f"{agent.capitalize()}", total=100)

    return progress, task_ids


def load_and_sample_instances(
    instance_ids_file: Path | None,
    num_samples: int | None,
    seed: int,
) -> list[str]:
    """Load instance IDs and sample ONCE for all agents.

    Args:
        instance_ids_file: Optional JSON file containing instance IDs.
        num_samples: Number of instances to sample (None = all).
        seed: Random seed for reproducible sampling.

    Returns:
        List of instance IDs to use for all agents.
    """
    # Load instance IDs from file if provided
    if instance_ids_file:
        with open(instance_ids_file) as f:
            data = json.load(f)
            if isinstance(data, list):
                # Handle list of strings or list of dicts with instance_id
                if data and isinstance(data[0], dict):
                    instance_ids = [item.get("instance_id") for item in data if item.get("instance_id")]
                else:
                    instance_ids = data
            elif isinstance(data, dict):
                instance_ids = list(data.keys())
            else:
                raise ValueError("Invalid instance IDs format: expected list or dict")

        console.print(f"Loaded {len(instance_ids)} instance IDs from {instance_ids_file}")

        # Sample if requested
        if num_samples and num_samples < len(instance_ids):
            instance_ids = sample_items(instance_ids, num_samples, seed)
            console.print(f"Sampled {len(instance_ids)} instances (seed={seed})")

        return instance_ids

    # Otherwise, sample from SWE-bench Pro dataset
    console.print("[bold]Loading SWE-bench Pro...[/bold]")
    loader = SWEBenchProLoader()

    if num_samples:
        tasks = loader.sample(n=num_samples, seed=seed)
        console.print(f"Sampled {len(tasks)} instances (seed={seed})")
    else:
        tasks = loader.load()
        console.print(f"Loaded {len(tasks)} instances")

    return [t.instance_id for t in tasks]


@register_agent("exploration")
class ExplorationRunner(BaseAgentRunner):
    """Runner for exploration agent experiments."""

    name = "exploration"
    experiment_type = "exploration"

    def __init__(self, config: ExperimentConfig, console: Console):
        super().__init__(config, console)
        self.progress_stats = ProgressStats()
        self.progress_callback: callable | None = None
        self.timing = TimingStats()

    async def run(self) -> dict[str, Any]:
        """Run exploration agent on configured instances."""
        exp_config = self.config.exploration
        if not exp_config:
            return {}

        self.timing.start("exploration")

        self.console.print("\n[bold cyan]Running Exploration Agent[/bold cyan]")
        self.console.print(f"Methods: {exp_config.methods}")
        self.console.print(f"Instances: {len(self.config.instance_ids)}")

        # Load tasks for the specified instance IDs
        loader = SWEBenchProLoader()
        all_tasks = {t.instance_id: t for t in loader.load()}

        tasks = []
        for iid in self.config.instance_ids:
            if iid in all_tasks:
                tasks.append(all_tasks[iid])
            else:
                normalized = normalize_instance_id(iid)
                for task_id, task in all_tasks.items():
                    if normalize_instance_id(task_id) == normalized:
                        tasks.append(task)
                        break
                else:
                    self.console.print(f"[yellow]Warning: {iid} not found in dataset[/yellow]")

        self.console.print(f"Matched {len(tasks)} tasks")

        # Create estimators
        estimators = {}
        for method in exp_config.methods:
            m = method.replace("exploration_", "") if method.startswith("exploration_") else method
            if m in EXPLORATION_METHODS:
                estimators[method] = ExplorationElicitation(
                    method=m,
                    step_limit=exp_config.step_limit,
                    timeout=exp_config.timeout,
                    step_timeout=exp_config.step_timeout,
                    environment_class=self.config.environment_class,
                    model=self.config.model,
                    model_class=self.config.model_class,
                )
            else:
                self.console.print(f"[yellow]Unknown exploration method: {method}[/yellow]")

        if not estimators:
            self.console.print("[red]No valid exploration methods specified[/red]")
            return {}

        # Initialize cache
        cache = self._init_cache("exploration")
        cached_results = self._load_cached_results(
            list(estimators.keys()),
            {t.instance_id for t in tasks},
        ) if cache else {}

        # Find tasks that need to run
        tasks_to_run = self._find_tasks_needing_run(tasks, list(estimators.keys()), cached_results)

        n_cached = len(tasks) - len(tasks_to_run)
        self.progress_stats.total = len(tasks)
        self.progress_stats.cached = n_cached

        if n_cached > 0:
            self.console.print(f"[green]Found {n_cached} cached, running {len(tasks_to_run)} new[/green]")

        # Update progress with cached count
        if self.progress_callback:
            self.progress_callback(self.name, self.progress_stats)

        # Define caching callback
        def on_result_callback(method: str, task, prediction: float | None, raw_response: str, metadata: dict | None):
            if not cache:
                return
            metadata = metadata or {}
            cache.set(method, task.instance_id, prediction, raw_response or "", metadata)
            messages = metadata.get("messages")
            if messages:
                cache.save_trajectory(
                    method=method,
                    instance_id=task.instance_id,
                    messages=messages,
                    exit_status=metadata.get("exit_status", "Unknown"),
                    cost=metadata.get("exploration_cost", 0.0),
                    n_steps=metadata.get("n_steps", 0),
                    prediction=prediction,
                )

        # Define progress callback for run_estimators_on_tasks
        def task_progress_callback(completed: int, total: int):
            self.progress_stats.completed = completed + n_cached
            if self.progress_callback:
                self.progress_callback(self.name, self.progress_stats)

        # Create step callback factory for --watch mode
        def create_step_callback_factory():
            if not self.config.watch:
                return None
            
            def step_callback_factory(task):
                instance_id = task.instance_id
                
                def step_callback(step_num: int, output: dict):
                    action = output.get("action", "")
                    result = output.get("output", "")
                    thought = output.get("thought", "")
                    # Truncate long outputs
                    if len(result) > 1000:
                        result = result[:1000] + "\n... (truncated)"
                    self.console.print(f"\n[bold cyan] Step {step_num}[/bold cyan]")
                    if thought:
                        # Truncate very long thoughts
                        if len(thought) > 500:
                            thought = thought[:500] + "..."
                        self.console.print(f"[dim italic]{thought}[/dim italic]")
                    self.console.print(f"[yellow]$ {action}[/yellow]")
                    if result.strip():
                        self.console.print(result)
                    self.console.print("[dim]" + "─" * 60 + "[/dim]")
                
                return step_callback
            
            return step_callback_factory
        
        step_callback_factory = create_step_callback_factory()

        # Run exploration
        if tasks_to_run:
            if self.config.parallel > 1:
                self.console.print(f"[bold]Parallel tasks:[/bold] {self.config.parallel}")
            new_results = await run_estimators_on_tasks(
                tasks=tasks_to_run,
                estimators=estimators,
                console=self.console,
                max_concurrent_tasks=self.config.parallel,
                on_result=on_result_callback,
                cache=cache,
                progress_callback=task_progress_callback,
                show_tqdm=self.progress_callback is None,  # Hide tqdm when using Rich Progress
                step_callback_factory=step_callback_factory,
            )
        else:
            new_results = {method: {"predictions": [], "raw_responses": [], "metadata": []}
                          for method in estimators}
            self.console.print("[green]All instances already cached[/green]")

        # Merge results
        results = self._merge_cached_and_new_results(
            tasks, estimators, cached_results, new_results, tasks_to_run
        )

        # Compute metrics
        metrics_by_method = {}
        if self.config.ground_truth_path and self.config.ground_truth_path.exists():
            self.console.print(f"\n[bold]Computing metrics with ground truth:[/bold] {self.config.ground_truth_path}")
            ground_truth = load_ground_truth(self.config.ground_truth_path)
            task_ids = [t.instance_id for t in tasks]
            metrics_by_method = self._compute_and_print_metrics(
                results, task_ids, ground_truth, title="Exploration Metrics"
            )

        # Save results
        output_dir = self._get_output_dir("exploration")
        save_experiment_results(
            output_dir=output_dir,
            model_name=self.config.model,
            seed=42,
            tasks=tasks,
            estimators=estimators,
            results=results,
            metrics=metrics_by_method,
            experiment_type="exploration",
            extra_metadata={"ground_truth_path": str(self.config.ground_truth_path)} if self.config.ground_truth_path else None,
        )
        self.console.print(f"[green]Exploration results saved to:[/green] {output_dir}")

        print_summary_statistics(results, self.console)

        self.timing.stop("exploration")
        return {"results": results, "timing": self.timing, "stats": self.progress_stats}


@register_agent("review")
class ReviewRunner(BaseAgentRunner):
    """Runner for review agent experiments."""

    name = "review"
    experiment_type = "review"

    def __init__(self, config: ExperimentConfig, console: Console):
        super().__init__(config, console)
        self.progress_stats = ProgressStats()
        self.progress_callback: callable | None = None
        self.timing = TimingStats()

    async def run(self) -> dict[str, Any]:
        """Run review agent on configured instances."""
        review_config = self.config.review
        if not review_config:
            return {}

        self.timing.start("review")

        self.console.print("\n[bold cyan]Running Review Agent[/bold cyan]")
        self.console.print(f"Methods: {review_config.methods}")
        self.console.print(f"Trajectory dir: {review_config.traj_dir}")
        self.console.print(f"Instances: {len(self.config.instance_ids)}")

        # Validate methods
        for method in review_config.methods:
            if method not in REVIEW_METHODS:
                raise ValueError(f"Unknown review method: {method}. Available: {list(REVIEW_METHODS.keys())}")

        # Load ground truth
        if not self.config.ground_truth_path or not self.config.ground_truth_path.exists():
            self.console.print("[red]Ground truth file required for review[/red]")
            return {}

        ground_truth = load_ground_truth(self.config.ground_truth_path)
        self.console.print(f"Ground truth: {len(ground_truth)} instances")

        # Load trajectories
        trajectories = load_trajectories(review_config.traj_dir)
        self.console.print(f"Loaded {len(trajectories)} trajectories")

        # Match and filter trajectories
        matched_trajs = match_trajectories_to_ground_truth(trajectories, ground_truth)
        self.console.print(f"Matched {len(matched_trajs)} trajectories to ground truth")

        if not matched_trajs:
            self.console.print("[red]No trajectories matched ground truth![/red]")
            return {}

        # Filter to specified instance IDs
        id_set = set(self.config.instance_ids)
        for iid in list(id_set):
            id_set.add(normalize_instance_id(iid))
            if not iid.startswith("instance_"):
                id_set.add(f"instance_{iid}")

        filtered_trajs = [
            traj for traj in matched_trajs
            if traj.get("matched_id", traj.get("instance_id", "")) in id_set
            or normalize_instance_id(traj.get("matched_id", traj.get("instance_id", ""))) in id_set
        ]
        self.console.print(f"Filtered to {len(filtered_trajs)} instances")

        # Filter to trajectories with patches
        trajs_with_patches = []
        for traj in filtered_trajs:
            patch = get_submission(traj)
            if patch:
                traj["patch"] = patch
                trajs_with_patches.append(traj)

        self.console.print(f"Trajectories with patches: {len(trajs_with_patches)}")

        if not trajs_with_patches:
            self.console.print("[red]No trajectories have patches![/red]")
            return {}

        # Load problem statements
        loader = SWEBenchProLoader()
        all_tasks = {t.instance_id: t for t in loader.load()}

        # Create estimators
        estimators = {
            method: ReviewElicitation(
                method=method,
                step_limit=review_config.step_limit,
                cost_limit=review_config.cost_limit,
                timeout=review_config.timeout,
                step_timeout=review_config.step_timeout,
                environment_class=self.config.environment_class,
                model=self.config.model,
                model_class=self.config.model_class,
            )
            for method in review_config.methods
        }

        # Initialize cache
        cache = self._init_cache("review")

        # Set up progress tracking
        total_tasks = len(trajs_with_patches) * len(review_config.methods)
        self.progress_stats.total = total_tasks

        # Run evaluation
        self.console.print(f"\n[bold]Running review evaluation...[/bold]")
        if self.config.parallel > 1:
            self.console.print(f"Parallel tasks: {self.config.parallel}")

        results = {
            method: {
                "predictions": [],
                "raw_responses": [],
                "instance_ids": [],
                "labels": [],
                "exit_statuses": [],
                "n_steps": [],
                "costs": [],
            }
            for method in review_config.methods
        }

        cached_count = 0
        new_count = 0
        failed_count = 0
        semaphore = asyncio.Semaphore(self.config.parallel)

        # Create tqdm progress bar (shown when not using external progress callback)
        from tqdm import tqdm
        pbar = tqdm(
            total=total_tasks,
            desc="Review evaluation",
            disable=self.progress_callback is not None,
        )

        def update_progress():
            """Update progress stats and call callback."""
            self.progress_stats.completed = new_count
            self.progress_stats.cached = cached_count
            self.progress_stats.failed = failed_count
            pbar.update(1)
            if self.progress_callback:
                self.progress_callback(self.name, self.progress_stats)

        def create_step_callback(instance_id: str):
            """Create step callback for --watch mode."""
            if not self.config.watch:
                return None
            
            def step_callback(step_num: int, output: dict):
                action = output.get("action", "")
                result = output.get("output", "")
                thought = output.get("thought", "")
                # Truncate long outputs
                if len(result) > 1000:
                    result = result[:1000] + "\n... (truncated)"
                self.console.print(f"\n[bold cyan] Step {step_num}[/bold cyan]")
                if thought:
                    # Truncate very long thoughts
                    if len(thought) > 500:
                        thought = thought[:500] + "..."
                    self.console.print(f"[dim italic]{thought}[/dim italic]")
                self.console.print(f"[yellow]$ {action}[/yellow]")
                if result.strip():
                    self.console.print(result)
                self.console.print("[dim]" + "─" * 60 + "[/dim]")
            
            return step_callback

        async def process_trajectory_method(traj: dict, method: str, estimator: ReviewElicitation):
            nonlocal cached_count, new_count, failed_count
            async with semaphore:
                instance_id = traj.get("matched_id", traj.get("instance_id"))
                label = traj["ground_truth"]
                patch = truncate_patch(traj["patch"])

                method_cache_key = f"review_{method}"
                if cache:
                    cached_result = cache.get(method_cache_key, instance_id)
                    if cached_result:
                        cached_prediction = cached_result.get("prediction")
                        if cached_prediction is None:
                            failed_count += 1
                            update_progress()
                            return
                        results[method]["predictions"].append(cached_prediction)
                        results[method]["raw_responses"].append(cached_result.get("raw_response"))
                        results[method]["instance_ids"].append(instance_id)
                        results[method]["labels"].append(label)
                        results[method]["exit_statuses"].append(cached_result.get("metadata", {}).get("exit_status", ""))
                        results[method]["n_steps"].append(cached_result.get("metadata", {}).get("n_steps", 0))
                        results[method]["costs"].append(cached_result.get("metadata", {}).get("review_cost", 0.0))
                        cached_count += 1
                        update_progress()
                        return

                # Find matching task
                task = None
                for task_id, t in all_tasks.items():
                    if task_id == instance_id or normalize_instance_id(task_id) == normalize_instance_id(instance_id):
                        task = t
                        break

                if not task:
                    failed_count += 1
                    update_progress()
                    return

                try:
                    step_callback = create_step_callback(instance_id)
                    result = await estimator.estimate_patch(
                        problem_statement=task.problem_statement,
                        patch=patch,
                        repo=task.repo,
                        base_commit=task.base_commit,
                        instance_id=task.instance_id,
                        cache=cache,
                        step_callback=step_callback,
                    )

                    if result.probability is None:
                        if cache:
                            cache.set(
                                method=method_cache_key,
                                instance_id=instance_id,
                                prediction=None,
                                raw_response=result.raw_response,
                                metadata=result.metadata,
                            )
                        failed_count += 1
                        update_progress()
                        return

                    results[method]["predictions"].append(result.probability)
                    results[method]["raw_responses"].append(result.raw_response)
                    results[method]["instance_ids"].append(instance_id)
                    results[method]["labels"].append(label)
                    results[method]["exit_statuses"].append(result.metadata.get("exit_status", ""))
                    results[method]["n_steps"].append(result.metadata.get("n_steps", 0))
                    results[method]["costs"].append(result.metadata.get("review_cost", 0.0))
                    new_count += 1

                    if cache:
                        cache.set(
                            method=method_cache_key,
                            instance_id=instance_id,
                            prediction=result.probability,
                            raw_response=result.raw_response,
                            metadata=result.metadata,
                        )
                        messages = result.metadata.get("messages") if result.metadata else None
                        if messages:
                            cache.save_trajectory(
                                method=method_cache_key,
                                instance_id=instance_id,
                                messages=messages,
                                exit_status=result.metadata.get("exit_status", "Unknown"),
                                cost=result.metadata.get("review_cost", 0.0),
                                n_steps=result.metadata.get("n_steps", 0),
                                prediction=result.probability,
                            )
                    update_progress()
                except Exception as e:
                    self.console.print(f"[red]Error on {instance_id}/{method}: {e}[/red]")
                    failed_count += 1
                    update_progress()

        # Create all tasks
        review_tasks = []
        for traj in trajs_with_patches:
            for method, estimator in estimators.items():
                review_tasks.append(process_trajectory_method(traj, method, estimator))

        await asyncio.gather(*review_tasks)
        pbar.close()

        if cached_count > 0 or new_count > 0:
            self.console.print(f"[green]Cache: {cached_count} cached, {new_count} new[/green]")

        # Compute metrics
        self.console.print("\n[bold]Computing metrics...[/bold]")
        metrics = {}

        for method in review_config.methods:
            preds = results[method]["predictions"]
            labels = results[method]["labels"]

            if len(preds) == 0:
                continue

            method_metrics = compute_standard_metrics(preds, labels)
            method_metrics["avg_steps"] = (
                sum(results[method]["n_steps"]) / len(results[method]["n_steps"])
                if results[method]["n_steps"] else 0
            )
            method_metrics["avg_cost"] = (
                sum(results[method]["costs"]) / len(results[method]["costs"])
                if results[method]["costs"] else 0
            )
            metrics[method] = method_metrics

        # Save results
        output_dir = self._get_output_dir("review")
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_type": "review",
                    "model": self.config.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_trajectories": len(trajs_with_patches),
                    "methods": review_config.methods,
                    "review_config": {
                        "step_limit": review_config.step_limit,
                        "cost_limit": review_config.cost_limit,
                        "timeout": review_config.timeout,
                    },
                    "results": {
                        method: {
                            "predictions": results[method]["predictions"],
                            "instance_ids": results[method]["instance_ids"],
                            "labels": results[method]["labels"],
                            "exit_statuses": results[method]["exit_statuses"],
                            "n_steps": results[method]["n_steps"],
                            "costs": results[method]["costs"],
                        }
                        for method in review_config.methods
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        self.console.print(f"[green]Review results saved to:[/green] {results_file}")

        # Print summary
        self.console.print("\n[bold]Review Results Summary[/bold]")
        table_data = [{"method": method, **m} for method, m in metrics.items()]
        print_metrics_table(
            table_data,
            ["method", "n_samples", "auroc", "ece", "brier", "overconfidence"],
            self.console,
        )

        self.timing.stop("review")
        return {"results": results, "metrics": metrics, "timing": self.timing, "stats": self.progress_stats}


@register_agent("checkpoint")
class CheckpointRunner(BaseAgentRunner):
    """Runner for checkpoint agent experiments (live confidence tracking)."""

    name = "checkpoint"
    experiment_type = "checkpoint"

    def __init__(self, config: ExperimentConfig, console: Console):
        super().__init__(config, console)
        self.progress_stats = ProgressStats()
        self.progress_callback: callable | None = None
        self.timing = TimingStats()

    async def run(self) -> dict[str, Any]:
        """Run checkpoint agent on configured instances."""
        checkpoint_config = self.config.checkpoint
        if not checkpoint_config:
            return {}

        self.timing.start("checkpoint")

        self.console.print("\n[bold cyan]Running Checkpoint Agent[/bold cyan]")
        self.console.print(f"Confidence interval: every {checkpoint_config.confidence_interval} steps")
        self.console.print(f"Instances: {len(self.config.instance_ids)}")

        # Load tasks for the specified instance IDs
        loader = SWEBenchProLoader()
        all_tasks = {t.instance_id: t for t in loader.load()}

        tasks = []
        for iid in self.config.instance_ids:
            if iid in all_tasks:
                tasks.append(all_tasks[iid])
            else:
                normalized = normalize_instance_id(iid)
                for task_id, task in all_tasks.items():
                    if normalize_instance_id(task_id) == normalized:
                        tasks.append(task)
                        break
                else:
                    self.console.print(f"[yellow]Warning: {iid} not found in dataset[/yellow]")

        self.console.print(f"Matched {len(tasks)} tasks")

        # Create checkpoint estimator
        estimator = CheckpointElicitation(
            confidence_interval=checkpoint_config.confidence_interval,
            step_limit=checkpoint_config.step_limit,
            timeout=checkpoint_config.timeout,
            step_timeout=checkpoint_config.step_timeout,
            environment_class=self.config.environment_class,
            model=self.config.model,
            model_class=self.config.model_class,
        )

        # Initialize cache
        cache = self._init_cache("checkpoint")

        # Set up progress tracking
        self.progress_stats.total = len(tasks)

        # Run checkpoint elicitation
        results = {
            "predictions": [],
            "raw_responses": [],
            "instance_ids": [],
            "confidence_traces": [],
            "exit_statuses": [],
            "n_steps": [],
            "costs": [],
        }

        completed = 0
        cached_count = 0
        semaphore = asyncio.Semaphore(self.config.parallel)

        async def process_task(task):
            nonlocal completed, cached_count
            async with semaphore:
                # Check cache first
                if cache:
                    cached_result = cache.get("checkpoint", task.instance_id)
                    if cached_result:
                        results["predictions"].append(cached_result.get("prediction"))
                        results["raw_responses"].append(cached_result.get("raw_response", ""))
                        results["instance_ids"].append(task.instance_id)
                        results["confidence_traces"].append(cached_result.get("metadata", {}).get("confidence_trace", []))
                        results["exit_statuses"].append(cached_result.get("metadata", {}).get("exit_status", ""))
                        results["n_steps"].append(cached_result.get("metadata", {}).get("n_steps", 0))
                        results["costs"].append(cached_result.get("metadata", {}).get("checkpoint_cost", 0.0))
                        cached_count += 1
                        completed += 1
                        self.progress_stats.completed = completed
                        self.progress_stats.cached = cached_count
                        if self.progress_callback:
                            self.progress_callback(self.name, self.progress_stats)
                        return

                try:
                    result = await estimator.estimate(task)
                    results["predictions"].append(result.probability)
                    results["raw_responses"].append(result.raw_response)
                    results["instance_ids"].append(task.instance_id)
                    results["confidence_traces"].append(result.metadata.get("confidence_trace", []))
                    results["exit_statuses"].append(result.metadata.get("exit_status", ""))
                    results["n_steps"].append(result.metadata.get("n_steps", 0))
                    results["costs"].append(result.metadata.get("checkpoint_cost", 0.0))

                    if cache:
                        cache.set(
                            method="checkpoint",
                            instance_id=task.instance_id,
                            prediction=result.probability,
                            raw_response=result.raw_response,
                            metadata=result.metadata,
                        )
                except Exception as e:
                    self.console.print(f"[red]Error on {task.instance_id}: {e}[/red]")

                completed += 1
                self.progress_stats.completed = completed
                if self.progress_callback:
                    self.progress_callback(self.name, self.progress_stats)

        self.console.print(f"\n[bold]Running checkpoint evaluation...[/bold]")
        if self.config.parallel > 1:
            self.console.print(f"Parallel tasks: {self.config.parallel}")

        await asyncio.gather(*[process_task(task) for task in tasks])

        if cached_count > 0:
            self.console.print(f"[green]Cache: {cached_count} cached, {len(tasks) - cached_count} new[/green]")

        # Compute metrics if ground truth available
        metrics = {}
        if self.config.ground_truth_path and self.config.ground_truth_path.exists():
            self.console.print(f"\n[bold]Computing metrics with ground truth:[/bold] {self.config.ground_truth_path}")
            ground_truth = load_ground_truth(self.config.ground_truth_path)

            # Match predictions to ground truth
            labels = []
            for iid in results["instance_ids"]:
                normalized = normalize_instance_id(iid)
                label = ground_truth.get(iid) or ground_truth.get(normalized) or ground_truth.get(f"instance_{normalized}")
                labels.append(1 if label else 0)

            if results["predictions"]:
                metrics = compute_standard_metrics(results["predictions"], labels)

        # Save results
        output_dir = self._get_output_dir("checkpoint")
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_type": "checkpoint",
                    "model": self.config.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_instances": len(tasks),
                    "checkpoint_config": {
                        "step_limit": checkpoint_config.step_limit,
                        "timeout": checkpoint_config.timeout,
                        "confidence_interval": checkpoint_config.confidence_interval,
                    },
                    "results": results,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        self.console.print(f"[green]Checkpoint results saved to:[/green] {results_file}")

        self.timing.stop("checkpoint")
        return {"results": results, "metrics": metrics, "timing": self.timing, "stats": self.progress_stats}


@register_agent("checkpoint_posthoc")
class CheckpointPosthocRunner(BaseAgentRunner):
    """Runner for post-hoc checkpoint analysis of existing trajectories."""

    name = "checkpoint_posthoc"
    experiment_type = "checkpoint_posthoc"

    def __init__(self, config: ExperimentConfig, console: Console):
        super().__init__(config, console)
        self.progress_stats = ProgressStats()
        self.progress_callback: callable | None = None
        self.timing = TimingStats()

    async def run(self) -> dict[str, Any]:
        """Run post-hoc checkpoint analysis on trajectories."""
        posthoc_config = self.config.checkpoint_posthoc
        if not posthoc_config:
            return {}

        self.timing.start("checkpoint_posthoc")

        self.console.print("\n[bold cyan]Running Checkpoint Post-hoc Analysis[/bold cyan]")
        self.console.print(f"Confidence interval: every {posthoc_config.confidence_interval} steps")
        self.console.print(f"Trajectory dir: {posthoc_config.traj_dir}")

        if not posthoc_config.traj_dir:
            self.console.print("[red]No trajectory directory specified for checkpoint_posthoc[/red]")
            return {}

        # Load ground truth
        if not self.config.ground_truth_path or not self.config.ground_truth_path.exists():
            self.console.print("[red]Ground truth file required for checkpoint_posthoc[/red]")
            return {}

        ground_truth = load_ground_truth(self.config.ground_truth_path)
        self.console.print(f"Ground truth: {len(ground_truth)} instances")

        # Load trajectories
        trajectories = load_trajectories(posthoc_config.traj_dir)
        self.console.print(f"Loaded {len(trajectories)} trajectories")

        # Match and filter trajectories
        matched_trajs = match_trajectories_to_ground_truth(trajectories, ground_truth)
        self.console.print(f"Matched {len(matched_trajs)} trajectories to ground truth")

        if not matched_trajs:
            self.console.print("[red]No trajectories matched ground truth![/red]")
            return {}

        # Filter to specified instance IDs
        id_set = set(self.config.instance_ids)
        for iid in list(id_set):
            id_set.add(normalize_instance_id(iid))
            if not iid.startswith("instance_"):
                id_set.add(f"instance_{iid}")

        filtered_trajs = [
            traj for traj in matched_trajs
            if traj.get("matched_id", traj.get("instance_id", "")) in id_set
            or normalize_instance_id(traj.get("matched_id", traj.get("instance_id", ""))) in id_set
        ]
        self.console.print(f"Filtered to {len(filtered_trajs)} instances")

        # Create estimator with CLI-specified model
        settings = replace(get_settings(), model=self.config.model)
        estimator = CheckpointPosthocElicitation(
            confidence_interval=posthoc_config.confidence_interval,
            settings=settings,
        )

        # Initialize cache
        cache = self._init_cache("checkpoint_posthoc")

        # Set up progress tracking
        self.progress_stats.total = len(filtered_trajs)

        # Run post-hoc analysis
        results = {
            "predictions": [],
            "instance_ids": [],
            "labels": [],
            "confidence_traces": [],
            "total_steps": [],
        }

        completed = 0
        cached_count = 0
        semaphore = asyncio.Semaphore(self.config.parallel)

        async def process_trajectory(traj):
            nonlocal completed, cached_count
            async with semaphore:
                instance_id = traj.get("matched_id", traj.get("instance_id"))
                traj_path = traj.get("traj_path") or traj.get("path")
                resolved = traj.get("ground_truth", False)

                if not traj_path:
                    self.console.print(f"[yellow]Warning: No trajectory path for {instance_id}[/yellow]")
                    completed += 1
                    return

                # Check cache first
                if cache:
                    cached_result = cache.get("checkpoint_posthoc", instance_id)
                    if cached_result:
                        results["predictions"].append(cached_result.get("prediction"))
                        results["instance_ids"].append(instance_id)
                        results["labels"].append(1 if resolved else 0)
                        results["confidence_traces"].append(cached_result.get("metadata", {}).get("confidence_trace", []))
                        results["total_steps"].append(cached_result.get("metadata", {}).get("total_agent_turns", 0))
                        cached_count += 1
                        completed += 1
                        self.progress_stats.completed = completed
                        self.progress_stats.cached = cached_count
                        if self.progress_callback:
                            self.progress_callback(self.name, self.progress_stats)
                        return

                try:
                    result = await estimator.estimate_trajectory(traj_path, resolved)
                    results["predictions"].append(result.probability)
                    results["instance_ids"].append(instance_id)
                    results["labels"].append(1 if resolved else 0)
                    results["confidence_traces"].append(result.metadata.get("confidence_trace", []))
                    results["total_steps"].append(result.metadata.get("total_agent_turns", 0))

                    # Save to cache
                    if cache:
                        cache.set(
                            method="checkpoint_posthoc",
                            instance_id=instance_id,
                            prediction=result.probability,
                            raw_response=result.raw_response,
                            metadata=result.metadata,
                        )
                except Exception as e:
                    self.console.print(f"[red]Error on {instance_id}: {e}[/red]")

                completed += 1
                self.progress_stats.completed = completed
                if self.progress_callback:
                    self.progress_callback(self.name, self.progress_stats)

        self.console.print(f"\n[bold]Running post-hoc checkpoint analysis...[/bold]")
        if self.config.parallel > 1:
            self.console.print(f"Parallel tasks: {self.config.parallel}")

        await asyncio.gather(*[process_trajectory(traj) for traj in filtered_trajs])

        if cached_count > 0:
            self.console.print(f"[green]Cache: {cached_count} cached, {len(filtered_trajs) - cached_count} new[/green]")

        # Compute metrics
        metrics = {}
        if results["predictions"]:
            metrics = compute_standard_metrics(results["predictions"], results["labels"])

        # Save results
        output_dir = self._get_output_dir("checkpoint_posthoc")
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_type": "checkpoint_posthoc",
                    "model": self.config.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_trajectories": len(filtered_trajs),
                    "posthoc_config": {
                        "confidence_interval": posthoc_config.confidence_interval,
                    },
                    "results": results,
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        self.console.print(f"[green]Checkpoint post-hoc results saved to:[/green] {results_file}")

        # Print summary
        if metrics:
            self.console.print("\n[bold]Checkpoint Post-hoc Results Summary[/bold]")
            print_metrics_table(
                [{"method": "checkpoint_posthoc", **metrics}],
                ["method", "n_samples", "auroc", "ece", "brier", "overconfidence"],
                self.console,
            )

        self.timing.stop("checkpoint_posthoc")
        return {"results": results, "metrics": metrics, "timing": self.timing, "stats": self.progress_stats}


@register_agent("mid_execution")
class MidExecutionRunner(BaseAgentRunner):
    """Runner for mid-execution trajectory evaluation experiments."""

    name = "mid_execution"
    experiment_type = "mid_execution"

    def __init__(self, config: ExperimentConfig, console: Console):
        super().__init__(config, console)
        self.progress_stats = ProgressStats()
        self.progress_callback: callable | None = None
        self.timing = TimingStats()

    async def run(self) -> dict[str, Any]:
        """Run mid-execution evaluation on configured instances."""
        mid_config = self.config.mid_execution
        if not mid_config:
            return {}

        self.timing.start("mid_execution")

        progress_pct = int(mid_config.progress_fraction * 100)
        self.console.print("\n[bold cyan]Running Mid-Execution Evaluation Agent[/bold cyan]")
        self.console.print(f"Progress fraction: {progress_pct}%")
        self.console.print(f"Trajectory dir: {mid_config.traj_dir}")
        self.console.print(f"Instances: {len(self.config.instance_ids)}")

        # Load ground truth
        if not self.config.ground_truth_path or not self.config.ground_truth_path.exists():
            self.console.print("[red]Ground truth file required for mid_execution[/red]")
            return {}

        ground_truth = load_ground_truth(self.config.ground_truth_path)
        self.console.print(f"Ground truth: {len(ground_truth)} instances")

        # Load trajectories
        trajectories = load_trajectories(mid_config.traj_dir)
        self.console.print(f"Loaded {len(trajectories)} trajectories")

        # Match and filter trajectories
        matched_trajs = match_trajectories_to_ground_truth(trajectories, ground_truth)
        self.console.print(f"Matched {len(matched_trajs)} trajectories to ground truth")

        if not matched_trajs:
            self.console.print("[red]No trajectories matched ground truth![/red]")
            return {}

        # Filter to specified instance IDs
        id_set = set(self.config.instance_ids)
        for iid in list(id_set):
            id_set.add(normalize_instance_id(iid))
            if not iid.startswith("instance_"):
                id_set.add(f"instance_{iid}")

        filtered_trajs = [
            traj for traj in matched_trajs
            if traj.get("matched_id", traj.get("instance_id", "")) in id_set
            or normalize_instance_id(traj.get("matched_id", traj.get("instance_id", ""))) in id_set
        ]
        self.console.print(f"Filtered to {len(filtered_trajs)} instances")

        if not filtered_trajs:
            self.console.print("[red]No trajectories match instance IDs![/red]")
            return {}

        # Load problem statements
        loader = SWEBenchProLoader()
        all_tasks = {t.instance_id: t for t in loader.load()}

        # Create estimator
        estimator = MidExecutionElicitation(
            progress_fraction=mid_config.progress_fraction,
            method="direct",
            step_limit=mid_config.step_limit,
            timeout=mid_config.timeout,
            step_timeout=mid_config.step_timeout,
            environment_class=self.config.environment_class,
            model=self.config.model,
            model_class=self.config.model_class,
        )

        # Initialize cache
        cache = self._init_cache("mid_execution")
        method_cache_key = f"mid_execution_direct_{progress_pct}pct"

        # Set up progress tracking
        self.progress_stats.total = len(filtered_trajs)

        # Run evaluation
        self.console.print(f"\n[bold]Running mid-execution evaluation...[/bold]")
        if self.config.parallel > 1:
            self.console.print(f"Parallel tasks: {self.config.parallel}")

        results = {
            "predictions": [],
            "raw_responses": [],
            "instance_ids": [],
            "labels": [],
            "exit_statuses": [],
            "n_steps": [],
            "costs": [],
            "checkpoint_steps": [],
            "total_trajectory_steps": [],
        }

        cached_count = 0
        new_count = 0
        failed_count = 0
        semaphore = asyncio.Semaphore(self.config.parallel)

        # Create tqdm progress bar
        from tqdm import tqdm
        pbar = tqdm(
            total=len(filtered_trajs),
            desc=f"Mid-execution ({progress_pct}%)",
            disable=self.progress_callback is not None,
        )

        def update_progress():
            """Update progress stats and call callback."""
            self.progress_stats.completed = new_count
            self.progress_stats.cached = cached_count
            self.progress_stats.failed = failed_count
            pbar.update(1)
            if self.progress_callback:
                self.progress_callback(self.name, self.progress_stats)

        def create_step_callback(instance_id: str):
            """Create step callback for --watch mode."""
            if not self.config.watch:
                return None

            def step_callback(step_num: int, output: dict):
                action = output.get("action", "")
                result = output.get("output", "")
                thought = output.get("thought", "")
                if len(result) > 1000:
                    result = result[:1000] + "\n... (truncated)"
                self.console.print(f"\n[bold cyan] Step {step_num}[/bold cyan]")
                if thought:
                    if len(thought) > 500:
                        thought = thought[:500] + "..."
                    self.console.print(f"[dim italic]{thought}[/dim italic]")
                self.console.print(f"[yellow]$ {action}[/yellow]")
                if result.strip():
                    self.console.print(result)
                self.console.print("[dim]" + "─" * 60 + "[/dim]")

            return step_callback

        async def process_trajectory(traj: dict):
            nonlocal cached_count, new_count, failed_count
            async with semaphore:
                instance_id = traj.get("matched_id", traj.get("instance_id"))
                label = traj.get("ground_truth", False)

                # Check cache
                if cache:
                    cached_result = cache.get(method_cache_key, instance_id)
                    if cached_result:
                        cached_prediction = cached_result.get("prediction")
                        if cached_prediction is None:
                            failed_count += 1
                            update_progress()
                            return
                        results["predictions"].append(cached_prediction)
                        results["raw_responses"].append(cached_result.get("raw_response"))
                        results["instance_ids"].append(instance_id)
                        results["labels"].append(1 if label else 0)
                        meta = cached_result.get("metadata", {})
                        results["exit_statuses"].append(meta.get("exit_status", ""))
                        results["n_steps"].append(meta.get("n_steps", 0))
                        results["costs"].append(meta.get("mid_execution_cost", 0.0))
                        results["checkpoint_steps"].append(meta.get("checkpoint_step", 0))
                        results["total_trajectory_steps"].append(meta.get("total_trajectory_steps", 0))
                        cached_count += 1
                        update_progress()
                        return

                # Find matching task
                task = None
                for task_id, t in all_tasks.items():
                    if task_id == instance_id or normalize_instance_id(task_id) == normalize_instance_id(instance_id):
                        task = t
                        break

                if not task:
                    self.console.print(f"[yellow]Warning: Task not found for {instance_id}[/yellow]")
                    failed_count += 1
                    update_progress()
                    return

                try:
                    step_callback = create_step_callback(instance_id)
                    result = await estimator.estimate_trajectory(
                        problem_statement=task.problem_statement,
                        trajectory=traj,
                        repo=task.repo,
                        base_commit=task.base_commit,
                        instance_id=task.instance_id,
                        cache=cache,
                        step_callback=step_callback,
                    )

                    if result.probability is None:
                        if cache:
                            cache.set(
                                method=method_cache_key,
                                instance_id=instance_id,
                                prediction=None,
                                raw_response=result.raw_response,
                                metadata=result.metadata,
                            )
                        failed_count += 1
                        update_progress()
                        return

                    results["predictions"].append(result.probability)
                    results["raw_responses"].append(result.raw_response)
                    results["instance_ids"].append(instance_id)
                    results["labels"].append(1 if label else 0)
                    results["exit_statuses"].append(result.metadata.get("exit_status", ""))
                    results["n_steps"].append(result.metadata.get("n_steps", 0))
                    results["costs"].append(result.metadata.get("mid_execution_cost", 0.0))
                    results["checkpoint_steps"].append(result.metadata.get("checkpoint_step", 0))
                    results["total_trajectory_steps"].append(result.metadata.get("total_trajectory_steps", 0))
                    new_count += 1

                    if cache:
                        cache.set(
                            method=method_cache_key,
                            instance_id=instance_id,
                            prediction=result.probability,
                            raw_response=result.raw_response,
                            metadata=result.metadata,
                        )
                        messages = result.metadata.get("messages") if result.metadata else None
                        if messages:
                            cache.save_trajectory(
                                method=method_cache_key,
                                instance_id=instance_id,
                                messages=messages,
                                exit_status=result.metadata.get("exit_status", "Unknown"),
                                cost=result.metadata.get("mid_execution_cost", 0.0),
                                n_steps=result.metadata.get("n_steps", 0),
                                prediction=result.probability,
                            )
                    update_progress()
                except Exception as e:
                    self.console.print(f"[red]Error on {instance_id}: {e}[/red]")
                    failed_count += 1
                    update_progress()

        # Process all trajectories
        await asyncio.gather(*[process_trajectory(traj) for traj in filtered_trajs])
        pbar.close()

        if cached_count > 0 or new_count > 0:
            self.console.print(f"[green]Cache: {cached_count} cached, {new_count} new, {failed_count} failed[/green]")

        # Compute metrics
        metrics = {}
        if results["predictions"]:
            metrics = compute_standard_metrics(results["predictions"], results["labels"])
            metrics["avg_steps"] = (
                sum(results["n_steps"]) / len(results["n_steps"])
                if results["n_steps"] else 0
            )
            metrics["avg_cost"] = (
                sum(results["costs"]) / len(results["costs"])
                if results["costs"] else 0
            )

        # Save results
        output_dir = self._get_output_dir(f"mid_execution_{progress_pct}pct")
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "experiment_type": "mid_execution",
                    "model": self.config.model,
                    "timestamp": datetime.now().isoformat(),
                    "n_trajectories": len(filtered_trajs),
                    "mid_execution_config": {
                        "step_limit": mid_config.step_limit,
                        "timeout": mid_config.timeout,
                        "progress_fraction": mid_config.progress_fraction,
                    },
                    "results": {
                        "predictions": results["predictions"],
                        "instance_ids": results["instance_ids"],
                        "labels": results["labels"],
                        "exit_statuses": results["exit_statuses"],
                        "n_steps": results["n_steps"],
                        "costs": results["costs"],
                        "checkpoint_steps": results["checkpoint_steps"],
                        "total_trajectory_steps": results["total_trajectory_steps"],
                    },
                    "metrics": metrics,
                },
                f,
                indent=2,
            )
        self.console.print(f"[green]Mid-execution results saved to:[/green] {results_file}")

        # Print summary
        if metrics:
            self.console.print(f"\n[bold]Mid-Execution Results Summary ({progress_pct}%)[/bold]")
            print_metrics_table(
                [{"method": f"mid_execution_{progress_pct}%", **metrics}],
                ["method", "n_samples", "auroc", "ece", "brier", "overconfidence"],
                self.console,
            )

        self.timing.stop("mid_execution")
        return {"results": results, "metrics": metrics, "timing": self.timing, "stats": self.progress_stats}


async def run_experiment(config: ExperimentConfig, console: Console) -> dict[str, Any]:
    """Run experiment with configured agents.

    Agents run in parallel by default unless sequential_agents is True.
    Handles graceful shutdown on SIGINT/SIGTERM.

    Args:
        config: Experiment configuration.
        console: Rich console for output.

    Returns:
        Dictionary with results from each agent.
    """
    from agentic_uncertainty.scripts._shared import AGENT_RUNNERS

    # Track overall timing
    overall_timing = TimingStats()
    overall_timing.start("total")

    # Print configuration summary
    print_config_table(config, console)

    if not config.active_agents:
        console.print("[yellow]No agents configured to run[/yellow]")
        return {}

    # Create runners
    runners = []
    for agent_name in config.active_agents:
        if agent_name not in AGENT_RUNNERS:
            console.print(f"[red]Unknown agent: {agent_name}[/red]")
            continue
        runner_class = AGENT_RUNNERS[agent_name]
        runners.append((agent_name, runner_class(config, console)))

    if not runners:
        return {}

    # Run agents with progress tracking
    all_results = {}
    agent_timings = {}

    try:
        if config.sequential_agents or len(runners) == 1:
            # Sequential execution with simple progress
            console.print(f"\n[bold]Running {len(runners)} agent(s) sequentially...[/bold]")
            for agent_name, runner in runners:
                # Check for shutdown before starting each agent
                if is_shutdown_requested():
                    console.print(f"[yellow]Shutdown requested, skipping remaining agents.[/yellow]")
                    break
                console.print(f"\n{'─' * 60}")
                result = await runner.run()
                all_results[agent_name] = result
                if "timing" in result:
                    agent_timings[agent_name] = result["timing"]
        else:
            # Parallel execution with live progress display
            console.print(f"\n[bold]Running {len(runners)} agents in parallel...[/bold]")
            console.print(f"{'─' * 60}")

            # Set up progress tracking
            progress, task_ids = create_progress_display([name for name, _ in runners])
            stats_by_agent: dict[str, ProgressStats] = {}

            def update_progress(agent_name: str, stats: ProgressStats):
                """Callback to update progress display."""
                stats_by_agent[agent_name] = stats
                if agent_name in task_ids:
                    total = stats.total if stats.total > 0 else 1
                    completed = stats.completed + stats.cached
                    progress.update(task_ids[agent_name], completed=completed, total=total)

            # Set progress callbacks on runners
            for agent_name, runner in runners:
                runner.progress_callback = update_progress

            # Run with live progress display
            with progress:
                tasks = [runner.run() for _, runner in runners]
                # Use return_exceptions=True to allow partial results on cancellation
                results = await asyncio.gather(*tasks, return_exceptions=True)

            for (agent_name, _), result in zip(runners, results):
                if isinstance(result, Exception):
                    console.print(f"[red]Agent {agent_name} failed: {result}[/red]")
                    all_results[agent_name] = {"error": str(result)}
                else:
                    all_results[agent_name] = result
                    if isinstance(result, dict) and "timing" in result:
                        agent_timings[agent_name] = result["timing"]

    except asyncio.CancelledError:
        console.print("[yellow]Experiment cancelled.[/yellow]")
        raise
    finally:
        # Stop overall timing
        overall_timing.stop("total")

    # Build combined timing stats for display
    combined_timing = TimingStats()
    for agent_name, timing in agent_timings.items():
        # Use the agent name as the phase name (e.g., "Exploration", "Review")
        for phase, duration in timing.phases.items():
            combined_timing.phases[agent_name.capitalize()] = duration

    # Print timing summary (total is calculated automatically)
    print_timing_summary(combined_timing, console)

    if is_shutdown_requested():
        console.print(f"\n[yellow]Experiment interrupted (partial results may be available).[/yellow]")
    else:
        console.print(f"\n[bold green]Experiment complete![/bold green]")
    return all_results


def build_config_from_args(args, instance_ids: list[str]) -> ExperimentConfig:
    """Build ExperimentConfig from parsed CLI arguments.

    Args:
        args: Parsed argparse namespace.
        instance_ids: Pre-sampled instance IDs.

    Returns:
        ExperimentConfig instance.
    """
    # Build exploration config if requested
    exploration_config = None
    if "exploration" in args.agents:
        exploration_config = ExplorationConfig(
            methods=args.exploration_methods,
            step_limit=args.exploration_step_limit,
            timeout=args.exploration_timeout,
            step_timeout=args.exploration_step_timeout,
        )

    # Build review config if requested
    review_config = None
    if "review" in args.agents:
        review_config = ReviewConfig(
            methods=args.review_methods,
            step_limit=args.review_step_limit,
            cost_limit=args.review_cost_limit,
            timeout=args.review_timeout,
            step_timeout=args.review_step_timeout,
            traj_dir=args.traj_dir,
        )

    # Build checkpoint config if requested
    checkpoint_config = None
    if "checkpoint" in args.agents:
        checkpoint_config = CheckpointConfig(
            step_limit=args.checkpoint_step_limit,
            timeout=args.checkpoint_timeout,
            step_timeout=args.checkpoint_step_timeout,
            confidence_interval=args.checkpoint_confidence_interval,
        )

    # Build checkpoint_posthoc config if requested
    checkpoint_posthoc_config = None
    if "checkpoint_posthoc" in args.agents:
        # Use checkpoint-posthoc-specific traj_dir, or fall back to --traj-dir
        traj_dir = getattr(args, "checkpoint_posthoc_traj_dir", None) or args.traj_dir
        checkpoint_posthoc_config = CheckpointPosthocConfig(
            confidence_interval=args.checkpoint_posthoc_confidence_interval,
            traj_dir=traj_dir,
        )

    # Build mid_execution config if requested
    mid_execution_config = None
    if "mid_execution" in args.agents:
        # Use mid-execution-specific traj_dir, or fall back to --traj-dir
        traj_dir = getattr(args, "mid_execution_traj_dir", None) or args.traj_dir
        mid_execution_config = MidExecutionConfig(
            step_limit=args.mid_execution_step_limit,
            timeout=args.mid_execution_timeout,
            step_timeout=args.mid_execution_step_timeout,
            progress_fraction=args.progress_fraction,
            traj_dir=traj_dir,
        )

    return ExperimentConfig(
        instance_ids=instance_ids,
        model=args.model,
        model_class=getattr(args, "model_class", ""),
        environment_class=args.environment_class,
        parallel=args.parallel,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        exploration=exploration_config,
        review=review_config,
        checkpoint=checkpoint_config,
        checkpoint_posthoc=checkpoint_posthoc_config,
        mid_execution=mid_execution_config,
        sequential_agents=getattr(args, "sequential_agents", False),
        watch=getattr(args, "watch", False),
    )


def main():
    """CLI entry point with argument parsing."""
    # Set up signal handlers for graceful shutdown
    def handle_shutdown_signal(signum, frame):
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        
        if is_shutdown_requested():
            # Second signal - force exit
            console.print(f"\n[red]Received {sig_name} again. Forcing exit...[/red]")
            cleanup_all_environments()
            sys.exit(1)
        
        request_shutdown()
        console.print(f"\n[yellow]Received {sig_name}. Shutting down gracefully...[/yellow]")
        console.print("[yellow]Press Ctrl+C again to force exit.[/yellow]")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    
    parser = argparse.ArgumentParser(
        description="Unified experiment runner for exploration and review agents. "
                    "Samples instances ONCE and uses them for all requested agents."
    )

    # Agent selection
    add_agent_selection_args(parser)

    # Instance selection
    parser.add_argument(
        "--instance-ids",
        type=Path,
        default=None,
        help="JSON file containing list of instance IDs to use",
    )
    add_sampling_args(parser)

    # Common arguments
    add_output_args(parser)
    add_parallel_args(parser)
    add_environment_args(parser)
    add_cache_args(parser)

    # Ground truth
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to eval_results.json with ground truth outcomes",
    )

    # Agent-specific arguments
    add_exploration_args(parser)
    add_review_args(parser)
    add_checkpoint_args(parser)
    add_checkpoint_posthoc_args(parser)
    add_mid_execution_args(parser)

    # Execution options
    parser.add_argument(
        "--sequential-agents",
        action="store_true",
        help="Run agents sequentially instead of in parallel (default: parallel)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging with debug information",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stream agent steps live to console (shows commands and outputs in real-time)",
    )

    args = parser.parse_args()

    # Configure logging (suppress noisy warnings, enable better error reporting)
    configure_logging(verbose=args.verbose)

    # Validate arguments
    if "review" in args.agents and not args.traj_dir:
        parser.error("--traj-dir is required when --agents includes 'review'")
    if "checkpoint_posthoc" in args.agents:
        traj_dir = getattr(args, "checkpoint_posthoc_traj_dir", None) or args.traj_dir
        if not traj_dir:
            parser.error("--checkpoint-posthoc-traj-dir or --traj-dir is required when --agents includes 'checkpoint_posthoc'")
    if "mid_execution" in args.agents:
        traj_dir = getattr(args, "mid_execution_traj_dir", None) or args.traj_dir
        if not traj_dir:
            parser.error("--mid-execution-traj-dir or --traj-dir is required when --agents includes 'mid_execution'")

    # Create timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("results") / f"experiment_{timestamp}"

    # Load and sample instances ONCE
    console.print("[bold]Sampling instances...[/bold]")
    instance_ids = load_and_sample_instances(
        instance_ids_file=args.instance_ids,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    # Build configuration
    config = build_config_from_args(args, instance_ids)

    # Run the experiment with cleanup on exit
    try:
        asyncio.run(run_experiment(config, console))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    finally:
        # Ensure all environments are cleaned up
        n_envs = get_active_environment_count()
        if n_envs > 0:
            console.print(f"[yellow]Cleaning up {n_envs} active environment(s)...[/yellow]")
            cleaned = cleanup_all_environments()
            console.print(f"[green]Cleaned up {cleaned} environment(s).[/green]")


if __name__ == "__main__":
    main()
