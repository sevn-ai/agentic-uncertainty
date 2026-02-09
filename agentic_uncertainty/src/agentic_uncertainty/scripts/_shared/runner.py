"""Shared experiment runner utilities."""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, TypeVar

import numpy as np
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from agentic_uncertainty.elicitation import UncertaintyEstimator
from agentic_uncertainty.scripts._shared.shutdown import is_shutdown_requested

if TYPE_CHECKING:
    from agentic_uncertainty.scripts._shared.cache import ResultCache

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# Type alias for the on_result callback
OnResultCallback = Callable[[str, Any, float | None, str, dict | None], None]

# Type alias for progress callback: (completed, total) -> None
ProgressCallback = Callable[[int, int], None]

# Type alias for step callback: (step_num, output_dict) -> None
StepCallback = Callable[[int, dict], None]

# Type alias for step callback factory: (task) -> StepCallback | None
StepCallbackFactory = Callable[[Any], StepCallback | None]


async def run_estimators_on_tasks(
    tasks: list,
    estimators: dict[str, UncertaintyEstimator],
    console: Console,
    progress_desc: str = "Processing tasks",
    max_concurrent_tasks: int = 1,
    on_result: OnResultCallback | None = None,
    cache: "ResultCache | None" = None,
    progress_callback: ProgressCallback | None = None,
    show_tqdm: bool = True,
    step_callback_factory: StepCallbackFactory | None = None,
) -> dict[str, dict]:
    """Run estimators on all tasks with error handling and optional parallelization.

    Args:
        tasks: List of Task objects to evaluate.
        estimators: Dict mapping method names to estimator instances.
        console: Rich Console for output.
        progress_desc: Description for progress bar.
        max_concurrent_tasks: Number of tasks to process in parallel (default: 1).
        on_result: Optional callback called after each result is obtained.
            Signature: (method_name, task, prediction, raw_response, metadata) -> None
            This enables incremental caching - results are saved as they complete.
        cache: Optional ResultCache for checkpointing during agent runs.
        progress_callback: Optional callback for external progress tracking.
            Signature: (completed, total) -> None
            If provided, updates are sent here instead of (or in addition to) tqdm.
        show_tqdm: Whether to show tqdm progress bar. Set False when using external progress.
        step_callback_factory: Optional factory to create step callbacks for live streaming.
            Signature: (task) -> StepCallback | None
            If provided, creates a callback for each task to stream agent steps.

    Returns:
        Dictionary with predictions, raw_responses, and metadata for each method.
    """
    # Initialize results structure with None placeholders for correct ordering
    results = {
        method: {
            "predictions": [None] * len(tasks),
            "raw_responses": [None] * len(tasks),
            "metadata": [None] * len(tasks),
        }
        for method in estimators
    }

    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    completed_count = 0
    total_count = len(tasks)
    
    # Only create tqdm if requested
    pbar = tqdm(total=total_count, desc=progress_desc, disable=not show_tqdm)

    async def process_task(task_idx: int, task):
        """Process a single task with all estimators."""
        nonlocal completed_count
        async with semaphore:
            # Check for shutdown before processing
            if is_shutdown_requested():
                logger.info(f"Skipping task {task.instance_id} due to shutdown request")
                for method_name in estimators:
                    results[method_name]["metadata"][task_idx] = {"skipped": "shutdown_requested"}
                completed_count += 1
                pbar.update(1)
                if progress_callback:
                    progress_callback(completed_count, total_count)
                return

            for method_name, estimator in estimators.items():
                # Check for shutdown between methods
                if is_shutdown_requested():
                    logger.info(f"Stopping task {task.instance_id} processing due to shutdown")
                    break

                try:
                    # Create step callback for this task if factory provided
                    step_callback = step_callback_factory(task) if step_callback_factory else None
                    
                    # Pass cache and step_callback if the estimator supports them
                    try:
                        result = await estimator.estimate(task, cache=cache, step_callback=step_callback)
                    except TypeError:
                        # Fallback for estimators that don't accept extra parameters
                        try:
                            result = await estimator.estimate(task, cache=cache)
                        except TypeError:
                            result = await estimator.estimate(task)
                    if result.probability is None:
                        logger.warning(
                            "Task %s/%s: No valid confidence returned (will be filtered)",
                            task.instance_id,
                            method_name,
                        )
                        # Store None - will be filtered during analysis
                        results[method_name]["predictions"][task_idx] = None
                    else:
                        results[method_name]["predictions"][task_idx] = result.probability
                    results[method_name]["raw_responses"][task_idx] = result.raw_response
                    results[method_name]["metadata"][task_idx] = result.metadata

                    # Call incremental callback if provided
                    if on_result:
                        on_result(
                            method_name,
                            task,
                            result.probability,
                            result.raw_response,
                            result.metadata,
                        )
                except Exception as e:
                    # Log error with context
                    logger.error(
                        "Task %s/%s failed: %s",
                        task.instance_id,
                        method_name,
                        str(e),
                        exc_info=logger.isEnabledFor(logging.DEBUG),
                    )
                    # Store None instead of 0.5 - will be filtered during analysis
                    results[method_name]["predictions"][task_idx] = None
                    results[method_name]["raw_responses"][task_idx] = ""
                    results[method_name]["metadata"][task_idx] = {"error": str(e)}

                    # Call incremental callback for errors too (to cache failed attempts)
                    if on_result:
                        on_result(method_name, task, None, "", {"error": str(e)})
            
            # Update progress
            completed_count += 1
            pbar.update(1)
            if progress_callback:
                progress_callback(completed_count, total_count)

    # Run all tasks concurrently (limited by semaphore)
    await asyncio.gather(*[process_task(i, task) for i, task in enumerate(tasks)])
    pbar.close()

    return results


def save_experiment_results(
    output_dir: Path,
    model_name: str,
    seed: int,
    tasks: list,
    estimators: dict[str, UncertaintyEstimator],
    results: dict[str, dict],
    metrics: dict | None = None,
    extra_metadata: dict | None = None,
    experiment_type: str | None = None,
) -> Path:
    """Save experiment results to JSON file.

    Args:
        output_dir: Directory to save results in.
        model_name: Name of the model used.
        seed: Random seed used for sampling.
        tasks: List of Task objects that were evaluated.
        estimators: Dict of estimator instances.
        results: Results dict from run_estimators_on_tasks().
        metrics: Optional metrics to include in output.
        extra_metadata: Optional additional metadata to include.
        experiment_type: Type of experiment (e.g., 'pre_execution', 'in_context').

    Returns:
        Path to the saved results file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "results.json"

    output_data = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "num_instances": len(tasks),
        "seed": seed,
        "instance_ids": [t.instance_id for t in tasks],
        "methods": list(estimators.keys()),
        "results": {
            method: {
                "predictions": data["predictions"],
                "raw_responses": data["raw_responses"],
                "metadata": data["metadata"],
            }
            for method, data in results.items()
        },
    }

    if experiment_type:
        output_data["experiment_type"] = experiment_type

    if metrics is not None:
        output_data["metrics"] = metrics

    if extra_metadata:
        output_data.update(extra_metadata)

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    return output_file


def print_summary_statistics(
    results: dict[str, dict],
    console: Console,
    title: str = "Summary Statistics",
) -> None:
    """Print summary statistics for experiment results.

    Args:
        results: Results dict from run_estimators_on_tasks().
        console: Rich Console for output.
        title: Title for the statistics table.
    """
    table = Table(show_header=True, title=title)
    table.add_column("Method")
    table.add_column("Mean Prediction")
    table.add_column("Std Dev")
    table.add_column("Valid/Total")

    for method, data in results.items():
        preds = data["predictions"]
        if preds:
            # Filter out None values
            valid_preds = [p for p in preds if p is not None]
            total_count = len(preds)
            valid_count = len(valid_preds)

            if valid_preds:
                mean_pred = np.mean(valid_preds)
                std_pred = np.std(valid_preds)
                table.add_row(
                    method,
                    f"{mean_pred:.3f}",
                    f"{std_pred:.3f}",
                    f"{valid_count}/{total_count}",
                )
            else:
                table.add_row(method, "N/A", "N/A", f"0/{total_count}")

            # Warn if any predictions were filtered
            if valid_count < total_count:
                filtered_count = total_count - valid_count
                console.print(
                    f"[yellow]Warning: {filtered_count} predictions for {method} "
                    f"were filtered out due to missing confidence values[/yellow]"
                )

    console.print(table)


def load_results(results_path: Path) -> dict[str, Any]:
    """Load experiment results from JSON file.

    Args:
        results_path: Path to results.json file.

    Returns:
        Dictionary with experiment results.
    """
    with open(results_path) as f:
        return json.load(f)


def merge_results(result_files: list[Path]) -> dict[str, Any]:
    """Merge multiple result files into a unified structure.

    Args:
        result_files: List of paths to results.json files.

    Returns:
        Dictionary with merged results indexed by model and experiment type.
    """
    merged = {
        "experiments": {},
        "timestamp": datetime.now().isoformat(),
    }

    for path in result_files:
        data = load_results(path)
        model = data.get("model", "unknown")
        exp_type = data.get("experiment_type", "unknown")

        if exp_type not in merged["experiments"]:
            merged["experiments"][exp_type] = {}

        merged["experiments"][exp_type][model] = {
            "path": str(path),
            "num_instances": data.get("num_instances"),
            "methods": data.get("methods", []),
            "timestamp": data.get("timestamp"),
        }

    return merged


def get_output_dir(
    output_dir: Path | None,
    experiment_name: str,
    base_dir: str = "results",
) -> Path:
    """Get or create a timestamped output directory.

    If output_dir is None, creates a timestamped directory under base_dir.

    Args:
        output_dir: User-specified output directory (or None).
        experiment_name: Name of the experiment (e.g., 'terminal', 'traces').
        base_dir: Base directory for results (default: 'results').

    Returns:
        Path to the output directory.

    Example:
        output_dir = get_output_dir(args.output_dir, "terminal")
        # Returns: Path("results/terminal_20240115_143052")
    """
    if output_dir is not None:
        return Path(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(base_dir) / f"{experiment_name}_{timestamp}"


async def run_async_with_progress(
    items: list[T],
    processor: Callable[[T], Awaitable[R]],
    max_concurrent: int = 1,
    desc: str = "Processing",
    console: Console | None = None,
) -> list[R | None]:
    """Run an async processor on items with semaphore and progress bar.

    Args:
        items: List of items to process.
        processor: Async function that processes a single item.
        max_concurrent: Maximum number of concurrent tasks.
        desc: Description for the progress bar.
        console: Optional Rich console for error output.

    Returns:
        List of results in the same order as input items.
        Failed items will have None as their result.

    Example:
        async def process_traj(traj):
            # ... processing logic ...
            return result

        results = await run_async_with_progress(
            trajectories,
            process_traj,
            max_concurrent=4,
            desc="Processing trajectories",
        )
    """
    results: list[R | None] = [None] * len(items)
    semaphore = asyncio.Semaphore(max_concurrent)
    pbar = tqdm(total=len(items), desc=desc)

    async def process_item(idx: int, item: T) -> None:
        async with semaphore:
            # Check for shutdown before processing
            if is_shutdown_requested():
                logger.debug(f"Skipping item {idx} due to shutdown request")
                pbar.update(1)
                return

            try:
                results[idx] = await processor(item)
            except Exception as e:
                if console:
                    console.print(f"[red]Error processing item {idx}: {e}[/red]")
                results[idx] = None
            finally:
                pbar.update(1)

    await asyncio.gather(*[process_item(i, item) for i, item in enumerate(items)])
    pbar.close()

    return results
