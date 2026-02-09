"""Agent runner protocol and base implementation.

Provides a common interface for experiment runners (exploration, review)
with shared functionality for caching, metrics, and result saving.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.console import Console

from .cache import ResultCache
from .config import ExperimentConfig
from .metrics import compute_standard_metrics
from .tables import print_metrics_table

if TYPE_CHECKING:
    from agentic_uncertainty.data import Task


# Registry for agent runners
AGENT_RUNNERS: dict[str, type["BaseAgentRunner"]] = {}


def register_agent(name: str):
    """Decorator to register an agent runner class."""
    def decorator(cls: type[BaseAgentRunner]) -> type[BaseAgentRunner]:
        AGENT_RUNNERS[name] = cls
        return cls
    return decorator


@runtime_checkable
class AgentRunner(Protocol):
    """Protocol defining the interface for agent runners."""

    name: str

    async def run(
        self,
        config: ExperimentConfig,
        console: Console,
    ) -> dict[str, Any]:
        """Run the agent on configured instances.

        Args:
            config: Experiment configuration.
            console: Rich console for output.

        Returns:
            Dictionary with results and metrics.
        """
        ...


class BaseAgentRunner(ABC):
    """Base class for agent runners with shared functionality.

    Provides common methods for cache management, metrics computation,
    and result serialization.
    """

    name: str = ""
    experiment_type: str = ""

    def __init__(self, config: ExperimentConfig, console: Console):
        """Initialize the runner.

        Args:
            config: Experiment configuration.
            console: Rich console for output.
        """
        self.config = config
        self.console = console
        self._cache: ResultCache | None = None

    @abstractmethod
    async def run(self) -> dict[str, Any]:
        """Run the agent. Subclasses must implement this."""
        ...

    def _init_cache(self, cache_subdir: str | None = None) -> ResultCache | None:
        """Initialize cache if cache_dir is configured.

        Args:
            cache_subdir: Optional subdirectory within cache_dir.

        Returns:
            ResultCache instance or None if caching disabled.
        """
        if not self.config.cache_dir:
            return None

        cache_path = self.config.cache_dir
        if cache_subdir:
            cache_path = cache_path / cache_subdir
            cache_path.mkdir(parents=True, exist_ok=True)

        self._cache = ResultCache(
            cache_path,
            model_name=self.config.model,
            console=self.console,
        )
        self.console.print(f"[bold]Cache:[/bold] {cache_path}")
        return self._cache

    def _get_output_dir(self, subdir: str | None = None) -> Path:
        """Get output directory, creating if needed.

        Args:
            subdir: Optional subdirectory within output_dir.

        Returns:
            Path to output directory.
        """
        output_dir = self.config.output_dir
        if subdir:
            output_dir = output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _compute_and_print_metrics(
        self,
        results: dict[str, dict],
        task_ids: list[str],
        ground_truth: dict[str, bool],
        title: str = "Metrics",
        extra_columns: list[str] | None = None,
    ) -> dict[str, dict]:
        """Compute metrics for results and print table.

        Args:
            results: Results dict keyed by method name.
            task_ids: List of task/instance IDs in order.
            ground_truth: Dict mapping instance_id to success bool.
            title: Title for the metrics table.
            extra_columns: Additional columns to include in table.

        Returns:
            Dict of metrics keyed by method name.
        """
        metrics_by_method = {}

        for method, data in results.items():
            preds = data.get("predictions", [])
            matched_preds = []
            matched_labels = []

            for i, iid in enumerate(task_ids):
                if iid in ground_truth and i < len(preds) and preds[i] is not None:
                    matched_preds.append(preds[i])
                    matched_labels.append(ground_truth[iid])

            if matched_preds:
                metrics = compute_standard_metrics(matched_preds, matched_labels)
                metrics_by_method[method] = metrics
                results[method]["metrics"] = metrics

        if metrics_by_method:
            table_data = [{"method": m, **metrics} for m, metrics in metrics_by_method.items()]
            columns = ["method", "auroc", "ece", "brier", "overconfidence"]
            if extra_columns:
                columns.extend(extra_columns)
            print_metrics_table(table_data, columns=columns, console=self.console, title=title)

        return metrics_by_method

    def _save_results(
        self,
        output_dir: Path,
        results: dict[str, Any],
        metrics: dict[str, dict] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save experiment results to JSON file.

        Args:
            output_dir: Directory to save results.
            results: Results dictionary.
            metrics: Optional metrics dictionary.
            extra_metadata: Additional metadata to include.

        Returns:
            Path to saved results file.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "results.json"

        output_data = {
            "experiment_type": self.experiment_type,
            "model": self.config.model,
            "timestamp": datetime.now().isoformat(),
            "environment_class": self.config.environment_class,
            "results": results,
        }

        if metrics:
            output_data["metrics"] = metrics

        if extra_metadata:
            output_data.update(extra_metadata)

        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        self.console.print(f"[green]{self.experiment_type.capitalize()} results saved to:[/green] {results_file}")
        return results_file

    def _merge_cached_and_new_results(
        self,
        tasks: list[Task],
        estimators: dict,
        cached_results: dict[str, dict[str, dict]],
        new_results: dict[str, dict],
        tasks_to_run: list[Task],
    ) -> dict[str, dict]:
        """Merge cached results with newly computed results.

        Args:
            tasks: All tasks in order.
            estimators: Dict of estimator instances keyed by method.
            cached_results: Cached results keyed by method, then instance_id.
            new_results: New results keyed by method with lists.
            tasks_to_run: Tasks that were actually run (subset of tasks).

        Returns:
            Merged results dictionary.
        """
        task_id_to_idx = {t.instance_id: i for i, t in enumerate(tasks)}
        run_task_ids = [t.instance_id for t in tasks_to_run] if tasks_to_run else []

        results = {}
        for method in estimators:
            results[method] = {
                "predictions": [None] * len(tasks),
                "raw_responses": [None] * len(tasks),
                "metadata": [None] * len(tasks),
            }

            # Fill cached results
            if method in cached_results:
                for iid, data in cached_results[method].items():
                    if iid in task_id_to_idx:
                        idx = task_id_to_idx[iid]
                        results[method]["predictions"][idx] = data.get("prediction")
                        results[method]["raw_responses"][idx] = data.get("raw_response")
                        results[method]["metadata"][idx] = data.get("metadata")

            # Fill new results
            if method in new_results and tasks_to_run:
                for run_idx, iid in enumerate(run_task_ids):
                    if iid in task_id_to_idx:
                        idx = task_id_to_idx[iid]
                        if run_idx < len(new_results[method]["predictions"]):
                            pred = new_results[method]["predictions"][run_idx]
                            if pred is not None:
                                results[method]["predictions"][idx] = pred
                                results[method]["raw_responses"][idx] = new_results[method]["raw_responses"][run_idx]
                                results[method]["metadata"][idx] = new_results[method]["metadata"][run_idx]

        return results

    def _load_cached_results(
        self,
        methods: list[str],
        task_ids: set[str],
    ) -> dict[str, dict[str, dict]]:
        """Load cached results for given methods and task IDs.

        Args:
            methods: List of method names.
            task_ids: Set of task IDs to look for.

        Returns:
            Dict keyed by method, then instance_id, with cached data.
        """
        if not self._cache:
            return {}

        cached_results = {}
        for method in methods:
            cached_ids = self._cache.get_cached_instance_ids(method)
            cached_results[method] = {}
            for iid in cached_ids & task_ids:
                cached_data = self._cache.get(method, iid)
                if cached_data:
                    cached_results[method][iid] = cached_data

        return cached_results

    def _find_tasks_needing_run(
        self,
        tasks: list[Task],
        methods: list[str],
        cached_results: dict[str, dict[str, dict]],
    ) -> list[Task]:
        """Find tasks that need to be run (not fully cached).

        Args:
            tasks: All tasks.
            methods: Methods to check.
            cached_results: Already loaded cached results.

        Returns:
            List of tasks that need at least one method run.
        """
        tasks_to_run = []
        for task in tasks:
            needs_run = False
            for method in methods:
                if task.instance_id not in cached_results.get(method, {}):
                    needs_run = True
                    break
            if needs_run:
                tasks_to_run.append(task)
        return tasks_to_run
