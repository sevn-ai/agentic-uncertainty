"""Unified evaluation script for cache results.

Aggregates results from cache/ directories, matches with ground truth,
computes metrics (AUROC, Brier, ECE, Overconfidence), and generates
tables + plots (ROC curves, calibration plots).

Usage:
    # Single model evaluation
    uv run evaluate-cache \\
        --cache-dir cache/gpt-5.2-codex \\
        --ground-truth data/trajectories/gpt-5.2-codex/evaluation/eval_results.json \\
        --output-dir results/gpt52 \\
        --plots --latex

    # Multi-model comparison
    uv run evaluate-cache \\
        --cache-dir cache/ \\
        --ground-truth data/trajectories/gpt-5.2-codex/evaluation/eval_results.json \\
        --output-dir results/comparison \\
        --plots --latex --compare-models
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from agentic_uncertainty.evaluation import (
    auprc_with_ci,
    auroc_with_ci,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from agentic_uncertainty.evaluation.plotting import (
    plot_calibration_comparison,
    plot_confidence_histograms_comparison,
    plot_roc_curves_comparison,
    plot_roc_curves_multi_model,
)

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class MetricsResult:
    """Computed metrics for a single method."""

    n_samples: int
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    auprc: float
    auprc_ci_lower: float
    auprc_ci_upper: float
    brier: float
    ece: float
    mce: float
    overconfidence: float
    mean_prediction: float
    base_rate: float


# Method display names for paper formatting
METHOD_DISPLAY_NAMES = {
    "exploration_direct": "Pre-Execution",
    "review_direct": "Post-Execution",
    "review_adversarial": "Adv.\\ Post-Exec.",
    "mid_execution_direct_25pct": "Checkpoint (25\\%)",
    "mid_execution_direct_50pct": "Checkpoint (50\\%)",
    "mid_execution_direct_75pct": "Checkpoint (75\\%)",
    # Ensembles
    "ensemble_average": "Average",
    "ensemble_min": "Conservative ($\\min$)",
    "ensemble_max": "Aggressive ($\\max$)",
}


def scan_cache_directory(cache_dir: Path) -> dict[str, list[dict]]:
    """Scan cache dir and return {method: [results]} grouped by method.

    Handles the nested directory structure:
    - cache/{model}/review/{target_model}/review_direct/*.json
    - cache/{model}/review/{target_model}/review_adversarial/*.json
    - cache/{model}/exploration/{target_model}/direct/*.json

    Args:
        cache_dir: Path to model's cache directory (e.g., cache/gpt-5.2-codex)

    Returns:
        Dict mapping method name to list of result dicts.
    """
    results_by_method: dict[str, list[dict]] = {}

    # Find all JSON files (excluding .traj.json and .checkpoint.json)
    for json_path in cache_dir.rglob("*.json"):
        # Skip trajectory and checkpoint files
        if json_path.name.endswith(".traj.json") or json_path.name.endswith(
            ".checkpoint.json"
        ):
            continue

        try:
            with open(json_path) as f:
                data = json.load(f)

            # Extract required fields
            instance_id = data.get("instance_id")
            prediction = data.get("prediction")
            method = data.get("method")

            if instance_id is None or prediction is None or method is None:
                logger.debug(f"Skipping {json_path}: missing required fields")
                continue

            # Normalize method names
            # "direct" from exploration becomes "exploration_direct"
            # "review_direct" and "review_adversarial" stay as-is
            if method == "direct":
                # Check if this is from exploration dir
                if "exploration" in str(json_path):
                    method = "exploration_direct"

            if method not in results_by_method:
                results_by_method[method] = []

            results_by_method[method].append(
                {
                    "instance_id": instance_id,
                    "prediction": prediction,
                    "method": method,
                    "path": str(json_path),
                }
            )

        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Error reading {json_path}: {e}")
            continue

    return results_by_method


def scan_all_models(cache_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Scan cache directory for all models.

    Args:
        cache_dir: Path to cache directory (e.g., cache/)

    Returns:
        Dict: model -> method -> [results]
    """
    model_results: dict[str, dict[str, list[dict]]] = {}

    # Check if cache_dir is a single model dir or contains multiple models
    # Single model: cache/gpt-5.2-codex (has review/, exploration/ subdirs)
    # Multi model: cache/ (has gpt-5.2-codex/, gemini-3-pro/ subdirs)

    subdirs = [d for d in cache_dir.iterdir() if d.is_dir()]

    # Check if this looks like a model directory (has review/exploration)
    has_method_dirs = any(
        (d.name in ("review", "exploration", "review_direct", "review_final"))
        for d in subdirs
    )

    if has_method_dirs:
        # Single model dir
        model_name = cache_dir.name
        model_results[model_name] = scan_cache_directory(cache_dir)
    else:
        # Multi-model dir
        for model_dir in subdirs:
            if model_dir.name.startswith(".") or model_dir.name == "deprecated":
                continue
            model_name = model_dir.name
            results = scan_cache_directory(model_dir)
            if results:
                model_results[model_name] = results

    return model_results


def find_common_instances(
    model_results: dict[str, dict[str, list[dict]]],
    ground_truth: dict[str, bool] | dict[str, dict[str, bool]],
    per_model_ground_truth: bool = False,
) -> set[str]:
    """Find instances that exist across ALL models AND have ground truth.

    This ensures apples-to-apples comparison:
    1. Instance must have ground truth label
    2. Instance must have prediction in EVERY model being compared
    3. Instance must have prediction in EVERY method being compared

    Args:
        model_results: model -> method -> [results]
        ground_truth: instance_id -> success/failure (single ground truth)
                     OR model -> instance_id -> success/failure (per-model ground truth)
        per_model_ground_truth: If True, ground_truth is per-model dict.

    Returns:
        Set of instance_ids to use for evaluation.
    """
    if not model_results:
        return set()

    # Start with ground truth instances
    if per_model_ground_truth:
        # Intersect ground truth instances across all models
        common = None
        for model in model_results:
            if model in ground_truth:
                model_gt_instances = set(ground_truth[model].keys())
                if common is None:
                    common = model_gt_instances
                else:
                    common &= model_gt_instances
        if common is None:
            return set()
    else:
        common = set(ground_truth.keys())

    # Collect all methods across all models
    all_methods = set()
    for methods in model_results.values():
        all_methods.update(methods.keys())

    # For each model+method combo, intersect with instances that have predictions
    for model, methods in model_results.items():
        for method in all_methods:
            if method in methods:
                method_instances = {r["instance_id"] for r in methods[method]}
                common &= method_instances

    return common


def filter_to_common_instances(
    model_results: dict[str, dict[str, list[dict]]],
    common_instances: set[str],
    ground_truth: dict[str, bool] | dict[str, dict[str, bool]],
    per_model_ground_truth: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Filter all results to only include common instances.

    Args:
        model_results: model -> method -> [results]
        common_instances: Set of instance_ids to keep.
        ground_truth: instance_id -> success/failure (single ground truth)
                     OR model -> instance_id -> success/failure (per-model ground truth)
        per_model_ground_truth: If True, ground_truth is per-model dict.

    Returns:
        model -> method -> {
            'predictions': np.ndarray,
            'labels': np.ndarray,
            'instance_ids': list[str],
        }
    """
    filtered: dict[str, dict[str, dict[str, Any]]] = {}

    for model, methods in model_results.items():
        filtered[model] = {}

        # Get ground truth for this model
        if per_model_ground_truth:
            model_gt = ground_truth.get(model, {})
        else:
            model_gt = ground_truth

        for method, results in methods.items():
            # Filter to common instances
            filtered_results = [
                r for r in results if r["instance_id"] in common_instances
            ]

            # Deduplicate: if multiple predictions for same instance, average them
            instance_preds: dict[str, list[float]] = {}
            for r in filtered_results:
                iid = r["instance_id"]
                if iid not in instance_preds:
                    instance_preds[iid] = []
                instance_preds[iid].append(r["prediction"])

            # Sort by instance_id for consistency
            sorted_ids = sorted(instance_preds.keys())

            instance_ids = sorted_ids
            predictions = np.array([np.mean(instance_preds[iid]) for iid in sorted_ids])
            labels = np.array([model_gt[iid] for iid in sorted_ids], dtype=float)

            filtered[model][method] = {
                "predictions": predictions,
                "labels": labels,
                "instance_ids": instance_ids,
            }

    return filtered


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> MetricsResult:
    """Compute all evaluation metrics.

    Args:
        predictions: Predicted probabilities.
        labels: Binary labels.

    Returns:
        MetricsResult with all metrics.
    """
    n_samples = len(predictions)
    base_rate = float(np.mean(labels))
    mean_pred = float(np.mean(predictions))

    # Discrimination
    auroc_result = auroc_with_ci(predictions, labels)
    auprc_result = auprc_with_ci(predictions, labels)

    # Calibration
    brier = brier_score(predictions, labels)
    ece = expected_calibration_error(predictions, labels)
    mce = maximum_calibration_error(predictions, labels)

    # Overconfidence: mean prediction - base rate
    overconfidence = mean_pred - base_rate

    return MetricsResult(
        n_samples=n_samples,
        auroc=auroc_result.auroc,
        auroc_ci_lower=auroc_result.ci_lower,
        auroc_ci_upper=auroc_result.ci_upper,
        auprc=auprc_result.auprc,
        auprc_ci_lower=auprc_result.ci_lower,
        auprc_ci_upper=auprc_result.ci_upper,
        brier=brier,
        ece=ece,
        mce=mce,
        overconfidence=overconfidence,
        mean_prediction=mean_pred,
        base_rate=base_rate,
    )


def compute_all_metrics(
    filtered_data: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, MetricsResult]]:
    """Compute metrics for all models and methods.

    Args:
        filtered_data: model -> method -> {'predictions', 'labels', ...}

    Returns:
        model -> method -> MetricsResult
    """
    metrics: dict[str, dict[str, MetricsResult]] = {}

    for model, methods in filtered_data.items():
        metrics[model] = {}
        for method, data in methods.items():
            metrics[model][method] = compute_metrics(
                data["predictions"], data["labels"]
            )

    return metrics


def compute_ensembles(
    filtered_data: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute ensemble predictions by combining exploration and review.

    For each model, creates three ensemble methods:
    - ensemble_average: mean of exploration and review predictions
    - ensemble_min: minimum (conservative) of exploration and review
    - ensemble_max: maximum (aggressive) of exploration and review

    Args:
        filtered_data: model -> method -> {'predictions', 'labels', 'instance_ids'}

    Returns:
        Updated filtered_data with ensemble methods added.
    """
    for model, methods in filtered_data.items():
        # Check which base methods are available
        exploration = methods.get("exploration_direct")
        review = methods.get("review_direct")

        if exploration is None or review is None:
            logger.info(f"Skipping ensembles for {model}: missing exploration or review")
            continue

        # Verify instance alignment (sorted comparison since order shouldn't matter)
        if sorted(exploration["instance_ids"]) != sorted(review["instance_ids"]):
            logger.warning(f"Instance mismatch for {model}, skipping ensembles")
            continue

        # Re-sort both to ensure alignment
        exp_order = {iid: i for i, iid in enumerate(exploration["instance_ids"])}
        rev_order = [exp_order[iid] for iid in review["instance_ids"]]
        if rev_order != list(range(len(rev_order))):
            # Review is in different order, reorder it
            reorder = np.argsort([exp_order[iid] for iid in review["instance_ids"]])
            review["predictions"] = review["predictions"][reorder]
            review["labels"] = review["labels"][reorder]
            review["instance_ids"] = [review["instance_ids"][i] for i in reorder]

        exp_preds = exploration["predictions"]
        rev_preds = review["predictions"]
        labels = exploration["labels"]
        instance_ids = exploration["instance_ids"]

        # Compute ensemble predictions
        avg_preds = (exp_preds + rev_preds) / 2
        min_preds = np.minimum(exp_preds, rev_preds)
        max_preds = np.maximum(exp_preds, rev_preds)

        methods["ensemble_average"] = {
            "predictions": avg_preds,
            "labels": labels,
            "instance_ids": instance_ids,
        }
        methods["ensemble_min"] = {
            "predictions": min_preds,
            "labels": labels,
            "instance_ids": instance_ids,
        }
        methods["ensemble_max"] = {
            "predictions": max_preds,
            "labels": labels,
            "instance_ids": instance_ids,
        }

        logger.info(f"Added ensembles for {model}")

    return filtered_data


def print_filtering_summary(
    model_results: dict[str, dict[str, list[dict]]],
    ground_truth: dict[str, bool] | dict[str, dict[str, bool]],
    common_instances: set[str],
    per_model_ground_truth: bool = False,
) -> None:
    """Print summary of instance filtering."""
    console.print("\n[bold]Instance Filtering Summary[/bold]")
    console.print("=" * 40)

    if per_model_ground_truth:
        console.print("Ground truth instances (per model):")
        for model, gt in sorted(ground_truth.items()):
            base_rate = sum(1 for v in gt.values() if v) / len(gt) if gt else 0
            console.print(f"  {model}: {len(gt)} instances, base rate {base_rate:.1%}")
    else:
        console.print(f"Ground truth instances:           {len(ground_truth)}")

    for model, methods in sorted(model_results.items()):
        for method, results in sorted(methods.items()):
            n_instances = len({r["instance_id"] for r in results})
            console.print(f"{model}/{method}:".ljust(35) + f"{n_instances}")

    console.print("=" * 40)
    console.print(
        f"[bold]Common instances (used):[/bold]           {len(common_instances)}"
    )


def print_results_table(
    metrics: dict[str, dict[str, MetricsResult]],
    n_instances: int,
    base_rate: float | None = None,
) -> None:
    """Print results table to console."""
    # Check if all models have the same base rate
    model_base_rates = {}
    for model, methods in metrics.items():
        first_method = next(iter(methods))
        model_base_rates[model] = methods[first_method].base_rate

    if base_rate is not None and len(set(model_base_rates.values())) == 1:
        title = f"Results (N={n_instances}, base rate={base_rate:.1%})"
    else:
        # Per-model base rates
        rates_str = ", ".join(f"{m}: {r:.0%}" for m, r in model_base_rates.items())
        title = f"Results (N={n_instances}, base rates: {rates_str})"
    table = Table(title=title)

    table.add_column("Model", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("AUROC", justify="center")
    table.add_column("Mean", justify="center")
    table.add_column("Overconf", justify="center")
    table.add_column("Brier", justify="center")
    table.add_column("ECE", justify="center")

    # Sort methods in display order
    method_order = [
        "exploration_direct",
        "review_direct",
        "review_adversarial",
        "mid_execution_direct_25pct",
        "mid_execution_direct_50pct",
        "mid_execution_direct_75pct",
        "ensemble_average",
        "ensemble_min",
        "ensemble_max",
    ]

    for model in sorted(metrics.keys()):
        methods_dict = metrics[model]
        # Sort methods by predefined order
        sorted_methods = sorted(
            methods_dict.keys(),
            key=lambda m: method_order.index(m) if m in method_order else 999,
        )

        for method in sorted_methods:
            m = methods_dict[method]
            # Use display name if available
            display_name = METHOD_DISPLAY_NAMES.get(method, method).replace("\\", "")
            auroc_str = f"{m.auroc:.3f}"
            table.add_row(
                model,
                display_name,
                auroc_str,
                f"{m.mean_prediction:.2f}",
                f"{m.overconfidence:+.2f}",
                f"{m.brier:.3f}",
                f"{m.ece:.3f}",
            )

    console.print(table)


def generate_latex_table(
    metrics: dict[str, dict[str, MetricsResult]],
    output_path: Path,
    n_instances: int,
    base_rate: float,
) -> str:
    """Generate publication-ready LaTeX table matching paper format.

    Generates table with columns: Method, AUROC↑, Overconf., ECE↓, Brier↓
    Separates base methods from ensemble methods with \\midrule.

    Args:
        metrics: model -> method -> MetricsResult
        output_path: Path to save .tex file.
        n_instances: Number of instances evaluated.
        base_rate: Base rate of success.

    Returns:
        LaTeX table string.
    """
    # For single-model case, generate one table
    # For multi-model, generate separate tables per model
    models = list(metrics.keys())

    all_tables = []

    for model in models:
        model_metrics = metrics[model]

        # Separate base methods from ensembles
        base_methods = []
        ensemble_methods = []
        for method in model_metrics:
            if method.startswith("ensemble_"):
                ensemble_methods.append(method)
            else:
                base_methods.append(method)

        # Sort methods in desired order
        method_order = [
            "exploration_direct",
            "review_direct",
            "review_adversarial",
            "mid_execution_direct_25pct",
            "mid_execution_direct_50pct",
            "mid_execution_direct_75pct",
        ]
        base_methods = [m for m in method_order if m in base_methods]
        # Add any methods not in the predefined order
        base_methods += [m for m in model_metrics if m not in base_methods and not m.startswith("ensemble_")]

        ensemble_order = ["ensemble_average", "ensemble_min", "ensemble_max"]
        ensemble_methods = [m for m in ensemble_order if m in ensemble_methods]

        # Find best values for highlighting
        all_methods = base_methods + ensemble_methods
        if all_methods:
            best_auroc = max(model_metrics[m].auroc for m in all_methods)
            best_brier = min(model_metrics[m].brier for m in all_methods)
            best_ece = min(model_metrics[m].ece for m in all_methods)
            best_overconf = min(abs(model_metrics[m].overconfidence) for m in all_methods)

        model_label = model.replace("-", " ").replace("_", " ").title()

        lines = [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{Uncertainty estimation results for {model_label} (N={n_instances}, base rate={base_rate:.1%}).}}",
            f"\\label{{tab:{model.replace('-', '_')}_evaluation}}",
            "\\small",
            "\\setlength{\\tabcolsep}{3pt}",
            "\\begin{tabular}{@{}lcccc@{}}",
            "\\toprule",
            "Method & AUROC$\\uparrow$ & Overconf. & ECE$\\downarrow$ & Brier$\\downarrow$ \\\\",
            "\\midrule",
        ]

        def format_row(method: str, m: MetricsResult, highlight_row: str | None = None) -> str:
            """Format a single table row."""
            method_name = METHOD_DISPLAY_NAMES.get(method, method.replace("_", "\\_"))

            # Format values with bold for best
            auroc_val = f"{m.auroc:.3f}"
            if abs(m.auroc - best_auroc) < 0.001:
                auroc_val = f"\\textbf{{{auroc_val}}}"

            overconf_val = f"{m.overconfidence:+.2f}"
            if abs(abs(m.overconfidence) - best_overconf) < 0.001:
                overconf_val = f"\\textbf{{{overconf_val}}}"

            ece_val = f"{m.ece:.3f}"
            if abs(m.ece - best_ece) < 0.001:
                ece_val = f"\\textbf{{{ece_val}}}"

            brier_val = f"{m.brier:.3f}"
            if abs(m.brier - best_brier) < 0.001:
                brier_val = f"\\textbf{{{brier_val}}}"

            row = f"{method_name} & {auroc_val} & {overconf_val} & {ece_val} & {brier_val} \\\\"

            if highlight_row:
                row = f"\\rowcolor{{{highlight_row}}} " + row

            return row

        # Add base methods
        for method in base_methods:
            m = model_metrics[method]
            # Highlight checkpoint methods (worst calibration)
            highlight = None
            if "mid_execution" in method or "checkpoint" in method.lower():
                highlight = "FailColor!15"
            lines.append(format_row(method, m, highlight))

        # Add separator and ensemble methods
        if ensemble_methods:
            lines.append("\\midrule")
            for method in ensemble_methods:
                m = model_metrics[method]
                # Highlight conservative (best calibration)
                highlight = None
                if method == "ensemble_min":
                    highlight = "ExpColor!15"
                lines.append(format_row(method, m, highlight))

        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )

        all_tables.append("\n".join(lines))

    table_str = "\n\n".join(all_tables)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def generate_multi_model_latex_table(
    metrics: dict[str, dict[str, MetricsResult]],
    output_path: Path,
    model_display_names: dict[str, str] | None = None,
    model_base_rates: dict[str, float] | None = None,
) -> str:
    """Generate multi-model comparison LaTeX table matching paper Table 1 format.

    Generates a wide table with columns for each model:
    Method | AUROC↑ Overconf. ECE↓ Brier↓ | ... (repeated for each model)

    Args:
        metrics: model -> method -> MetricsResult
        output_path: Path to save .tex file.
        model_display_names: Optional mapping from model keys to display names.
        model_base_rates: Optional mapping from model keys to base rates (for header).

    Returns:
        LaTeX table string.
    """
    models = list(metrics.keys())
    n_models = len(models)

    if n_models == 0:
        return ""

    # Default display names
    if model_display_names is None:
        model_display_names = {
            "gpt-5.2-codex": "GPT-5.2-Codex",
            "gemini-3-pro-preview": "Gemini-3-Pro-Preview",
            "claude-opus-4-5": "Claude-Opus-4.5",
        }

    # Get base rates from metrics if not provided
    if model_base_rates is None:
        model_base_rates = {}
        for model in models:
            first_method = next(iter(metrics[model]))
            model_base_rates[model] = metrics[model][first_method].base_rate

    # Method ordering
    method_order = [
        "exploration_direct",
        "review_direct",
        "review_adversarial",
    ]
    ensemble_order = ["ensemble_average", "ensemble_min", "ensemble_max"]

    # Collect methods present in all models
    all_model_methods = [set(metrics[m].keys()) for m in models]
    common_methods = set.intersection(*all_model_methods) if all_model_methods else set()

    base_methods = [m for m in method_order if m in common_methods]
    ensemble_methods = [m for m in ensemble_order if m in common_methods]
    all_methods = base_methods + ensemble_methods

    if not all_methods:
        logger.warning("No common methods found across models")
        return ""

    # Find best values per model for bolding
    best_values: dict[str, dict[str, float]] = {}
    for model in models:
        model_metrics = metrics[model]
        methods_to_check = [m for m in all_methods if m in model_metrics]
        if methods_to_check:
            best_values[model] = {
                "auroc": max(model_metrics[m].auroc for m in methods_to_check),
                "brier": min(model_metrics[m].brier for m in methods_to_check),
                "ece": min(model_metrics[m].ece for m in methods_to_check),
                "overconf": min(abs(model_metrics[m].overconfidence) for m in methods_to_check),
            }

    # Build column specification: ll + (cccc|) * n_models (last without |)
    col_spec = "@{}ll" + "cccc|" * (n_models - 1) + "cccc@{}"

    # Build header with model names and base rates
    model_headers = []
    for i, model in enumerate(models):
        display_name = model_display_names.get(model, model)
        base_rate_pct = int(model_base_rates.get(model, 0) * 100)
        sep = "|" if i < n_models - 1 else ""
        model_headers.append(f"\\multicolumn{{4}}{{c{sep}}}{{\\textbf{{{display_name}}} ({base_rate_pct}\\%)}}")

    # Build cmidrule commands
    cmidrules = []
    for i in range(n_models):
        start_col = 3 + i * 4
        end_col = start_col + 3
        trim = "lr" if i < n_models - 1 else "l"
        cmidrules.append(f"\\cmidrule({trim}){{{start_col}-{end_col}}}")

    # Build column header row
    col_headers = ["AUROC$\\uparrow$", "Overconf.", "ECE$\\downarrow$", "Brier$\\downarrow$"]
    header_row = "& Method & " + " & ".join(col_headers * n_models) + " \\\\"

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Unified discrimination and calibration results across judge models. "
        "Pre-execution achieves best discrimination (AUROC) despite having less information than post-execution. "
        "Adversarial post-execution achieves best calibration (ECE, Brier). "
        "Best values per model are \\textbf{bolded}.}",
        "\\label{tab:unified_results}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "& & " + " & ".join(model_headers) + " \\\\",
        " ".join(cmidrules),
        header_row,
        "\\midrule",
    ]

    def format_value(val: float, best_val: float, fmt: str, is_lower_better: bool = True) -> str:
        """Format a value, bolding if it matches the best."""
        formatted = fmt.format(val)
        if is_lower_better:
            is_best = abs(val - best_val) < 0.001
        else:
            is_best = abs(val - best_val) < 0.001
        if is_best:
            return f"\\textbf{{{formatted}}}"
        return formatted

    def format_row(method: str, row_label: str | None = None) -> str:
        """Format a single table row across all models."""
        method_name = METHOD_DISPLAY_NAMES.get(method, method.replace("_", "\\_"))

        cells = []
        for model in models:
            m = metrics[model].get(method)
            bv = best_values.get(model, {})

            if m is None:
                cells.extend(["--"] * 4)
            else:
                auroc_val = format_value(m.auroc, bv.get("auroc", 0), "{:.3f}", is_lower_better=False)

                # Special handling: compare absolute value for overconfidence
                if abs(abs(m.overconfidence) - bv.get("overconf", 0)) < 0.001:
                    overconf_val = f"\\textbf{{{m.overconfidence:+.2f}}}"
                else:
                    overconf_val = f"{m.overconfidence:+.2f}"

                ece_val = format_value(m.ece, bv.get("ece", 0), "{:.3f}")
                brier_val = format_value(m.brier, bv.get("brier", 0), "{:.3f}")

                cells.extend([auroc_val, overconf_val, ece_val, brier_val])

        row_prefix = row_label if row_label else ""
        return f"{row_prefix}& {method_name} & " + " & ".join(cells) + " \\\\"

    # Add base methods with "Single" label
    if base_methods:
        n_base = len(base_methods)
        single_label = f"\\multirow{{{n_base}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\scriptsize Single}}}}"
        for i, method in enumerate(base_methods):
            label = single_label if i == 0 else ""
            lines.append(format_row(method, label))

    # Add separator and ensemble methods
    if ensemble_methods:
        lines.append("\\midrule")
        n_ens = len(ensemble_methods)
        ensemble_label = f"\\multirow{{{n_ens}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\scriptsize Ensemble}}}}"
        for i, method in enumerate(ensemble_methods):
            label = ensemble_label if i == 0 else ""
            lines.append(format_row(method, label))

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])

    table_str = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def save_results_json(
    metrics: dict[str, dict[str, MetricsResult]],
    output_path: Path,
    cache_dir: Path,
    ground_truth_path: Path,
    filtering_stats: dict[str, Any],
    n_instances: int,
    base_rate: float,
) -> None:
    """Save results as JSON.

    Args:
        metrics: model -> method -> MetricsResult
        output_path: Path to save results.json.
        cache_dir: Cache directory used.
        ground_truth_path: Ground truth file path.
        filtering_stats: Instance filtering statistics.
        n_instances: Number of instances evaluated.
        base_rate: Base rate of success.
    """
    output = {
        "metadata": {
            "cache_dir": str(cache_dir),
            "ground_truth": str(ground_truth_path),
            "timestamp": datetime.now().isoformat(),
        },
        "filtering": filtering_stats,
        "summary": {
            "n_instances": n_instances,
            "base_rate": base_rate,
        },
        "metrics": {
            model: {method: asdict(m) for method, m in methods.items()}
            for model, methods in metrics.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def generate_plots(
    filtered_data: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    compare_models: bool = False,
) -> None:
    """Generate visualization plots.

    Args:
        filtered_data: model -> method -> {'predictions', 'labels', ...}
        output_dir: Directory to save plots.
        compare_models: If True, generate multi-model comparison plots.
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # If single model, flatten to method level
    models = list(filtered_data.keys())

    if len(models) == 1 or not compare_models:
        # Single model or no comparison: combine all model/method pairs
        combined_results: dict[str, tuple] = {}

        for model, methods in filtered_data.items():
            for method, data in methods.items():
                key = f"{model}/{method}" if len(models) > 1 else method
                combined_results[key] = (data["predictions"], data["labels"])

        # ROC curves
        plot_roc_curves_comparison(
            combined_results,
            title="ROC Curves",
            save_path=plots_dir / "roc_curves.pdf",
        )
        console.print(f"[green]Saved:[/green] {plots_dir / 'roc_curves.pdf'}")

        # Calibration curves
        plot_calibration_comparison(
            combined_results,
            title="Calibration Curves",
            save_path=plots_dir / "calibration_curves.pdf",
        )
        console.print(f"[green]Saved:[/green] {plots_dir / 'calibration_curves.pdf'}")

        # Confidence histograms
        plot_confidence_histograms_comparison(
            combined_results,
            save_path=plots_dir / "confidence_histograms.pdf",
        )
        console.print(
            f"[green]Saved:[/green] {plots_dir / 'confidence_histograms.pdf'}"
        )

    else:
        # Multi-model comparison
        model_results_for_plot: dict[str, dict[str, tuple]] = {}
        for model, methods in filtered_data.items():
            model_results_for_plot[model] = {}
            for method, data in methods.items():
                model_results_for_plot[model][method] = (
                    data["predictions"],
                    data["labels"],
                )

        # Multi-model ROC
        plot_roc_curves_multi_model(
            model_results_for_plot,
            save_path=plots_dir / "roc_curves_multi_model.pdf",
        )
        console.print(
            f"[green]Saved:[/green] {plots_dir / 'roc_curves_multi_model.pdf'}"
        )

        # Also generate per-model plots
        for model, methods in filtered_data.items():
            model_dir = plots_dir / model
            model_dir.mkdir(exist_ok=True)

            model_combined: dict[str, tuple] = {}
            for method, data in methods.items():
                model_combined[method] = (data["predictions"], data["labels"])

            plot_roc_curves_comparison(
                model_combined,
                title=f"ROC Curves - {model}",
                save_path=model_dir / "roc_curves.pdf",
            )
            plot_calibration_comparison(
                model_combined,
                title=f"Calibration Curves - {model}",
                save_path=model_dir / "calibration_curves.pdf",
            )
            plot_confidence_histograms_comparison(
                model_combined,
                save_path=model_dir / "confidence_histograms.pdf",
            )
            console.print(f"[green]Saved:[/green] {model_dir}/*.pdf")

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate uncertainty predictions from cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Cache directory (single model or parent of multiple models)",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Path to eval_results.json with ground truth labels (single file for all models)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=None,
        help="Directory containing per-model ground truth (e.g., data/trajectories/). "
             "Expects {dir}/{model}/evaluation/eval_results.json for each model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Output directory for results (default: results/evaluation)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX tables",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Generate multi-model comparison plots (when cache-dir contains multiple models)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--exclude-methods",
        nargs="+",
        default=[],
        help="Method name patterns to exclude (e.g., 'mid_execution' excludes all mid_execution_*)",
    )
    parser.add_argument(
        "--exclude-instances",
        type=Path,
        default=None,
        help="JSON file with list of instance IDs to exclude (ensures consistent instances across models)",
    )

    args = parser.parse_args()

    # Validate ground truth arguments
    if args.ground_truth is None and args.ground_truth_dir is None:
        console.print("[red]Error: Must specify either --ground-truth or --ground-truth-dir[/red]")
        return
    if args.ground_truth is not None and args.ground_truth_dir is not None:
        console.print("[red]Error: Cannot specify both --ground-truth and --ground-truth-dir[/red]")
        return

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Determine if using per-model ground truth
    per_model_ground_truth = args.ground_truth_dir is not None

    # Load excluded instances if provided
    excluded_instances: set[str] = set()
    if args.exclude_instances:
        console.print(f"[bold]Loading excluded instances:[/bold] {args.exclude_instances}")
        with open(args.exclude_instances) as f:
            excluded_list = json.load(f)
            excluded_instances = set(excluded_list)
        console.print(f"  Excluding {len(excluded_instances)} instances")

    # Scan cache first to know which models we have
    console.print(f"[bold]Scanning cache:[/bold] {args.cache_dir}")
    model_results = scan_all_models(args.cache_dir)

    if not model_results:
        console.print("[red]Error: No results found in cache directory[/red]")
        return

    # Now load ground truth
    ground_truth: dict = {}
    if per_model_ground_truth:
        console.print(f"[bold]Loading per-model ground truth from:[/bold] {args.ground_truth_dir}")
        models_without_gt = []
        for model in model_results:
            gt_path = args.ground_truth_dir / model / "evaluation" / "eval_results.json"
            if gt_path.exists():
                with open(gt_path) as f:
                    model_gt = json.load(f)
                # Apply exclusions
                if excluded_instances:
                    model_gt = {k: v for k, v in model_gt.items() if k not in excluded_instances}
                ground_truth[model] = model_gt
                base_rate = sum(1 for v in model_gt.values() if v) / len(model_gt) if model_gt else 0
                console.print(f"  {model}: {len(model_gt)} instances, base rate {base_rate:.1%}")
            else:
                console.print(f"[yellow]  Warning: No ground truth for {model}, excluding from analysis[/yellow]")
                models_without_gt.append(model)
        # Remove models without ground truth
        for model in models_without_gt:
            del model_results[model]
    else:
        console.print(f"[bold]Loading ground truth:[/bold] {args.ground_truth}")
        with open(args.ground_truth) as f:
            ground_truth = json.load(f)
        # Apply exclusions
        if excluded_instances:
            ground_truth = {k: v for k, v in ground_truth.items() if k not in excluded_instances}

    # Filter out excluded methods
    if args.exclude_methods:
        console.print(f"[yellow]Excluding methods matching:[/yellow] {args.exclude_methods}")
        for model in model_results:
            methods_to_remove = []
            for method in model_results[model]:
                if any(pattern in method for pattern in args.exclude_methods):
                    methods_to_remove.append(method)
            for method in methods_to_remove:
                del model_results[model][method]
                console.print(f"  Excluded: {model}/{method}")

    # Report what we found
    for model, methods in model_results.items():
        n_results = sum(len(results) for results in methods.values())
        console.print(f"  {model}: {n_results} results across {len(methods)} methods")

    # Find common instances
    common_instances = find_common_instances(model_results, ground_truth, per_model_ground_truth)

    if not common_instances:
        console.print("[red]Error: No common instances found[/red]")
        return

    # Print filtering summary
    print_filtering_summary(model_results, ground_truth, common_instances, per_model_ground_truth)

    # Build filtering stats for JSON output
    if per_model_ground_truth:
        gt_instances_count = {model: len(gt) for model, gt in ground_truth.items()}
    else:
        gt_instances_count = len(ground_truth)
    filtering_stats = {
        "ground_truth_instances": gt_instances_count,
        "per_model_ground_truth": per_model_ground_truth,
        "per_model_method": {},
        "common_instances": len(common_instances),
        "filtered_instance_ids": sorted(common_instances),
    }
    for model, methods in model_results.items():
        for method, results in methods.items():
            key = f"{model}/{method}"
            filtering_stats["per_model_method"][key] = len(
                {r["instance_id"] for r in results}
            )

    # Filter to common instances
    filtered_data = filter_to_common_instances(
        model_results, common_instances, ground_truth, per_model_ground_truth
    )

    # Compute ensemble predictions (average, min, max of exploration + review)
    console.print("\n[bold]Computing ensemble predictions...[/bold]")
    filtered_data = compute_ensembles(filtered_data)

    # Compute metrics
    console.print("[bold]Computing metrics...[/bold]")
    metrics = compute_all_metrics(filtered_data)

    # Get common stats
    n_instances = len(common_instances)
    # Get base rate from any method (all should be the same)
    first_model = next(iter(metrics))
    first_method = next(iter(metrics[first_model]))
    base_rate = metrics[first_model][first_method].base_rate

    # Print results
    print_results_table(metrics, n_instances, base_rate)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_path = args.output_dir / "results.json"
    gt_path = args.ground_truth if args.ground_truth else args.ground_truth_dir
    save_results_json(
        metrics,
        json_path,
        args.cache_dir,
        gt_path,
        filtering_stats,
        n_instances,
        base_rate,
    )
    console.print(f"\n[green]Saved:[/green] {json_path}")

    # Generate LaTeX table
    if args.latex:
        # Per-model tables
        latex_path = args.output_dir / "table.tex"
        generate_latex_table(metrics, latex_path, n_instances, base_rate)
        console.print(f"[green]Saved:[/green] {latex_path}")

        # Multi-model combined table (if multiple models or --compare-models)
        if len(metrics) > 1 or args.compare_models:
            multi_latex_path = args.output_dir / "table_unified.tex"
            generate_multi_model_latex_table(metrics, multi_latex_path)
            console.print(f"[green]Saved:[/green] {multi_latex_path}")

    # Generate plots
    if args.plots:
        console.print("\n[bold]Generating plots...[/bold]")
        generate_plots(filtered_data, args.output_dir, args.compare_models)

    console.print(f"\n[bold]Results saved to:[/bold] {args.output_dir}")


if __name__ == "__main__":
    main()
