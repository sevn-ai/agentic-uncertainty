"""Centralized table utilities for experiment result display.

Provides consistent table formatting across all experiment scripts using Rich tables.

Usage:
    from agentic_uncertainty.scripts._shared.tables import (
        print_metrics_table,
        print_position_table,
        print_kv_table,
        fmt_metric,
        fmt_auroc_ci,
        fmt_signed,
    )
"""

from dataclasses import dataclass
from typing import Any, Callable

from rich.console import Console
from rich.table import Table


# --- Formatters ---


def fmt_metric(value: float | None, precision: int = 3) -> str:
    """Format a metric value with fixed precision."""
    if value is None:
        return "--"
    return f"{value:.{precision}f}"


def fmt_signed(value: float | None, precision: int = 3) -> str:
    """Format a metric value with sign (+/-) prefix."""
    if value is None:
        return "--"
    return f"{value:+.{precision}f}"


def fmt_percent(value: float | None, precision: int = 1) -> str:
    """Format a value as percentage."""
    if value is None:
        return "--"
    return f"{value:.{precision}%}"


def fmt_currency(value: float | None, precision: int = 2) -> str:
    """Format a value as currency."""
    if value is None:
        return "--"
    return f"${value:.{precision}f}"


def fmt_int(value: int | float | None) -> str:
    """Format an integer value."""
    if value is None:
        return "--"
    return str(int(value))


def fmt_auroc_ci(
    auroc: float | None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    precision: int = 3,
) -> str:
    """Format AUROC with optional confidence interval.

    Examples:
        fmt_auroc_ci(0.75) -> "0.750"
        fmt_auroc_ci(0.75, 0.70, 0.80) -> "0.750 [0.700, 0.800]"
    """
    if auroc is None:
        return "--"
    base = f"{auroc:.{precision}f}"
    if ci_lower is not None and ci_upper is not None:
        return f"{base} [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
    return base


# --- Column Definitions ---


@dataclass
class Column:
    """Definition of a table column."""

    key: str
    header: str
    formatter: Callable[[Any], str] = fmt_metric
    style: str | None = None


# Common column definitions for metrics tables
METRICS_COLUMNS = {
    "method": Column("method", "Method", formatter=str),
    "n": Column("n", "N", formatter=fmt_int),
    "n_samples": Column("n_samples", "N", formatter=fmt_int),
    "auroc": Column("auroc", "AUROC", formatter=fmt_metric),
    "ece": Column("ece", "ECE", formatter=fmt_metric),
    "brier": Column("brier", "Brier", formatter=fmt_metric),
    "overconfidence": Column("overconfidence", "Overconf.", formatter=fmt_signed),
    "mean_confidence": Column("mean_confidence", "Mean Conf.", formatter=fmt_metric),
    "actual_success_rate": Column("actual_success_rate", "Actual Rate", formatter=fmt_metric),
    "false_commitment_rate": Column("false_commitment_rate", "FCR", formatter=fmt_metric),
    "mean_prediction": Column("mean_prediction", "Mean Pred.", formatter=fmt_metric),
    "std_prediction": Column("std_prediction", "Std Dev", formatter=fmt_metric),
    "position": Column("position", "Position", formatter=str),
    "checkpoint": Column("checkpoint", "Checkpoint", formatter=str),
}


# --- Table Builders ---


def print_metrics_table(
    data: list[dict[str, Any]],
    columns: list[str | Column],
    console: Console,
    title: str | None = None,
    row_key: str = "method",
) -> None:
    """Print a table of metrics (e.g., Method vs AUROC/ECE/Brier).

    Args:
        data: List of dicts with metric values. Each dict represents a row.
        columns: List of column keys (from METRICS_COLUMNS) or Column objects.
        console: Rich Console for output.
        title: Optional table title.
        row_key: Key in data dict that identifies the row (default: "method").

    Example:
        data = [
            {"method": "direct", "auroc": 0.75, "ece": 0.12, "brier": 0.22},
            {"method": "calibrated", "auroc": 0.78, "ece": 0.10, "brier": 0.20},
        ]
        print_metrics_table(data, ["method", "auroc", "ece", "brier"], console)
    """
    table = Table(show_header=True, title=title)

    # Resolve column definitions
    resolved_cols = []
    for col in columns:
        if isinstance(col, str):
            if col in METRICS_COLUMNS:
                resolved_cols.append(METRICS_COLUMNS[col])
            else:
                # Create a simple column for unknown keys
                resolved_cols.append(Column(col, col.replace("_", " ").title()))
        else:
            resolved_cols.append(col)

    # Add columns to table
    for col in resolved_cols:
        table.add_column(col.header, style=col.style)

    # Add rows
    for row_data in data:
        row_values = []
        for col in resolved_cols:
            value = row_data.get(col.key)
            row_values.append(col.formatter(value))
        table.add_row(*row_values)

    console.print(table)


def print_position_table(
    metrics_by_position: dict[str, dict[str, Any]],
    console: Console,
    title: str = "Results by Position",
    include_ci: bool = True,
) -> None:
    """Print a position-based metrics table (e.g., for in-context learning).

    Args:
        metrics_by_position: Dict mapping position (str) to metrics dict.
        console: Rich Console for output.
        title: Table title.
        include_ci: Whether to include confidence intervals for AUROC.

    Example:
        metrics = {
            "1": {"auroc": 0.75, "auroc_ci_lower": 0.70, "auroc_ci_upper": 0.80, ...},
            "2": {"auroc": 0.78, ...},
        }
        print_position_table(metrics, console)
    """
    table = Table(show_header=True, title=title)
    table.add_column("Position")
    table.add_column("AUROC")
    table.add_column("Mean Conf.")
    table.add_column("Actual Rate")
    table.add_column("Overconf.")
    table.add_column("N")

    for position in sorted(metrics_by_position.keys(), key=lambda x: int(x)):
        m = metrics_by_position[position]

        if include_ci:
            auroc_str = fmt_auroc_ci(
                m.get("auroc"),
                m.get("auroc_ci_lower"),
                m.get("auroc_ci_upper"),
            )
        else:
            auroc_str = fmt_metric(m.get("auroc"))

        table.add_row(
            str(position),
            auroc_str,
            fmt_metric(m.get("mean_confidence")),
            fmt_metric(m.get("actual_success_rate")),
            fmt_signed(m.get("overconfidence")),
            fmt_int(m.get("n")),
        )

    console.print(table)


def print_checkpoint_table(
    metrics: dict[str, dict[str, Any]],
    console: Console,
    title: str = "Results Summary",
) -> None:
    """Print a checkpoint-based metrics table (e.g., for traces experiment).

    Args:
        metrics: Dict mapping metric keys (e.g., "llm_direct_25%") to metrics dict.
        console: Rich Console for output.
        title: Table title.
    """
    table = Table(show_header=True, title=title)
    table.add_column("Checkpoint")
    table.add_column("Method")
    table.add_column("AUROC")
    table.add_column("ECE")
    table.add_column("Brier")
    table.add_column("FCR")

    for key in sorted(metrics.keys()):
        m = metrics[key]
        parts = key.split("_")

        if key.startswith("llm"):
            method = parts[1]
            cp = parts[2].replace("cp", "") if len(parts) > 2 else "--"
        else:
            method = "ML"
            cp = parts[1].replace("cp", "") if len(parts) > 1 else "--"

        table.add_row(
            cp,
            method,
            fmt_metric(m.get("auroc")),
            fmt_metric(m.get("ece")),
            fmt_metric(m.get("brier")),
            fmt_metric(m.get("false_commitment_rate")),
        )

    console.print(table)


def print_kv_table(
    data: dict[str, Any] | list[tuple[str, Any]],
    console: Console,
    title: str | None = None,
    key_header: str = "Metric",
    value_header: str = "Value",
    formatters: dict[str, Callable[[Any], str]] | None = None,
) -> None:
    """Print a simple key-value table (e.g., efficiency metrics).

    Args:
        data: Dict or list of (key, value) tuples.
        console: Rich Console for output.
        title: Optional table title.
        key_header: Header for key column.
        value_header: Header for value column.
        formatters: Optional dict mapping keys to formatter functions.

    Example:
        data = {"Instances": 100, "Resolved": 75, "Resolve Rate": 0.75}
        print_kv_table(data, console, title="Summary")
    """
    table = Table(show_header=True, title=title)
    table.add_column(key_header)
    table.add_column(value_header)

    formatters = formatters or {}

    if isinstance(data, dict):
        items = list(data.items())
    else:
        items = data

    for key, value in items:
        formatter = formatters.get(key, str)
        table.add_row(str(key), formatter(value))

    console.print(table)


def print_efficiency_table(
    metrics: dict[str, Any],
    console: Console,
    title: str = "Efficiency Metrics",
) -> None:
    """Print a control policy efficiency metrics table.

    Args:
        metrics: Dict with keys like num_instances, num_resolved, resolve_rate, etc.
        console: Rich Console for output.
        title: Table title.
    """
    rows = [
        ("Instances", fmt_int(metrics.get("num_instances"))),
        ("Resolved", fmt_int(metrics.get("num_resolved"))),
        ("Resolve Rate", fmt_percent(metrics.get("resolve_rate"))),
        ("Total Cost", fmt_currency(metrics.get("total_cost"))),
        ("Avg Cost/Instance", fmt_currency(metrics.get("avg_cost_per_instance"))),
        ("Avg Cost/Resolved", fmt_currency(metrics.get("avg_cost_per_resolved"))),
        ("Resolved/Dollar", fmt_metric(metrics.get("resolved_per_dollar"))),
    ]

    table = Table(show_header=True, title=title)
    table.add_column("Metric")
    table.add_column("Value")

    for metric, value in rows:
        table.add_row(metric, value)

    console.print(table)


def print_summary_table(
    results: dict[str, dict],
    console: Console,
    title: str = "Summary Statistics",
) -> None:
    """Print summary statistics for experiment results (legacy compatibility).

    This provides backward compatibility with the existing print_summary_statistics
    function in runner.py.

    Args:
        results: Results dict from run_estimators_on_tasks().
        console: Rich Console for output.
        title: Title for the statistics table.
    """
    import numpy as np

    data = []
    for method, method_data in results.items():
        preds = method_data.get("predictions", [])
        if preds:
            data.append({
                "method": method,
                "mean_prediction": float(np.mean(preds)),
                "std_prediction": float(np.std(preds)),
            })

    print_metrics_table(
        data,
        ["method", "mean_prediction", "std_prediction"],
        console,
        title=title,
    )
