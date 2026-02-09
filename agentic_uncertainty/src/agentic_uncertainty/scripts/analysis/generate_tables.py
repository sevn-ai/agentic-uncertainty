"""Generate LaTeX tables from experiment results.

Produces publication-ready tables for the paper by scanning the results directory.

Usage:
    generate-tables --results-dir results/ --output tables/
    generate-tables --results-dir results/ --tables pre_execution in_context
"""

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()

EXPERIMENT_TYPES = ["pre_execution", "in_context", "traces", "terminal", "control"]


def scan_results(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Scan results directory and build index.

    Returns:
        Dict of experiment_type -> model -> {path, data}
    """
    index = {}

    for exp_type in EXPERIMENT_TYPES:
        exp_dir = results_dir / exp_type
        if not exp_dir.exists():
            continue

        index[exp_type] = {}

        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            results_path = model_dir / "results.json"
            metrics_path = model_dir / "metrics.json"  # traces uses this

            data_path = None
            if results_path.exists():
                data_path = results_path
            elif metrics_path.exists():
                data_path = metrics_path

            if data_path:
                try:
                    with open(data_path) as f:
                        data = json.load(f)
                    index[exp_type][model_dir.name] = {
                        "path": data_path,
                        "data": data,
                    }
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to load {data_path}: {e}[/yellow]")

    return index


def format_metric(value: float, precision: int = 3) -> str:
    """Format a metric value."""
    return f"{value:.{precision}f}"


def generate_pre_execution_table(index: dict, output_dir: Path) -> str:
    """Generate pre-execution results table (Table 1).

    Columns: Method | Model 1 AUROC | Model 2 AUROC | ...
    """
    exp_data = index.get("pre_execution", {})
    if not exp_data:
        console.print("[yellow]No pre_execution results found[/yellow]")
        return ""

    # Collect all models and methods
    models = sorted(exp_data.keys())
    all_methods = set()
    for model_entry in exp_data.values():
        result = model_entry.get("data", {})
        all_methods.update(result.get("results", {}).keys())
    methods = sorted(all_methods)

    if not methods:
        console.print("[yellow]No methods found in pre_execution results[/yellow]")
        return ""

    # Build LaTeX table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pre-execution uncertainty estimation (AUROC). Higher is better.}",
        "\\label{tab:pre_execution}",
        "\\begin{tabular}{l" + "c" * len(models) + "}",
        "\\toprule",
        "Method & " + " & ".join(m.replace("_", "\\_")[:15] for m in models) + " \\\\",
        "\\midrule",
    ]

    # Add rows for each method
    for method in methods:
        row_values = [method.replace("_", "\\_")]
        aurocs = []

        for model in models:
            result = exp_data[model].get("data", {})
            method_data = result.get("results", {}).get(method, {})

            auroc = method_data.get("auroc")
            if auroc is not None:
                row_values.append(format_metric(auroc))
                aurocs.append((len(row_values) - 1, auroc))
            else:
                row_values.append("--")

        # Bold the best value
        if aurocs:
            best_idx, _ = max(aurocs, key=lambda x: x[1])
            row_values[best_idx] = "\\textbf{" + row_values[best_idx] + "}"

        lines.append(" & ".join(row_values) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    # Save to file
    output_path = output_dir / "pre_execution.tex"
    with open(output_path, "w") as f:
        f.write(table_str)

    console.print(f"[green]Generated:[/green] {output_path}")
    return table_str


def generate_in_context_table(index: dict, output_dir: Path) -> str:
    """Generate in-context learning results table (Table 2).

    Shows calibration improvement across sequence positions.
    """
    exp_data = index.get("in_context", {})
    if not exp_data:
        console.print("[yellow]No in_context results found[/yellow]")
        return ""

    # Build table
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{In-context learning: Does experience reduce overconfidence?}",
        "\\label{tab:in_context}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Model & Pos 1 & Pos 2 & Pos 3 & Pos 4 & Pos 5 \\\\",
        "\\midrule",
    ]

    for model, model_entry in sorted(exp_data.items()):
        result = model_entry.get("data", {})
        metrics_by_pos = result.get("metrics_by_position", {})

        row = [model.replace("_", "\\_")[:20]]
        for pos in ["1", "2", "3", "4", "5"]:
            if pos in metrics_by_pos:
                overconf = metrics_by_pos[pos].get("overconfidence", 0)
                row.append(f"{overconf:+.3f}")
            else:
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\vspace{1mm}",
        "\\\\\\small{Values show overconfidence (mean confidence - base rate). Negative is better.}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path = output_dir / "in_context.tex"
    with open(output_path, "w") as f:
        f.write(table_str)

    console.print(f"[green]Generated:[/green] {output_path}")
    return table_str


def generate_traces_table(index: dict, output_dir: Path) -> str:
    """Generate uncertainty traces results table (Table 3).

    Shows AUROC at different checkpoints.
    """
    exp_data = index.get("traces", {})
    if not exp_data:
        console.print("[yellow]No traces results found[/yellow]")
        return ""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Uncertainty traces: AUROC at trajectory checkpoints.}",
        "\\label{tab:traces}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & Step 5 & Step 10 & Step 15 & Step 20 \\\\",
        "\\midrule",
    ]

    for model, model_entry in sorted(exp_data.items()):
        result = model_entry.get("data", {})
        metrics = result.get("metrics", {})

        # LLM predictions
        row = [f"LLM ({model[:10]})"]
        for cp in [5, 10, 15, 20]:
            key = f"llm_direct_cp{cp}"
            if key in metrics:
                row.append(format_metric(metrics[key].get("auroc", 0)))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

        # ML predictions
        row = ["ML (features)"]
        for cp in [5, 10, 15, 20]:
            key = f"ml_cp{cp}"
            if key in metrics:
                row.append(format_metric(metrics[key].get("auroc", 0)))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path = output_dir / "traces.tex"
    with open(output_path, "w") as f:
        f.write(table_str)

    console.print(f"[green]Generated:[/green] {output_path}")
    return table_str


def generate_terminal_table(index: dict, output_dir: Path) -> str:
    """Generate terminal self-evaluation results table."""
    exp_data = index.get("terminal", {})
    if not exp_data:
        console.print("[yellow]No terminal results found[/yellow]")
        return ""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Terminal self-evaluation: AUROC for patch success prediction.}",
        "\\label{tab:terminal}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Model & Direct & Failure Modes \\\\",
        "\\midrule",
    ]

    for model, model_entry in sorted(exp_data.items()):
        result = model_entry.get("data", {})
        results_data = result.get("results", {})

        row = [model.replace("_", "\\_")[:20]]
        for method in ["direct", "failure_modes"]:
            method_data = results_data.get(method, {})
            auroc = method_data.get("auroc")
            if auroc is not None:
                row.append(format_metric(auroc))
            else:
                row.append("--")

        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path = output_dir / "terminal.tex"
    with open(output_path, "w") as f:
        f.write(table_str)

    console.print(f"[green]Generated:[/green] {output_path}")
    return table_str


def generate_control_table(index: dict, output_dir: Path) -> str:
    """Generate control policy results table (Table 4).

    Shows efficiency metrics for different policies.
    """
    exp_data = index.get("control", {})
    if not exp_data:
        console.print("[yellow]No control results found[/yellow]")
        return ""

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Control policies: Efficiency comparison.}",
        "\\label{tab:control}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Policy & Resolve Rate & Cost/Resolved & Resolved/\\$ \\\\",
        "\\midrule",
    ]

    for model, model_entry in sorted(exp_data.items()):
        result = model_entry.get("data", {})
        metrics = result.get("metrics", {})
        policy = result.get("policy", "unknown")

        row = [
            policy.replace("_", "\\_"),
            f"{metrics.get('resolve_rate', 0):.1%}",
            f"\\${metrics.get('avg_cost_per_resolved', 0):.2f}",
            format_metric(metrics.get("resolved_per_dollar", 0)),
        ]
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path = output_dir / "control.tex"
    with open(output_path, "w") as f:
        f.write(table_str)

    console.print(f"[green]Generated:[/green] {output_path}")
    return table_str


TABLE_GENERATORS = {
    "pre_execution": generate_pre_execution_table,
    "in_context": generate_in_context_table,
    "traces": generate_traces_table,
    "terminal": generate_terminal_table,
    "control": generate_control_table,
}


def generate_all_tables(
    results_dir: Path,
    output_dir: Path,
    tables: list[str] | None = None,
) -> None:
    """Generate all requested tables.

    Args:
        results_dir: Directory containing experiment results.
        output_dir: Directory to save tables.
        tables: List of tables to generate (default: all).
    """
    console.print(f"[bold]Generating LaTeX tables[/bold]")
    console.print(f"Results: {results_dir}")
    console.print(f"Output: {output_dir}")

    # Scan results directory
    index = scan_results(results_dir)

    n_experiments = sum(len(models) for models in index.values())
    console.print(f"Found {n_experiments} experiment results across {len(index)} experiment types\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    tables_to_generate = tables or list(TABLE_GENERATORS.keys())

    for table_name in tables_to_generate:
        if table_name not in TABLE_GENERATORS:
            console.print(f"[yellow]Unknown table: {table_name}[/yellow]")
            continue

        generator = TABLE_GENERATORS[table_name]
        try:
            generator(index, output_dir)
        except Exception as e:
            console.print(f"[red]Error generating {table_name}: {e}[/red]")

    console.print(f"\n[bold]Tables saved to:[/bold] {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing experiment results (default: results/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tables"),
        help="Output directory for tables (default: results/tables/)",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLE_GENERATORS.keys()),
        help="Specific tables to generate (default: all)",
    )

    args = parser.parse_args()

    generate_all_tables(
        results_dir=args.results_dir,
        output_dir=args.output,
        tables=args.tables,
    )


if __name__ == "__main__":
    main()
