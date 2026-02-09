"""Mid-execution checkpoint analysis.

Analyzes mid-execution checkpoint experiments to produce:
1. Early Warning Detection - AUROC at each checkpoint with progression visualization
2. Calibration Dynamics - ECE/Brier evolution across checkpoints
3. Confidence Trajectory - Per-instance prediction evolution visualization

Usage:
    uv run analyze-mid-execution \\
        --cache-dir cache \\
        --ground-truth-dir data/trajectories \\
        --output-dir results/mid_execution_analysis \\
        --plots --latex
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from agentic_uncertainty.scripts.analysis.evaluate_cache import (
    MetricsResult,
    compute_metrics,
)
from agentic_uncertainty.evaluation.plotting import (
    plot_roc_curves_comparison,
    plot_calibration_comparison,
    setup_latex_style,
)

logger = logging.getLogger(__name__)
console = Console()

CHECKPOINTS = [25, 50, 75]

# Model display names
MODEL_DISPLAY_NAMES = {
    "gpt-5.2-codex": "GPT-5.2-Codex",
    "gemini-3-pro-preview": "Gemini-3-Pro-Preview",
}

# Method display names for checkpoint methods
CHECKPOINT_DISPLAY_NAMES = {
    "mid_execution_direct_25pct": "25%",
    "mid_execution_direct_50pct": "50%",
    "mid_execution_direct_75pct": "75%",
}


def find_mid_execution_dirs(
    cache_dir: Path,
    evaluator_model: str,
    trajectory_model: str,
) -> dict[int, Path]:
    """Find mid-execution cache directories for a model combination.

    Args:
        cache_dir: Base cache directory.
        evaluator_model: Model used for evaluation.
        trajectory_model: Model that generated trajectories.

    Returns:
        Dict mapping checkpoint % to directory path.
    """
    result = {}

    # Multiple possible path patterns to check:
    # 1. cache/{evaluator}/mid_execution/{trajectory_model}/mid_{pct}/mid_execution/{evaluator}/mid_execution_direct_{pct}pct/
    # 2. cache/{evaluator}/mid_execution/mid_{pct}/mid_execution/{evaluator}/mid_execution_direct_{pct}pct/ (same-model case)

    base_paths = [
        cache_dir / evaluator_model / "mid_execution" / trajectory_model,
        cache_dir / evaluator_model / "mid_execution",
    ]

    for checkpoint in CHECKPOINTS:
        for base_path in base_paths:
            # Look for the nested structure with method suffix
            checkpoint_path = (
                base_path / f"mid_{checkpoint}" / "mid_execution" / evaluator_model / f"mid_execution_direct_{checkpoint}pct"
            )
            if checkpoint_path.exists():
                result[checkpoint] = checkpoint_path
                break

            # Alternative: without trajectory_model prefix in path but with suffix
            alt_path = base_path / f"mid_{checkpoint}" / "mid_execution" / evaluator_model / "mid_execution_direct"
            if alt_path.exists():
                # Check for files directly
                json_files = list(alt_path.glob("*.json"))
                json_files = [f for f in json_files if not f.name.endswith(".traj.json") and not f.name.endswith(".checkpoint.json")]
                if json_files:
                    result[checkpoint] = alt_path
                    break

    return result


def load_checkpoint_results(
    cache_dir: Path,
    ground_truth_dir: Path,
    evaluator_model: str,
    trajectory_model: str,
) -> dict[int, dict[str, Any]] | None:
    """Load predictions for all checkpoints for a model combination.

    Args:
        cache_dir: Base cache directory.
        ground_truth_dir: Directory containing ground truth files.
        evaluator_model: Model used for evaluation.
        trajectory_model: Model that generated trajectories.

    Returns:
        Dict: checkpoint -> {
            'predictions': np.ndarray,
            'labels': np.ndarray,
            'instance_ids': list[str],
        }
        Or None if data not found.
    """
    # Find checkpoint directories
    checkpoint_dirs = find_mid_execution_dirs(cache_dir, evaluator_model, trajectory_model)

    if not checkpoint_dirs:
        logger.warning(f"No mid-execution dirs found for {evaluator_model} on {trajectory_model}")
        return None

    # Load ground truth
    gt_path = ground_truth_dir / trajectory_model / "evaluation" / "eval_results.json"
    if not gt_path.exists():
        logger.warning(f"Ground truth not found: {gt_path}")
        return None

    with open(gt_path) as f:
        ground_truth = json.load(f)

    results = {}

    for checkpoint, dir_path in sorted(checkpoint_dirs.items()):
        predictions = []
        labels = []
        instance_ids = []

        # Load all JSON files in directory
        for json_path in dir_path.glob("*.json"):
            if json_path.name.endswith(".traj.json") or json_path.name.endswith(".checkpoint.json"):
                continue

            try:
                with open(json_path) as f:
                    data = json.load(f)

                instance_id = data.get("instance_id")
                prediction = data.get("prediction")

                if instance_id is None or prediction is None:
                    continue

                # Match with ground truth
                if instance_id not in ground_truth:
                    logger.debug(f"Instance {instance_id} not in ground truth, skipping")
                    continue

                predictions.append(prediction)
                labels.append(float(ground_truth[instance_id]))
                instance_ids.append(instance_id)

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Error reading {json_path}: {e}")
                continue

        if predictions:
            # Sort by instance_id for consistency
            sorted_indices = np.argsort(instance_ids)
            results[checkpoint] = {
                "predictions": np.array(predictions)[sorted_indices],
                "labels": np.array(labels)[sorted_indices],
                "instance_ids": [instance_ids[i] for i in sorted_indices],
            }
            logger.info(f"  {checkpoint}%: {len(predictions)} instances")

    return results if results else None


def compute_trajectory_metrics(
    checkpoint_data: dict[int, dict[str, Any]],
) -> dict[str, float]:
    """Compute per-instance evolution metrics.

    Args:
        checkpoint_data: checkpoint -> {'predictions', 'labels', 'instance_ids'}

    Returns:
        Dict with:
        - mean_confidence_change: avg(p_75 - p_25)
        - confidence_volatility: std of changes
        - correct_direction_pct: how often confidence moves toward truth
        - overconfidence_trend: whether overconfidence increases with progress
    """
    if 25 not in checkpoint_data or 75 not in checkpoint_data:
        return {}

    data_25 = checkpoint_data[25]
    data_75 = checkpoint_data[75]

    # Find common instances
    ids_25 = set(data_25["instance_ids"])
    ids_75 = set(data_75["instance_ids"])
    common_ids = sorted(ids_25 & ids_75)

    if not common_ids:
        return {}

    # Build aligned arrays
    idx_25 = {iid: i for i, iid in enumerate(data_25["instance_ids"])}
    idx_75 = {iid: i for i, iid in enumerate(data_75["instance_ids"])}

    p_25 = np.array([data_25["predictions"][idx_25[iid]] for iid in common_ids])
    p_75 = np.array([data_75["predictions"][idx_75[iid]] for iid in common_ids])
    labels = np.array([data_25["labels"][idx_25[iid]] for iid in common_ids])

    # Compute metrics
    confidence_change = p_75 - p_25

    # Correct direction: for successes, confidence should increase; for failures, decrease
    correct_direction = (
        ((labels == 1) & (confidence_change > 0)) |
        ((labels == 0) & (confidence_change < 0))
    )

    # Overconfidence at each checkpoint
    overconf_25 = np.mean(p_25) - np.mean(labels)
    overconf_75 = np.mean(p_75) - np.mean(labels)

    return {
        "mean_confidence_change": float(np.mean(confidence_change)),
        "confidence_volatility": float(np.std(confidence_change)),
        "correct_direction_pct": float(np.mean(correct_direction) * 100),
        "overconfidence_25pct": float(overconf_25),
        "overconfidence_75pct": float(overconf_75),
        "overconfidence_trend": float(overconf_75 - overconf_25),
        "n_common_instances": len(common_ids),
    }


def generate_progression_latex(
    all_metrics: dict[str, dict[int, MetricsResult]],
    output_path: Path,
    n_instances: int,
    base_rate: float,
) -> str:
    """Generate LaTeX table showing metric progression across checkpoints.

    Args:
        all_metrics: model_key -> checkpoint -> MetricsResult
        output_path: Path to save .tex file.
        n_instances: Number of instances evaluated.
        base_rate: Base rate of success.

    Returns:
        LaTeX table string.
    """
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Mid-execution checkpoint analysis: discrimination and calibration at different trajectory progress points (N={n_instances}, base rate={base_rate:.1%}).}}",
        "\\label{tab:mid_execution_progression}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llccccc@{}}",
        "\\toprule",
        "Model & Checkpoint & AUROC$\\uparrow$ & Mean & Overconf. & Brier$\\downarrow$ & ECE$\\downarrow$ \\\\",
        "\\midrule",
    ]

    for model_key, checkpoint_metrics in sorted(all_metrics.items()):
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        for i, (checkpoint, m) in enumerate(sorted(checkpoint_metrics.items())):
            model_col = model_display if i == 0 else ""
            checkpoint_str = f"{checkpoint}\\%"

            auroc_str = f"{m.auroc:.3f}"
            mean_str = f"{m.mean_prediction:.2f}"
            overconf_str = f"{m.overconfidence:+.2f}"
            brier_str = f"{m.brier:.3f}"
            ece_str = f"{m.ece:.3f}"

            lines.append(
                f"{model_col} & {checkpoint_str} & {auroc_str} & {mean_str} & {overconf_str} & {brier_str} & {ece_str} \\\\"
            )

        # Add midrule between models
        if model_key != sorted(all_metrics.keys())[-1]:
            lines.append("\\midrule")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def generate_trajectory_metrics_latex(
    trajectory_metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> str:
    """Generate LaTeX table showing trajectory evolution metrics.

    Args:
        trajectory_metrics: model_key -> trajectory metrics dict
        output_path: Path to save .tex file.

    Returns:
        LaTeX table string.
    """
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Confidence trajectory dynamics from 25\\% to 75\\% trajectory completion.}",
        "\\label{tab:trajectory_dynamics}",
        "\\small",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        "Model & $\\Delta$ Conf. & Volatility & Correct Dir. & $\\Delta$ Overconf. \\\\",
        "\\midrule",
    ]

    for model_key, metrics in sorted(trajectory_metrics.items()):
        if not metrics:
            continue

        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        delta_conf = f"{metrics.get('mean_confidence_change', 0):+.3f}"
        volatility = f"{metrics.get('confidence_volatility', 0):.3f}"
        correct_dir = f"{metrics.get('correct_direction_pct', 0):.1f}\\%"
        delta_overconf = f"{metrics.get('overconfidence_trend', 0):+.3f}"

        lines.append(f"{model_display} & {delta_conf} & {volatility} & {correct_dir} & {delta_overconf} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    table_str = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(table_str)

    return table_str


def generate_plots(
    all_data: dict[str, dict[int, dict[str, Any]]],
    all_metrics: dict[str, dict[int, MetricsResult]],
    output_dir: Path,
) -> None:
    """Generate visualization plots for mid-execution analysis.

    Args:
        all_data: model_key -> checkpoint -> {'predictions', 'labels', ...}
        all_metrics: model_key -> checkpoint -> MetricsResult
        output_dir: Directory to save plots.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agentic_uncertainty.evaluation.plotting import (
        plot_auroc_progression,
        plot_confidence_trajectories,
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. AUROC Progression plot
    auroc_data = {}
    for model_key, checkpoint_metrics in all_metrics.items():
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        aurocs = []
        ci_lowers = []
        ci_uppers = []
        for checkpoint in CHECKPOINTS:
            if checkpoint in checkpoint_metrics:
                m = checkpoint_metrics[checkpoint]
                aurocs.append(m.auroc)
                ci_lowers.append(m.auroc_ci_lower)
                ci_uppers.append(m.auroc_ci_upper)
            else:
                aurocs.append(np.nan)
                ci_lowers.append(np.nan)
                ci_uppers.append(np.nan)
        auroc_data[model_display] = {
            "checkpoints": CHECKPOINTS,
            "aurocs": aurocs,
            "ci_lower": ci_lowers,
            "ci_upper": ci_uppers,
        }

    if auroc_data:
        plot_auroc_progression(
            auroc_data,
            save_path=plots_dir / "auroc_progression.pdf",
        )
        console.print(f"[green]Saved:[/green] {plots_dir / 'auroc_progression.pdf'}")

    # 2. Per-model ROC curves (all checkpoints overlaid)
    for model_key, checkpoint_data in all_data.items():
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        model_results: dict[str, tuple] = {}

        for checkpoint, data in sorted(checkpoint_data.items()):
            label = f"{checkpoint}%"
            model_results[label] = (data["predictions"], data["labels"])

        if model_results:
            model_slug = model_key.replace("-", "_").replace(".", "_")
            plot_roc_curves_comparison(
                model_results,
                title=f"ROC Curves - {model_display}",
                save_path=plots_dir / f"roc_curves_{model_slug}.pdf",
            )
            console.print(f"[green]Saved:[/green] {plots_dir / f'roc_curves_{model_slug}.pdf'}")

    # 3. Per-model calibration curves
    for model_key, checkpoint_data in all_data.items():
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
        model_results: dict[str, tuple] = {}

        for checkpoint, data in sorted(checkpoint_data.items()):
            label = f"{checkpoint}%"
            model_results[label] = (data["predictions"], data["labels"])

        if model_results:
            model_slug = model_key.replace("-", "_").replace(".", "_")
            plot_calibration_comparison(
                model_results,
                title=f"Calibration - {model_display}",
                save_path=plots_dir / f"calibration_{model_slug}.pdf",
            )
            console.print(f"[green]Saved:[/green] {plots_dir / f'calibration_{model_slug}.pdf'}")

    # 4. Confidence trajectories (per-instance evolution)
    for model_key, checkpoint_data in all_data.items():
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        if len(checkpoint_data) >= 2:
            model_slug = model_key.replace("-", "_").replace(".", "_")
            plot_confidence_trajectories(
                checkpoint_data,
                title=None,  # No title - caption will explain
                save_path=plots_dir / f"confidence_trajectories_{model_slug}.pdf",
            )
            console.print(f"[green]Saved:[/green] {plots_dir / f'confidence_trajectories_{model_slug}.pdf'}")

    plt.close("all")


def print_results_table(
    all_metrics: dict[str, dict[int, MetricsResult]],
    trajectory_metrics: dict[str, dict[str, float]],
) -> None:
    """Print results summary to console."""
    table = Table(title="Mid-Execution Checkpoint Analysis")

    table.add_column("Model", style="cyan")
    table.add_column("Checkpoint", style="magenta")
    table.add_column("AUROC", justify="center")
    table.add_column("Mean", justify="center")
    table.add_column("Overconf", justify="center")
    table.add_column("Brier", justify="center")
    table.add_column("ECE", justify="center")

    for model_key, checkpoint_metrics in sorted(all_metrics.items()):
        model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)

        for i, (checkpoint, m) in enumerate(sorted(checkpoint_metrics.items())):
            model_col = model_display if i == 0 else ""

            table.add_row(
                model_col,
                f"{checkpoint}%",
                f"{m.auroc:.3f}",
                f"{m.mean_prediction:.2f}",
                f"{m.overconfidence:+.2f}",
                f"{m.brier:.3f}",
                f"{m.ece:.3f}",
            )

    console.print(table)

    # Print trajectory dynamics
    if any(trajectory_metrics.values()):
        console.print("\n[bold]Confidence Trajectory Dynamics (25% → 75%)[/bold]")
        for model_key, metrics in sorted(trajectory_metrics.items()):
            if not metrics:
                continue
            model_display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            console.print(f"  {model_display}:")
            console.print(f"    Mean Δ confidence: {metrics.get('mean_confidence_change', 0):+.3f}")
            console.print(f"    Volatility: {metrics.get('confidence_volatility', 0):.3f}")
            console.print(f"    Correct direction: {metrics.get('correct_direction_pct', 0):.1f}%")
            console.print(f"    Δ overconfidence: {metrics.get('overconfidence_trend', 0):+.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze mid-execution checkpoint experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Base cache directory containing model subdirectories",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth (e.g., data/trajectories)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mid_execution_analysis"),
        help="Output directory for results",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    console.print("[bold]Mid-Execution Checkpoint Analysis[/bold]")
    console.print("=" * 50)

    # Model configurations to analyze
    # Each tuple: (evaluator_model, trajectory_model)
    model_configs = [
        ("gpt-5.2-codex", "gpt-5.2-codex"),
        ("gemini-3-pro-preview", "gemini-3-pro-preview"),
    ]

    all_data: dict[str, dict[int, dict[str, Any]]] = {}
    all_metrics: dict[str, dict[int, MetricsResult]] = {}
    trajectory_metrics: dict[str, dict[str, float]] = {}

    for evaluator, trajectory in model_configs:
        model_key = evaluator  # Use evaluator as key since same-model evaluation
        console.print(f"\n[bold]Loading {evaluator} on {trajectory} trajectories...[/bold]")

        checkpoint_data = load_checkpoint_results(
            args.cache_dir,
            args.ground_truth_dir,
            evaluator,
            trajectory,
        )

        if not checkpoint_data:
            console.print(f"[yellow]No data found for {model_key}[/yellow]")
            continue

        all_data[model_key] = checkpoint_data

        # Compute metrics for each checkpoint
        all_metrics[model_key] = {}
        for checkpoint, data in checkpoint_data.items():
            metrics = compute_metrics(data["predictions"], data["labels"])
            all_metrics[model_key][checkpoint] = metrics

        # Compute trajectory evolution metrics
        trajectory_metrics[model_key] = compute_trajectory_metrics(checkpoint_data)

    if not all_metrics:
        console.print("[red]Error: No data found for any model configuration[/red]")
        return

    # Get sample counts and base rate from first available model/checkpoint
    first_model = next(iter(all_metrics))
    first_checkpoint = next(iter(all_metrics[first_model]))
    n_instances = all_metrics[first_model][first_checkpoint].n_samples
    base_rate = all_metrics[first_model][first_checkpoint].base_rate

    # Print results
    print_results_table(all_metrics, trajectory_metrics)

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    json_output = {
        "metadata": {
            "cache_dir": str(args.cache_dir),
            "ground_truth_dir": str(args.ground_truth_dir),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "n_instances": n_instances,
            "base_rate": base_rate,
            "checkpoints": CHECKPOINTS,
        },
        "metrics": {
            model: {
                str(checkpoint): asdict(m)
                for checkpoint, m in checkpoint_metrics.items()
            }
            for model, checkpoint_metrics in all_metrics.items()
        },
        "trajectory_dynamics": trajectory_metrics,
    }

    json_path = args.output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)
    console.print(f"\n[green]Saved:[/green] {json_path}")

    # Save trajectory metrics separately
    traj_metrics_path = args.output_dir / "trajectory_metrics.json"
    with open(traj_metrics_path, "w") as f:
        json.dump(trajectory_metrics, f, indent=2)
    console.print(f"[green]Saved:[/green] {traj_metrics_path}")

    # Generate LaTeX tables
    if args.latex:
        latex_path = args.output_dir / "progression_table.tex"
        generate_progression_latex(all_metrics, latex_path, n_instances, base_rate)
        console.print(f"[green]Saved:[/green] {latex_path}")

        traj_latex_path = args.output_dir / "trajectory_dynamics_table.tex"
        generate_trajectory_metrics_latex(trajectory_metrics, traj_latex_path)
        console.print(f"[green]Saved:[/green] {traj_latex_path}")

    # Generate plots
    if args.plots:
        console.print("\n[bold]Generating plots...[/bold]")
        generate_plots(all_data, all_metrics, args.output_dir)

    console.print(f"\n[bold]Results saved to:[/bold] {args.output_dir}")


if __name__ == "__main__":
    main()
