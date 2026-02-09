"""Generate figures for the paper.

Produces:
1. Hero overconfidence figure (main result)
2. AUROC progression figure (mid-execution checkpoints)
3. Expanded results table (LaTeX)

Usage:
    uv run generate-paper-figures \\
        --cache-dir cache \\
        --ground-truth-dir data/trajectories \\
        --output-dir paper/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
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
)
from agentic_uncertainty.evaluation.plotting import (
    plot_overconfidence_hero,
    plot_auroc_progression,
    plot_confidence_histograms_by_model,
    plot_calibration_curves_by_model,
    plot_confidence_trajectory_lines,
    plot_delta_confidence_analysis,
    plot_adversarial_shift_decomposition,
)


def save_figure(fig, base_path: Path, close: bool = True) -> None:
    """Save figure as both PDF and PNG."""
    import matplotlib.pyplot as plt

    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    console.print(f"[green]Saved:[/green] {base_path}.pdf + .png")

    if close:
        plt.close(fig)


logger = logging.getLogger(__name__)
console = Console()


@dataclass
class MethodStats:
    """Statistics for a single method."""
    n_samples: int
    auroc: float
    auroc_ci_lower: float
    auroc_ci_upper: float
    auprc: float
    auprc_ci_lower: float
    auprc_ci_upper: float
    brier: float
    ece: float
    overconfidence: float
    mean_prediction: float
    base_rate: float
    mean_steps: float
    mean_cost: float


def load_method_results(
    cache_dir: Path,
    ground_truth: dict[str, bool],
    evaluator_model: str,
    method_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    """Load predictions for multiple methods."""
    results = {}

    for method_name, method_path in method_paths.items():
        if not method_path.exists():
            logger.warning(f"Path not found: {method_path}")
            continue

        predictions = []
        labels = []
        instance_ids = []
        steps = []
        costs = []

        for json_path in method_path.glob("*.json"):
            if json_path.name.endswith(".traj.json") or json_path.name.endswith(".checkpoint.json"):
                continue

            try:
                with open(json_path) as f:
                    data = json.load(f)

                instance_id = data.get("instance_id")
                prediction = data.get("prediction")

                if instance_id is None or prediction is None:
                    continue

                if instance_id not in ground_truth:
                    continue

                predictions.append(prediction)
                labels.append(float(ground_truth[instance_id]))
                instance_ids.append(instance_id)

                metadata = data.get("metadata", {})
                n_steps = metadata.get("n_steps", 0)
                cost = (
                    metadata.get("exploration_cost", 0) or
                    metadata.get("review_cost", 0) or
                    metadata.get("mid_execution_cost", 0) or
                    0
                )
                steps.append(n_steps)
                costs.append(cost)

            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Error reading {json_path}: {e}")
                continue

        if predictions:
            results[method_name] = {
                "predictions": np.array(predictions),
                "labels": np.array(labels),
                "instance_ids": instance_ids,
                "steps": np.array(steps),
                "costs": np.array(costs),
            }
            logger.info(f"  {method_name}: {len(predictions)} instances")

    return results


def compute_method_stats(data: dict[str, Any]) -> MethodStats:
    """Compute all statistics for a method."""
    preds = data["predictions"]
    labels = data["labels"]

    auroc_result = auroc_with_ci(preds, labels)
    auprc_result = auprc_with_ci(preds, labels)

    return MethodStats(
        n_samples=len(preds),
        auroc=auroc_result.auroc,
        auroc_ci_lower=auroc_result.ci_lower,
        auroc_ci_upper=auroc_result.ci_upper,
        auprc=auprc_result.auprc,
        auprc_ci_lower=auprc_result.ci_lower,
        auprc_ci_upper=auprc_result.ci_upper,
        brier=brier_score(preds, labels),
        ece=expected_calibration_error(preds, labels),
        overconfidence=float(np.mean(preds) - np.mean(labels)),
        mean_prediction=float(np.mean(preds)),
        base_rate=float(np.mean(labels)),
        mean_steps=float(np.mean(data["steps"])) if len(data["steps"]) > 0 else 0,
        mean_cost=float(np.mean(data["costs"])) if len(data["costs"]) > 0 else 0,
    )


CHECKPOINTS = [25, 50, 75]


def aggregate_per_instance_trajectories(
    mid_execution_data: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Convert checkpoint-indexed data to instance-indexed trajectories.

    Takes the mid-execution data structure (model -> checkpoint -> instance data)
    and reorganizes it so each instance has its full trajectory across checkpoints.

    Args:
        mid_execution_data: model_key -> checkpoint -> {
            'predictions': np.ndarray,
            'labels': np.ndarray,
            'instance_ids': list[str],
        }

    Returns:
        model -> instance_id -> {
            'checkpoints': [25, 50, 75],
            'confidences': [0.6, 0.7, 0.8],
            'label': True/False
        }
    """
    result = {}

    for model_key, checkpoint_data in mid_execution_data.items():
        # Find instances that appear in all checkpoints
        all_instance_sets = []
        for checkpoint in CHECKPOINTS:
            if checkpoint in checkpoint_data:
                all_instance_sets.append(set(checkpoint_data[checkpoint]["instance_ids"]))

        if not all_instance_sets:
            continue

        # Get common instances across all checkpoints
        common_instances = set.intersection(*all_instance_sets)

        if not common_instances:
            continue

        # Build per-instance trajectories
        instance_trajectories = {}
        for instance_id in common_instances:
            checkpoints_list = []
            confidences_list = []
            label = None

            for checkpoint in CHECKPOINTS:
                if checkpoint not in checkpoint_data:
                    continue

                data = checkpoint_data[checkpoint]
                try:
                    idx = data["instance_ids"].index(instance_id)
                    checkpoints_list.append(checkpoint)
                    confidences_list.append(float(data["predictions"][idx]))
                    if label is None:
                        label = bool(data["labels"][idx])
                except ValueError:
                    continue

            if len(checkpoints_list) == len(CHECKPOINTS) and label is not None:
                instance_trajectories[instance_id] = {
                    "checkpoints": checkpoints_list,
                    "confidences": confidences_list,
                    "label": label,
                }

        if instance_trajectories:
            result[model_key] = instance_trajectories

    return result


def load_mid_execution_data(
    cache_dir: Path,
    ground_truth_dir: Path,
    model_configs: list[tuple[str, str]],
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load mid-execution checkpoint data for all models.

    Args:
        cache_dir: Base cache directory.
        ground_truth_dir: Directory containing ground truth.
        model_configs: List of (evaluator_model, trajectory_model) tuples.

    Returns:
        Dict: model_key -> checkpoint -> {'predictions', 'labels', 'instance_ids'}
    """
    all_data = {}

    for evaluator, trajectory_model in model_configs:
        # Load ground truth
        gt_path = ground_truth_dir / trajectory_model / "evaluation" / "eval_results.json"
        if not gt_path.exists():
            logger.warning(f"Ground truth not found: {gt_path}")
            continue

        with open(gt_path) as f:
            ground_truth = json.load(f)

        checkpoint_data = {}

        for checkpoint in CHECKPOINTS:
            # Try multiple path patterns based on model type
            if "claude" in evaluator.lower():
                # Claude path: cache/{evaluator}/mid_execution/{evaluator}/mid_{pct}/mid_execution/claude-opus-4-5-20251101/mid_execution_direct_{pct}pct/
                possible_paths = [
                    cache_dir / evaluator / "mid_execution" / evaluator /
                    f"mid_{checkpoint}" / "mid_execution" / "claude-opus-4-5-20251101" /
                    f"mid_execution_direct_{checkpoint}pct"
                ]
            else:
                # Try both patterns - with and without trajectory_model subdirectory
                possible_paths = [
                    # Pattern 1: with trajectory_model (Gemini)
                    cache_dir / evaluator / "mid_execution" / trajectory_model /
                    f"mid_{checkpoint}" / "mid_execution" / evaluator /
                    f"mid_execution_direct_{checkpoint}pct",
                    # Pattern 2: without trajectory_model (GPT)
                    cache_dir / evaluator / "mid_execution" /
                    f"mid_{checkpoint}" / "mid_execution" / evaluator /
                    f"mid_execution_direct_{checkpoint}pct",
                ]

            checkpoint_path = None
            for path in possible_paths:
                if path.exists():
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                logger.debug(f"Checkpoint paths not found for {evaluator} {checkpoint}%")
                continue

            predictions = []
            labels = []
            instance_ids = []

            for json_path in checkpoint_path.glob("*.json"):
                if json_path.name.endswith(".traj.json") or json_path.name.endswith(".checkpoint.json"):
                    continue

                try:
                    with open(json_path) as f:
                        data = json.load(f)

                    instance_id = data.get("instance_id")
                    prediction = data.get("prediction")

                    if instance_id is None or prediction is None:
                        continue

                    if instance_id not in ground_truth:
                        continue

                    predictions.append(prediction)
                    labels.append(float(ground_truth[instance_id]))
                    instance_ids.append(instance_id)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Error reading {json_path}: {e}")
                    continue

            if predictions:
                checkpoint_data[checkpoint] = {
                    "predictions": np.array(predictions),
                    "labels": np.array(labels),
                    "instance_ids": instance_ids,
                }
                logger.info(f"  {evaluator} {checkpoint}%: {len(predictions)} instances")

        if checkpoint_data:
            all_data[evaluator] = checkpoint_data

    return all_data


def generate_auroc_progression_figure(
    mid_execution_data: dict[str, dict[int, dict[str, Any]]],
    output_path: Path,
) -> None:
    """Generate AUROC progression figure showing discrimination across checkpoints.

    Args:
        mid_execution_data: model_key -> checkpoint -> {'predictions', 'labels', ...}
        output_path: Base path for saving (without extension).
    """
    if not mid_execution_data:
        console.print("  [yellow]No mid-execution data, skipping AUROC progression[/yellow]")
        return

    # Model display names
    model_display_names = {
        "gpt-5.2-codex": "GPT-5.2-Codex",
        "gemini-3-pro-preview": "Gemini-3-Pro",
        "claude-opus-4-5": "Claude-Opus-4.5",
    }

    # Build data for plotting
    auroc_data = {}
    for model_key, checkpoint_data in mid_execution_data.items():
        model_display = model_display_names.get(model_key, model_key)

        aurocs = []
        ci_lowers = []
        ci_uppers = []

        for checkpoint in CHECKPOINTS:
            if checkpoint in checkpoint_data:
                data = checkpoint_data[checkpoint]
                result = auroc_with_ci(data["predictions"], data["labels"])
                aurocs.append(result.auroc)
                ci_lowers.append(result.ci_lower)
                ci_uppers.append(result.ci_upper)
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
        fig = plot_auroc_progression(auroc_data)
        save_figure(fig, output_path)


def generate_expanded_table_latex(
    all_stats: dict[str, dict[str, MethodStats]],
    output_path: Path,
) -> str:
    """Generate expanded LaTeX table."""
    method_order = [
        "exploration_direct",
        "review_direct",
        "review_adversarial",
    ]
    method_names = {
        "exploration_direct": "Pre-Execution",
        "review_direct": "Post-Execution",
        "review_adversarial": "Adv.\\ Post-Execution",
    }

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Uncertainty estimation results. All methods exhibit substantial overconfidence (mean estimate $\\gg$ base rate) with limited discriminative ability (AUROC near chance).}",
        "\\label{tab:expanded_results}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "Model & Method & AUROC$\\uparrow$ & AUPRC$\\uparrow$ & Mean Est. & Overconf. & Brier$\\downarrow$ & ECE$\\downarrow$ \\\\",
        "\\midrule",
    ]

    for model in sorted(all_stats.keys()):
        model_stats = all_stats[model]
        if "gpt" in model.lower():
            model_display = "GPT-5.2-Codex"
        elif "gemini" in model.lower():
            model_display = "Gemini-3-Pro"
        else:
            model_display = "Claude-Opus-4.5"

        for i, method in enumerate(method_order):
            if method not in model_stats:
                continue

            s = model_stats[method]
            model_col = model_display if i == 0 else ""

            lines.append(
                f"{model_col} & {method_names[method]} & {s.auroc:.3f} & {s.auprc:.3f} & "
                f"{s.mean_prediction:.2f} & {s.overconfidence:+.2f} & {s.brier:.3f} & {s.ece:.3f} \\\\"
            )

        if model != sorted(all_stats.keys())[-1]:
            lines.append("\\midrule")

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


def print_summary_table(all_stats: dict[str, dict[str, MethodStats]]) -> None:
    """Print summary to console."""
    table = Table(title="Results Summary")

    table.add_column("Model", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("AUROC", justify="center")
    table.add_column("AUPRC", justify="center")
    table.add_column("Mean Est.", justify="center")
    table.add_column("Overconf.", justify="center")

    for model in sorted(all_stats.keys()):
        model_display = "GPT" if "gpt" in model.lower() else ("Gemini" if "gemini" in model.lower() else "Claude")
        for method, stats in all_stats[model].items():
            method_short = method.replace("_direct", "").replace("exploration", "pre").replace("review", "post")
            table.add_row(
                model_display,
                method_short,
                f"{stats.auroc:.3f}",
                f"{stats.auprc:.3f}",
                f"{stats.mean_prediction:.2f}",
                f"{stats.overconfidence:+.2f}",
            )

    console.print(table)


def _compute_adversarial_shift_data(
    all_data: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute per-outcome shift statistics for the adversarial decomposition figure.

    For each model, compares review_direct vs review_adversarial on the same
    instances, split by pass/fail outcome.

    Args:
        all_data: model -> method -> {'predictions', 'labels', 'instance_ids'}

    Returns:
        model -> shift statistics dict, or empty dict if data is missing.
    """
    from scipy import stats as scipy_stats

    shift_data = {}

    for model, methods in all_data.items():
        if "review_direct" not in methods or "review_adversarial" not in methods:
            continue

        std_data = methods["review_direct"]
        adv_data = methods["review_adversarial"]

        # Match instances across methods
        std_by_id = {
            iid: pred
            for iid, pred in zip(std_data["instance_ids"], std_data["predictions"])
            if pred is not None
        }
        adv_by_id = {
            iid: pred
            for iid, pred in zip(adv_data["instance_ids"], adv_data["predictions"])
            if pred is not None
        }
        label_by_id = {
            iid: label
            for iid, label in zip(std_data["instance_ids"], std_data["labels"])
        }

        common = sorted(set(std_by_id) & set(adv_by_id) & set(label_by_id))
        if not common:
            continue

        std_arr = np.array([std_by_id[i] for i in common])
        adv_arr = np.array([adv_by_id[i] for i in common])
        labels = np.array([label_by_id[i] for i in common])
        delta = std_arr - adv_arr  # positive = standard higher

        pass_mask = labels == 1
        fail_mask = labels == 0

        delta_pass = delta[pass_mask]
        delta_fail = delta[fail_mask]

        # t-test for differential shift
        _, p_val = scipy_stats.ttest_ind(delta_fail, delta_pass)

        shift_data[model] = {
            "shift_pass": float(delta_pass.mean()),
            "shift_fail": float(delta_fail.mean()),
            "shift_pass_se": float(delta_pass.std(ddof=1) / np.sqrt(len(delta_pass))),
            "shift_fail_se": float(delta_fail.std(ddof=1) / np.sqrt(len(delta_fail))),
            "std_gap": float(std_arr[pass_mask].mean() - std_arr[fail_mask].mean()),
            "adv_gap": float(adv_arr[pass_mask].mean() - adv_arr[fail_mask].mean()),
            "p_value": float(p_val),
        }

    return shift_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache"),
        help="Base cache directory",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=Path("data/trajectories"),
        help="Directory containing ground truth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Generating Paper Figures[/bold]")
    console.print("=" * 50)

    # Model configurations
    model_configs = [
        ("gpt-5.2-codex", "gpt-5.2-codex"),
        ("gemini-3-pro-preview", "gemini-3-pro-preview"),
        ("claude-opus-4-5", "claude-opus-4-5"),
    ]

    all_stats: dict[str, dict[str, MethodStats]] = {}
    all_data: dict[str, dict[str, dict[str, Any]]] = {}

    for evaluator, trajectory_model in model_configs:
        console.print(f"\n[bold]Loading {evaluator}...[/bold]")

        gt_path = args.ground_truth_dir / trajectory_model / "evaluation" / "eval_results.json"
        if not gt_path.exists():
            console.print(f"[yellow]Ground truth not found: {gt_path}[/yellow]")
            continue

        with open(gt_path) as f:
            ground_truth = json.load(f)

        # Standard paths (GPT, Gemini)
        method_paths = {
            "exploration_direct": args.cache_dir / evaluator / "exploration" / trajectory_model / "direct",
            "review_direct": args.cache_dir / evaluator / "review" / trajectory_model / "review_direct",
            "review_adversarial": args.cache_dir / evaluator / "review" / trajectory_model / "review_adversarial",
        }

        # Claude has a different path structure
        if "claude" in evaluator.lower():
            method_paths = {
                "exploration_direct": args.cache_dir / evaluator / "exploration_direct" / "exploration" / "claude-opus-4-5-20251101" / "direct",
                "review_direct": args.cache_dir / evaluator / "review_direct" / "review" / "claude-opus-4-5-20251101" / "review_direct",
                "review_adversarial": args.cache_dir / evaluator / "review_adversarial" / "review" / "claude-opus-4-5-20251101" / "review_adversarial",
            }

        results = load_method_results(args.cache_dir, ground_truth, evaluator, method_paths)

        if not results:
            console.print(f"[yellow]No results found for {evaluator}[/yellow]")
            continue

        all_data[evaluator] = results
        all_stats[evaluator] = {method: compute_method_stats(data) for method, data in results.items()}

    if not all_stats:
        console.print("[red]No data found![/red]")
        return

    print_summary_table(all_stats)

    # Generate figures
    console.print("\n[bold]Generating figures...[/bold]")

    # 1. Hero figure (overconfidence visualization) - use loaded all_stats
    model_mapping = {
        "gpt-5.2-codex": "gpt",
        "gemini-3-pro-preview": "gemini",
        "claude-opus-4-5": "claude",
    }
    method_mapping = {
        "exploration_direct": "pre",
        "review_direct": "post",
        "review_adversarial": "adv",
    }

    hero_data = {}
    for full_name, short_name in model_mapping.items():
        if full_name not in all_stats:
            continue
        hero_data[short_name] = {}
        for method_full, method_short in method_mapping.items():
            if method_full not in all_stats[full_name]:
                continue
            s = all_stats[full_name][method_full]
            hero_data[short_name][method_short] = {
                "auroc": s.auroc,
                "ece": s.ece,
                "mean_conf": s.mean_prediction,
                "base_rate": s.base_rate,
            }

    if hero_data:
        fig = plot_overconfidence_hero(hero_data)
        save_figure(fig, args.output_dir / "hero_results")
    else:
        console.print("  [yellow]No hero data available[/yellow]")

    # 2. Confidence histograms by model (post-execution, colored by outcome)
    # Build data structure for the new plotting functions
    model_results_for_figures = {
        model: {
            method: (data["predictions"], data["labels"])
            for method, data in methods.items()
        }
        for model, methods in all_data.items()
    }

    if model_results_for_figures:
        fig = plot_confidence_histograms_by_model(model_results_for_figures)
        save_figure(fig, args.output_dir / "confidence_histograms_by_model")

        fig = plot_calibration_curves_by_model(model_results_for_figures)
        save_figure(fig, args.output_dir / "calibration_by_model")
    else:
        console.print("  [yellow]No data for histograms/calibration curves[/yellow]")

    # 3b. Adversarial shift decomposition figure
    console.print("\n[bold]Generating adversarial shift decomposition...[/bold]")
    shift_data = _compute_adversarial_shift_data(all_data)
    if shift_data:
        fig = plot_adversarial_shift_decomposition(shift_data)
        save_figure(fig, args.output_dir / "adversarial_shift_decomposition")
    else:
        console.print("  [yellow]Missing review_direct or review_adversarial data[/yellow]")

    # 4. AUROC Progression figure (mid-execution checkpoints)
    console.print("\n[bold]Loading mid-execution data...[/bold]")
    mid_execution_data = load_mid_execution_data(args.cache_dir, args.ground_truth_dir, model_configs)
    if mid_execution_data:
        generate_auroc_progression_figure(mid_execution_data, args.output_dir / "auroc_progression")

        # 4b. Per-instance confidence trajectory visualizations
        console.print("\n[bold]Generating confidence trajectory visualizations...[/bold]")
        instance_trajectories = aggregate_per_instance_trajectories(mid_execution_data)

        if instance_trajectories:
            # Confidence trajectory lines (sunk cost visualization)
            # Use tighter SEM bands, subtle individual lines, legend in lower left
            fig = plot_confidence_trajectory_lines(
                instance_trajectories,
                show_means=True,
                show_bands=True,
                show_individual=True,
                alpha_individual=0.12,
                y_min=0.0,  # Keep full range to show the dramatic decline
            )
            save_figure(fig, args.output_dir / "confidence_trajectory_lines")

            # Delta confidence analysis (violin plot)
            fig, delta_stats = plot_delta_confidence_analysis(instance_trajectories)
            save_figure(fig, args.output_dir / "delta_confidence_by_outcome")

            # Save stats JSON for paper reference
            with open(args.output_dir / "delta_confidence_stats.json", "w") as f:
                json.dump(delta_stats, f, indent=2)
            console.print(f"[green]Saved:[/green] {args.output_dir / 'delta_confidence_stats.json'}")
        else:
            console.print("  [yellow]No instance trajectories available[/yellow]")
    else:
        console.print("  [yellow]No mid-execution data found[/yellow]")

    # 5. LaTeX table
    generate_expanded_table_latex(all_stats, args.output_dir / "expanded_results_table.tex")
    console.print(f"[green]Saved:[/green] {args.output_dir / 'expanded_results_table.tex'}")

    # 6. Save JSON results
    json_output = {
        model: {
            method: {
                "n_samples": stats.n_samples,
                "auroc": stats.auroc,
                "auprc": stats.auprc,
                "brier": stats.brier,
                "ece": stats.ece,
                "overconfidence": stats.overconfidence,
                "mean_prediction": stats.mean_prediction,
                "base_rate": stats.base_rate,
            }
            for method, stats in model_stats.items()
        }
        for model, model_stats in all_stats.items()
    }

    with open(args.output_dir / "expanded_results.json", "w") as f:
        json.dump(json_output, f, indent=2)
    console.print(f"[green]Saved:[/green] {args.output_dir / 'expanded_results.json'}")

    console.print(f"\n[bold]Results saved to:[/bold] {args.output_dir}")


if __name__ == "__main__":
    main()
