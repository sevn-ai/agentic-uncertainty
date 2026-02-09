"""Online confidence analysis (Paper Section 3.3).

Analyzes trajectories generated with ConfidenceAgent, which elicits
confidence DURING live execution (Barkan et al. Experiment 3 replication).

This script processes trajectories that have embedded confidence_history
from the ConfidenceAgent, computing calibration and trajectory metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from agentic_uncertainty.data import load_ground_truth
from agentic_uncertainty.evaluation import auroc_with_ci, brier_score, expected_calibration_error
from agentic_uncertainty.scripts._shared import (
    compute_standard_metrics,
    get_output_dir,
    match_instance_to_ground_truth,
)

console = Console()


def load_online_trajectories(traj_dir: Path) -> list[dict]:
    """Load trajectories that have online confidence data.

    Args:
        traj_dir: Directory containing trajectory subdirectories.

    Returns:
        List of trajectory dicts with confidence_history.
    """
    trajs = []
    # Look for .traj.json files in subdirectories
    for traj_file in traj_dir.glob("*/*.traj.json"):
        with open(traj_file) as f:
            traj = json.load(f)

        # Check if this trajectory has online confidence data
        info = traj.get("info", {})
        if "confidence_history" in info or "confidence_summary" in info:
            traj["_traj_path"] = str(traj_file)
            trajs.append(traj)

    return trajs


def extract_confidence_trajectory(traj: dict) -> list[float]:
    """Extract confidence values from trajectory.

    Returns:
        List of confidence values (0-1 scale), or empty if none found.
    """
    info = traj.get("info", {})

    # Try confidence_summary first (has pre-computed trajectory)
    summary = info.get("confidence_summary", {})
    if "confidence_trajectory" in summary:
        return summary["confidence_trajectory"]

    # Fall back to confidence_history
    history = info.get("confidence_history", [])
    return [h["confidence"] for h in history if h.get("confidence") is not None]


def compute_trajectory_metrics(
    confidence_traj: list[float],
    resolved: bool,
) -> dict:
    """Compute metrics for a single confidence trajectory.

    Args:
        confidence_traj: List of confidence values over time.
        resolved: Whether the instance was resolved.

    Returns:
        Dict with trajectory-level metrics.
    """
    if not confidence_traj:
        return {}

    return {
        "n_steps": len(confidence_traj),
        "initial_confidence": confidence_traj[0],
        "final_confidence": confidence_traj[-1],
        "mean_confidence": np.mean(confidence_traj),
        "std_confidence": np.std(confidence_traj),
        "min_confidence": min(confidence_traj),
        "max_confidence": max(confidence_traj),
        "confidence_delta": confidence_traj[-1] - confidence_traj[0],
        "resolved": resolved,
    }


def compute_checkpoint_metrics(
    trajectories: list[dict],
    ground_truth: dict[str, bool],
    checkpoints: list[float] = [0.25, 0.5, 0.75, 1.0],
) -> dict[str, dict]:
    """Compute calibration metrics at normalized checkpoints.

    Args:
        trajectories: List of trajectory dicts with confidence data.
        ground_truth: Dict mapping instance_id to resolved status.
        checkpoints: Percentage-based checkpoints (0-1).

    Returns:
        Dict mapping checkpoint to metrics.
    """
    results = {}

    for cp in checkpoints:
        predictions = []
        labels = []

        for traj in trajectories:
            instance_id = traj.get("info", {}).get("instance_id", "")

            # Match to ground truth
            _, resolved = match_instance_to_ground_truth(instance_id, ground_truth)
            if resolved is None:
                continue

            # Get confidence at checkpoint
            conf_traj = extract_confidence_trajectory(traj)
            if not conf_traj:
                continue

            # Convert percentage to step index
            step_idx = max(0, int(cp * len(conf_traj)) - 1)
            confidence = conf_traj[step_idx]

            predictions.append(confidence)
            labels.append(1 if resolved else 0)

        if len(predictions) < 5:
            continue

        preds = np.array(predictions)
        y = np.array(labels)

        auroc_result = auroc_with_ci(preds, y)

        results[f"{cp:.0%}"] = {
            "checkpoint": cp,
            "n_samples": len(predictions),
            "auroc": auroc_result.auroc,
            "auroc_ci_lower": auroc_result.ci_lower,
            "auroc_ci_upper": auroc_result.ci_upper,
            "ece": expected_calibration_error(preds, y),
            "brier": brier_score(preds, y),
            "mean_confidence": float(preds.mean()),
            "actual_success_rate": float(y.mean()),
            "overconfidence": float(preds.mean() - y.mean()),
        }

    return results


def compute_dynamics_metrics(
    trajectories: list[dict],
    ground_truth: dict[str, bool],
) -> dict:
    """Compute metrics about confidence dynamics over time.

    Args:
        trajectories: List of trajectory dicts with confidence data.
        ground_truth: Dict mapping instance_id to resolved status.

    Returns:
        Dict with dynamics metrics.
    """
    resolved_deltas = []
    unresolved_deltas = []
    resolved_means = []
    unresolved_means = []

    for traj in trajectories:
        instance_id = traj.get("info", {}).get("instance_id", "")

        # Match to ground truth
        _, resolved = match_instance_to_ground_truth(instance_id, ground_truth)
        if resolved is None:
            continue

        conf_traj = extract_confidence_trajectory(traj)
        if len(conf_traj) < 2:
            continue

        delta = conf_traj[-1] - conf_traj[0]
        mean_conf = np.mean(conf_traj)

        if resolved:
            resolved_deltas.append(delta)
            resolved_means.append(mean_conf)
        else:
            unresolved_deltas.append(delta)
            unresolved_means.append(mean_conf)

    return {
        "resolved": {
            "n": len(resolved_deltas),
            "mean_delta": float(np.mean(resolved_deltas)) if resolved_deltas else None,
            "mean_confidence": float(np.mean(resolved_means)) if resolved_means else None,
        },
        "unresolved": {
            "n": len(unresolved_deltas),
            "mean_delta": float(np.mean(unresolved_deltas)) if unresolved_deltas else None,
            "mean_confidence": float(np.mean(unresolved_means)) if unresolved_means else None,
        },
    }


def run_analysis(
    traj_dir: Path,
    ground_truth_path: Path,
    output_dir: Path,
    checkpoints: list[float] = [0.25, 0.5, 0.75, 1.0],
) -> dict:
    """Run online confidence analysis.

    Args:
        traj_dir: Directory containing online confidence trajectories.
        ground_truth_path: Path to eval_results.json.
        output_dir: Directory for outputs.
        checkpoints: Percentage-based checkpoints for analysis.

    Returns:
        Results dictionary.
    """
    console.print("[bold]Online Confidence Analysis (Section 3.3)[/bold]")
    console.print(f"Trajectory dir: {traj_dir}")
    console.print(f"Checkpoints: {checkpoints}")

    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    ground_truth = load_ground_truth(ground_truth_path)
    console.print(f"Ground truth: {len(ground_truth)} instances")

    trajectories = load_online_trajectories(traj_dir)
    console.print(f"Loaded {len(trajectories)} trajectories with online confidence")

    if not trajectories:
        console.print("[red]No trajectories with confidence data found![/red]")
        console.print("Make sure trajectories were generated with --agent-class confidence")
        return {}

    # Compute metrics
    console.print("\n[bold]Computing metrics...[/bold]")

    checkpoint_metrics = compute_checkpoint_metrics(trajectories, ground_truth, checkpoints)
    dynamics_metrics = compute_dynamics_metrics(trajectories, ground_truth)

    # Compute overall calibration using final confidence
    final_predictions = []
    final_labels = []
    for traj in trajectories:
        instance_id = traj.get("info", {}).get("instance_id", "")
        conf_traj = extract_confidence_trajectory(traj)

        if not conf_traj:
            continue

        _, resolved = match_instance_to_ground_truth(instance_id, ground_truth)
        if resolved is not None:
            final_predictions.append(conf_traj[-1])
            final_labels.append(1 if resolved else 0)

    if final_predictions:
        overall_metrics = compute_standard_metrics(final_predictions, final_labels)
        # Add alias for compatibility
        overall_metrics["mean_confidence"] = overall_metrics["mean_prediction"]
        overall_metrics["actual_success_rate"] = overall_metrics["mean_label"]
    else:
        overall_metrics = {}

    # Prepare results
    results = {
        "experiment_type": "online_confidence",
        "timestamp": datetime.now().isoformat(),
        "traj_dir": str(traj_dir),
        "ground_truth_path": str(ground_truth_path),
        "n_trajectories": len(trajectories),
        "checkpoints": checkpoints,
        "overall_metrics": overall_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "dynamics_metrics": dynamics_metrics,
    }

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\nSaved results to {results_path}")

    # Print summary tables
    console.print("\n[bold]Overall Metrics (Final Confidence)[/bold]")
    if overall_metrics:
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("N", str(overall_metrics["n_samples"]))
        table.add_row("AUROC", f"{overall_metrics['auroc']:.3f} [{overall_metrics['auroc_ci_lower']:.3f}, {overall_metrics['auroc_ci_upper']:.3f}]")
        table.add_row("ECE", f"{overall_metrics['ece']:.3f}")
        table.add_row("Brier", f"{overall_metrics['brier']:.3f}")
        table.add_row("Mean Confidence", f"{overall_metrics['mean_confidence']:.3f}")
        table.add_row("Actual Success Rate", f"{overall_metrics['actual_success_rate']:.3f}")
        table.add_row("Overconfidence", f"{overall_metrics['overconfidence']:+.3f}")
        console.print(table)

    console.print("\n[bold]Metrics by Checkpoint[/bold]")
    if checkpoint_metrics:
        table = Table()
        table.add_column("Checkpoint")
        table.add_column("N")
        table.add_column("AUROC")
        table.add_column("ECE")
        table.add_column("Overconf.")
        for cp_label, metrics in sorted(checkpoint_metrics.items()):
            table.add_row(
                cp_label,
                str(metrics["n_samples"]),
                f"{metrics['auroc']:.3f}",
                f"{metrics['ece']:.3f}",
                f"{metrics['overconfidence']:+.3f}",
            )
        console.print(table)

    console.print("\n[bold]Confidence Dynamics[/bold]")
    if dynamics_metrics:
        table = Table()
        table.add_column("Outcome")
        table.add_column("N")
        table.add_column("Mean Delta")
        table.add_column("Mean Confidence")
        for outcome in ["resolved", "unresolved"]:
            m = dynamics_metrics[outcome]
            if m["n"] > 0:
                table.add_row(
                    outcome.capitalize(),
                    str(m["n"]),
                    f"{m['mean_delta']:+.3f}" if m["mean_delta"] is not None else "N/A",
                    f"{m['mean_confidence']:.3f}" if m["mean_confidence"] is not None else "N/A",
                )
        console.print(table)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze online confidence trajectories (Paper Section 3.3)"
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="Directory containing trajectories with online confidence data",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to eval_results.json with ground truth outcomes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/online_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
        help="Percentage-based checkpoints for analysis (default: 0.25 0.5 0.75 1.0)",
    )

    args = parser.parse_args()
    args.output_dir = get_output_dir(args.output_dir, "online")

    run_analysis(
        traj_dir=args.traj_dir,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        checkpoints=args.checkpoints,
    )


if __name__ == "__main__":
    main()
