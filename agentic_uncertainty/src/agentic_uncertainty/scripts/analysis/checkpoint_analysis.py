"""Analyze checkpoint posthoc results for online confidence monitoring.

This script analyzes checkpoint-based confidence elicitation to understand:
- How confidence evolves during execution (success vs failure)
- Per-step AUROC: how discrimination changes over execution
- False commitment rates compared to exploration/review
- Calibration at different execution stages

Usage:
    python -m agentic_uncertainty.scripts.analysis.checkpoint_analysis \
        --results results/checkpoint_posthoc_100/checkpoint_posthoc/results.json \
        --output-dir results/analysis/checkpoint/
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

from agentic_uncertainty.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
)

console = Console()


def load_checkpoint_results(path: Path) -> dict:
    """Load checkpoint posthoc results."""
    with open(path) as f:
        return json.load(f)


def compute_confidence_traces_by_outcome(
    confidence_traces: list[list[dict]],
    labels: list[int],
) -> dict:
    """Compute mean confidence vs step, split by success/failure.

    Returns:
        Dict with 'success' and 'failure' keys, each containing:
        - steps: list of step numbers
        - mean_confidence: mean confidence at each step
        - std_confidence: std of confidence at each step
        - n_samples: number of samples at each step
    """
    # Group traces by outcome
    success_traces = [t for t, l in zip(confidence_traces, labels) if l == 1]
    failure_traces = [t for t, l in zip(confidence_traces, labels) if l == 0]

    def aggregate_traces(traces: list[list[dict]]) -> dict:
        """Aggregate traces to compute mean/std at each step."""
        step_confidences = defaultdict(list)

        for trace in traces:
            for point in trace:
                step = point["step"]
                conf = point.get("confidence")
                if conf is not None:
                    step_confidences[step].append(conf)

        # Sort by step
        steps = sorted(step_confidences.keys())
        mean_conf = []
        std_conf = []
        n_samples = []

        for step in steps:
            confs = step_confidences[step]
            mean_conf.append(float(np.mean(confs)))
            std_conf.append(float(np.std(confs)))
            n_samples.append(len(confs))

        return {
            "steps": steps,
            "mean_confidence": mean_conf,
            "std_confidence": std_conf,
            "n_samples": n_samples,
        }

    return {
        "success": aggregate_traces(success_traces),
        "failure": aggregate_traces(failure_traces),
    }


def compute_per_step_auroc(
    confidence_traces: list[list[dict]],
    labels: list[int],
    min_samples: int = 10,
) -> dict:
    """Compute AUROC using confidence at each step.

    For each step S, compute AUROC using confidence values at step S
    (only for trajectories that have reached step S).

    Returns:
        Dict with:
        - steps: list of step numbers
        - auroc: AUROC at each step
        - n_samples: number of samples at each step
    """
    # Collect confidence at each step
    step_data = defaultdict(lambda: {"confidences": [], "labels": []})

    for trace, label in zip(confidence_traces, labels):
        for point in trace:
            step = point["step"]
            conf = point.get("confidence")
            if conf is not None:
                step_data[step]["confidences"].append(conf)
                step_data[step]["labels"].append(label)

    # Compute AUROC at each step
    steps = sorted(step_data.keys())
    aurocs = []
    n_samples = []
    valid_steps = []

    for step in steps:
        confs = np.array(step_data[step]["confidences"])
        labs = np.array(step_data[step]["labels"])

        # Need at least min_samples and both classes present
        if len(confs) >= min_samples and len(np.unique(labs)) == 2:
            auroc = roc_auc_score(labs, confs)
            aurocs.append(float(auroc))
            n_samples.append(len(confs))
            valid_steps.append(step)

    return {
        "steps": valid_steps,
        "auroc": aurocs,
        "n_samples": n_samples,
    }


def compute_false_commitment_rate(
    predictions: list[float],
    labels: list[int],
    threshold: float,
) -> dict:
    """Compute false commitment rate at a confidence threshold.

    False commitment = fraction of failures that have confidence >= threshold.
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # Get failures
    fail_mask = labels == 0
    n_failures = fail_mask.sum()

    if n_failures == 0:
        return {
            "threshold": threshold,
            "n_failures": 0,
            "n_false_commitments": 0,
            "false_commitment_rate": 0.0,
        }

    # Count failures with high confidence
    fail_confidences = predictions[fail_mask]
    n_false_commitments = (fail_confidences >= threshold).sum()

    return {
        "threshold": threshold,
        "n_failures": int(n_failures),
        "n_false_commitments": int(n_false_commitments),
        "false_commitment_rate": float(n_false_commitments / n_failures),
    }


def compute_calibration_by_stage(
    confidence_traces: list[list[dict]],
    labels: list[int],
    early_steps: tuple[int, int] = (5, 20),
    late_steps: tuple[int, int] = (40, 100),
) -> dict:
    """Compare calibration at early vs late execution stages.

    Args:
        confidence_traces: List of confidence traces per trajectory
        labels: Ground truth labels
        early_steps: (min_step, max_step) for early stage
        late_steps: (min_step, max_step) for late stage

    Returns:
        Dict with ECE and Brier for early and late stages
    """
    def get_stage_predictions(min_step: int, max_step: int) -> tuple[list, list]:
        """Get predictions and labels for trajectories in a step range."""
        preds = []
        labs = []

        for trace, label in zip(confidence_traces, labels):
            # Find the last confidence in the step range
            stage_confs = [
                p["confidence"]
                for p in trace
                if min_step <= p["step"] <= max_step and p.get("confidence") is not None
            ]
            if stage_confs:
                # Use the last confidence in the range
                preds.append(stage_confs[-1])
                labs.append(label)

        return preds, labs

    early_preds, early_labs = get_stage_predictions(*early_steps)
    late_preds, late_labs = get_stage_predictions(*late_steps)

    results = {}

    if len(early_preds) >= 10:
        results["early"] = {
            "step_range": early_steps,
            "n_samples": len(early_preds),
            "mean_prediction": float(np.mean(early_preds)),
            "mean_label": float(np.mean(early_labs)),
            "ece": expected_calibration_error(early_preds, early_labs),
            "brier": brier_score(early_preds, early_labs),
        }

    if len(late_preds) >= 10:
        results["late"] = {
            "step_range": late_steps,
            "n_samples": len(late_preds),
            "mean_prediction": float(np.mean(late_preds)),
            "mean_label": float(np.mean(late_labs)),
            "ece": expected_calibration_error(late_preds, late_labs),
            "brier": brier_score(late_preds, late_labs),
        }

    return results


def analyze(
    results_path: Path,
    output_dir: Path,
) -> dict:
    """Run full checkpoint analysis."""
    console.print("[bold]Checkpoint Posthoc Analysis[/bold]")
    console.print(f"Results: {results_path}")

    # Load data
    data = load_checkpoint_results(results_path)

    predictions = data["results"]["predictions"]
    labels = data["results"]["labels"]
    confidence_traces = data["results"]["confidence_traces"]
    metrics = data["metrics"]

    console.print(f"Loaded {len(predictions)} trajectories")
    console.print(f"Success rate: {np.mean(labels):.1%}")

    # 1. Confidence traces by outcome
    console.print("\n[bold]1. Confidence Evolution[/bold]")
    traces_analysis = compute_confidence_traces_by_outcome(confidence_traces, labels)

    success_mean_early = traces_analysis["success"]["mean_confidence"][:4]
    failure_mean_early = traces_analysis["failure"]["mean_confidence"][:4]

    if success_mean_early and failure_mean_early:
        console.print(f"Early steps (5-20) - Success: {np.mean(success_mean_early):.2f}, Failure: {np.mean(failure_mean_early):.2f}")

    # 2. Per-step AUROC
    console.print("\n[bold]2. Per-Step AUROC[/bold]")
    per_step_auroc = compute_per_step_auroc(confidence_traces, labels)

    table = Table(show_header=True)
    table.add_column("Step")
    table.add_column("AUROC")
    table.add_column("N")

    for step, auroc, n in zip(
        per_step_auroc["steps"][:8],
        per_step_auroc["auroc"][:8],
        per_step_auroc["n_samples"][:8],
    ):
        table.add_row(str(step), f"{auroc:.3f}", str(n))
    console.print(table)

    # 3. False commitment analysis
    console.print("\n[bold]3. False Commitment Rates[/bold]")
    thresholds = [0.70, 0.80, 0.90]
    fc_results = []

    for thresh in thresholds:
        fc = compute_false_commitment_rate(predictions, labels, thresh)
        fc_results.append(fc)
        console.print(
            f"  {thresh:.0%}: {fc['false_commitment_rate']:.1%} "
            f"({fc['n_false_commitments']}/{fc['n_failures']} failures)"
        )

    # 4. Calibration by execution stage
    console.print("\n[bold]4. Calibration by Stage[/bold]")
    stage_calibration = compute_calibration_by_stage(confidence_traces, labels)

    for stage, cal in stage_calibration.items():
        console.print(
            f"  {stage.capitalize()} (steps {cal['step_range'][0]}-{cal['step_range'][1]}): "
            f"ECE={cal['ece']:.3f}, Brier={cal['brier']:.3f}, N={cal['n_samples']}"
        )

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "results_path": str(results_path),
        "summary": {
            "n_trajectories": len(predictions),
            "success_rate": float(np.mean(labels)),
            "mean_prediction": metrics["mean_prediction"],
            "overconfidence": metrics["overconfidence"],
            "auroc": metrics["auroc"],
            "ece": metrics["ece"],
            "brier": metrics["brier"],
        },
        "confidence_traces": traces_analysis,
        "per_step_auroc": per_step_auroc,
        "false_commitment": {
            "thresholds": thresholds,
            "results": fc_results,
        },
        "stage_calibration": stage_calibration,
        # Data for paper figures
        "figure_data": {
            "confidence_evolution": {
                "success": {
                    "steps": traces_analysis["success"]["steps"],
                    "mean": traces_analysis["success"]["mean_confidence"],
                    "std": traces_analysis["success"]["std_confidence"],
                },
                "failure": {
                    "steps": traces_analysis["failure"]["steps"],
                    "mean": traces_analysis["failure"]["mean_confidence"],
                    "std": traces_analysis["failure"]["std_confidence"],
                },
            },
            "per_step_auroc": {
                "steps": per_step_auroc["steps"],
                "auroc": per_step_auroc["auroc"],
            },
            "false_commitment_comparison": {
                "thresholds": ["70%", "80%", "90%"],
                "exploration": [12.9, 0.0, 0.0],
                "review": [53.2, 8.1, 1.6],
                "checkpoint": [
                    fc_results[0]["false_commitment_rate"] * 100,
                    fc_results[1]["false_commitment_rate"] * 100,
                    fc_results[2]["false_commitment_rate"] * 100,
                ],
            },
        },
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "checkpoint_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nSaved results to {output_path}")

    # Print summary for paper
    console.print("\n[bold]Summary for Paper[/bold]")
    console.print(f"AUROC: {metrics['auroc']:.3f} (vs exploration 0.620, review 0.627)")
    console.print(f"ECE: {metrics['ece']:.3f} (vs exploration 0.267, review 0.311)")
    console.print(f"Brier: {metrics['brier']:.3f} (vs exploration 0.288, review 0.311)")
    console.print(f"Mean prediction: {metrics['mean_prediction']:.3f}")
    console.print(f"Overconfidence: +{metrics['overconfidence']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint posthoc results")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/checkpoint_posthoc_100/checkpoint_posthoc/results.json"),
        help="Path to checkpoint results JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/checkpoint"),
        help="Output directory for analysis results",
    )

    args = parser.parse_args()
    analyze(args.results, args.output_dir)


if __name__ == "__main__":
    main()
