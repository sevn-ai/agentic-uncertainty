"""Analyze false commitment: high-confidence failures.

Deep dive into cases where agents are confident but wrong:
- False commitment rates at various confidence thresholds
- Cost analysis: are overconfident failures more expensive?
- Correlation between confidence and cost for failures
- Worst offenders: highest confidence, failed, expensive

Usage:
    python -m agentic_uncertainty.scripts.analysis.false_commitment \
        --predictions results/pre_execution/results.json \
        --ground-truth path/to/eval_results.json \
        --output-dir results/analysis/false_commitment/
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy import stats

from agentic_uncertainty.data.trajectories import get_cost, load_trajectories

console = Console()


def load_ground_truth(path: Path) -> dict[str, bool]:
    """Load ground truth from eval_results.json."""
    with open(path) as f:
        data = json.load(f)
    return {k: bool(v) for k, v in data.items()}


def load_predictions(path: Path, method: str = "direct") -> dict[str, float]:
    """Load predictions from results.json."""
    with open(path) as f:
        data = json.load(f)

    instance_ids = data.get("instance_ids", [])
    predictions = data.get("results", {}).get(method, {}).get("predictions", [])

    if len(instance_ids) != len(predictions):
        raise ValueError(f"Mismatch: {len(instance_ids)} ids but {len(predictions)} predictions")

    return dict(zip(instance_ids, predictions))


def load_costs_from_json(path: Path) -> dict[str, float]:
    """Load costs from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_costs_from_trajectories(traj_dir: Path) -> dict[str, float]:
    """Extract costs from trajectory files."""
    trajectories = load_trajectories(traj_dir)
    costs = {}
    for traj in trajectories:
        instance_id = traj.get("instance_id", "")
        cost = get_cost(traj)
        if cost > 0:
            costs[instance_id] = cost
            costs[instance_id.replace("-", "__")] = cost
            costs[instance_id.replace("__", "-")] = cost
    return costs


def compute_false_commitment_rate(
    confidences: list[float],
    outcomes: list[bool],
    threshold: float,
) -> dict:
    """Compute false commitment rate at a threshold.

    False commitment = P(fail | confidence > threshold)
    """
    confidences = np.array(confidences)
    outcomes = np.array(outcomes)

    high_conf_mask = confidences >= threshold
    n_high_conf = high_conf_mask.sum()

    if n_high_conf == 0:
        return {
            "threshold": threshold,
            "n_high_conf": 0,
            "n_high_conf_fail": 0,
            "false_commitment_rate": 0.0,
        }

    high_conf_outcomes = outcomes[high_conf_mask]
    n_high_conf_fail = (~high_conf_outcomes).sum()

    return {
        "threshold": threshold,
        "n_high_conf": int(n_high_conf),
        "n_high_conf_fail": int(n_high_conf_fail),
        "false_commitment_rate": float(n_high_conf_fail / n_high_conf),
    }


def analyze(
    predictions_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    costs_path: Path | None = None,
    traj_dir: Path | None = None,
    method: str = "direct",
    thresholds: list[float] | None = None,
) -> dict:
    """Analyze false commitment patterns."""
    thresholds = thresholds or [0.5, 0.6, 0.7, 0.8, 0.9]

    console.print("[bold]False Commitment Analysis[/bold]")
    console.print(f"Predictions: {predictions_path}")
    console.print(f"Method: {method}")

    # Load data
    predictions = load_predictions(predictions_path, method)
    ground_truth = load_ground_truth(ground_truth_path)

    console.print(f"Predictions: {len(predictions)} instances")
    console.print(f"Ground truth: {len(ground_truth)} instances")

    # Load costs
    costs = {}
    if costs_path and costs_path.exists():
        costs = load_costs_from_json(costs_path)
        console.print(f"Loaded costs from JSON: {len(costs)} instances")
    elif traj_dir and traj_dir.exists():
        costs = load_costs_from_trajectories(traj_dir)
        console.print(f"Extracted costs from trajectories: {len(costs)} instances")

    # Match data
    matched_ids = []
    matched_confidences = []
    matched_outcomes = []
    matched_costs = []

    for iid, conf in predictions.items():
        if iid in ground_truth:
            matched_ids.append(iid)
            matched_confidences.append(conf)
            matched_outcomes.append(ground_truth[iid])
            matched_costs.append(costs.get(iid, 2.0))

    console.print(f"Matched: {len(matched_ids)} instances")

    if not matched_ids:
        console.print("[red]No matched instances![/red]")
        return {}

    confidences = np.array(matched_confidences)
    outcomes = np.array(matched_outcomes)
    costs_arr = np.array(matched_costs)

    # Separate failures
    fail_mask = ~outcomes
    fail_confidences = confidences[fail_mask]
    fail_costs = costs_arr[fail_mask]

    n_failures = fail_mask.sum()
    total_fail_cost = fail_costs.sum()

    # False commitment rates at each threshold
    fcr_results = []
    for thresh in thresholds:
        fcr = compute_false_commitment_rate(matched_confidences, matched_outcomes, thresh)
        fcr_results.append(fcr)

    # Correlation between confidence and cost for failures
    if n_failures >= 3:
        correlation, p_value = stats.pearsonr(fail_confidences, fail_costs)
    else:
        correlation, p_value = 0.0, 1.0

    # Waste breakdown by confidence bucket
    waste_by_confidence = {}
    for thresh in thresholds:
        high_conf_fail_mask = fail_mask & (confidences >= thresh)
        cost_high_conf_fail = costs_arr[high_conf_fail_mask].sum()
        waste_by_confidence[f"conf_{int(thresh*100)}+"] = {
            "n_instances": int(high_conf_fail_mask.sum()),
            "total_cost": float(cost_high_conf_fail),
            "fraction_of_waste": float(cost_high_conf_fail / total_fail_cost) if total_fail_cost > 0 else 0,
        }

    # Worst offenders
    fail_indices = np.where(fail_mask)[0]
    fail_scores = fail_confidences * fail_costs
    worst_order = np.argsort(fail_scores)[::-1]

    worst_offenders = []
    for i in worst_order[:10]:
        idx = fail_indices[i]
        worst_offenders.append({
            "instance_id": matched_ids[idx],
            "confidence": float(confidences[idx]),
            "cost": float(costs_arr[idx]),
            "score": float(fail_scores[i]),
        })

    results = {
        "timestamp": datetime.now().isoformat(),
        "predictions_path": str(predictions_path),
        "ground_truth_path": str(ground_truth_path),
        "method": method,
        "summary": {
            "n_total": len(matched_ids),
            "n_resolved": int(outcomes.sum()),
            "n_failures": int(n_failures),
            "total_cost": float(costs_arr.sum()),
            "total_fail_cost": float(total_fail_cost),
            "mean_confidence_all": float(confidences.mean()),
            "mean_confidence_resolved": float(confidences[outcomes].mean()) if outcomes.any() else 0,
            "mean_confidence_failed": float(fail_confidences.mean()) if n_failures > 0 else 0,
        },
        "false_commitment_rates": fcr_results,
        "cost_confidence_correlation": {
            "correlation": float(correlation),
            "p_value": float(p_value),
        },
        "waste_by_confidence": waste_by_confidence,
        "worst_offenders": worst_offenders,
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "false_commitment.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nSaved results to {output_path}")

    # Print summary
    console.print("\n[bold]False Commitment Rates[/bold]")
    table = Table(show_header=True)
    table.add_column("Threshold")
    table.add_column("N High Conf")
    table.add_column("N Failed")
    table.add_column("FCR")

    for fcr in fcr_results:
        table.add_row(
            f"{fcr['threshold']:.0%}",
            str(fcr["n_high_conf"]),
            str(fcr["n_high_conf_fail"]),
            f"{fcr['false_commitment_rate']:.1%}",
        )
    console.print(table)

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze false commitment")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--costs", type=Path, default=None)
    parser.add_argument("--traj-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", default="direct")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9])

    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("results") / f"false_commitment_{timestamp}"

    analyze(
        predictions_path=args.predictions,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        costs_path=args.costs,
        traj_dir=args.traj_dir,
        method=args.method,
        thresholds=args.thresholds,
    )


if __name__ == "__main__":
    main()
