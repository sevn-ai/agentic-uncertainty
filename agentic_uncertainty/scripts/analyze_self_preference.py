#!/usr/bin/env python3
"""
Self-Preference Ablation Analysis

Analyzes whether LLM judges exhibit self-preference bias when evaluating patches
from their own model family vs. a different model family.

Experimental Design: 2x2 Matrix
- GPT-self: GPT judge on GPT patches (from existing cache)
- GPT-cross: GPT judge on Gemini patches (new experiment)
- Gemini-self: Gemini judge on Gemini patches (new experiment)
- Gemini-cross: Gemini judge on GPT patches (new experiment)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class ConditionResult:
    """Results for a single experimental condition."""
    judge: str
    patch_source: str
    condition_type: str  # 'self' or 'cross'
    predictions: List[float]
    ground_truth: List[bool]
    instance_ids: List[str]

    @property
    def mean_confidence(self) -> float:
        return np.mean(self.predictions)

    @property
    def base_rate(self) -> float:
        return np.mean(self.ground_truth)

    @property
    def overconfidence(self) -> float:
        return self.mean_confidence - self.base_rate

    @property
    def n_samples(self) -> int:
        return len(self.predictions)


def compute_ece(predictions: List[float], ground_truth: List[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(ground_truth[mask])
            bin_confidence = np.mean(predictions[mask])
            bin_weight = np.sum(mask) / len(predictions)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def compute_auroc(predictions: List[float], ground_truth: List[bool]) -> float:
    """Compute AUROC."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(ground_truth, predictions)
    except ValueError:
        return 0.5  # Return 0.5 if all labels are the same


def compute_brier(predictions: List[float], ground_truth: List[bool]) -> float:
    """Compute Brier score."""
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth, dtype=float)
    return np.mean((predictions - ground_truth) ** 2)


def load_cache_results(cache_dir: str, instance_ids: List[str]) -> Dict[str, dict]:
    """Load cache results for specified instance IDs."""
    results = {}
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"Warning: Cache directory does not exist: {cache_dir}")
        return results

    # Build instance_id -> file mapping
    for json_file in cache_path.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                instance_id = data.get("instance_id")
                if instance_id in instance_ids:
                    results[instance_id] = data
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def load_ground_truth(gt_path: str) -> Dict[str, bool]:
    """Load ground truth results."""
    with open(gt_path) as f:
        return json.load(f)


def extract_condition_results(
    cache_results: Dict[str, dict],
    ground_truth: Dict[str, bool],
    judge: str,
    patch_source: str,
    condition_type: str
) -> ConditionResult:
    """Extract results for a condition from cache data."""
    predictions = []
    gt_values = []
    instance_ids = []

    for instance_id, data in cache_results.items():
        pred = data.get("prediction")
        if pred is not None and instance_id in ground_truth:
            predictions.append(pred)
            gt_values.append(ground_truth[instance_id])
            instance_ids.append(instance_id)

    return ConditionResult(
        judge=judge,
        patch_source=patch_source,
        condition_type=condition_type,
        predictions=predictions,
        ground_truth=gt_values,
        instance_ids=instance_ids
    )


def compute_self_preference_delta(
    self_result: ConditionResult,
    cross_result: ConditionResult
) -> Tuple[float, float, float]:
    """
    Compute self-preference effect.

    Returns:
        delta_mean: Mean difference (self - cross)
        delta_std: Std of per-instance differences
        p_value: p-value from paired t-test
    """
    # Align instances
    common_ids = set(self_result.instance_ids) & set(cross_result.instance_ids)

    self_preds = {iid: pred for iid, pred in zip(self_result.instance_ids, self_result.predictions)}
    cross_preds = {iid: pred for iid, pred in zip(cross_result.instance_ids, cross_result.predictions)}

    self_aligned = [self_preds[iid] for iid in common_ids]
    cross_aligned = [cross_preds[iid] for iid in common_ids]

    deltas = [s - c for s, c in zip(self_aligned, cross_aligned)]

    delta_mean = np.mean(deltas)
    delta_std = np.std(deltas, ddof=1)

    # Paired t-test
    if len(deltas) > 1:
        _, p_value = stats.ttest_rel(self_aligned, cross_aligned)
    else:
        p_value = 1.0

    return delta_mean, delta_std, p_value


def print_results_table(conditions: List[ConditionResult]):
    """Print Table 1: Self-Preference Ablation results."""
    print("\n" + "=" * 80)
    print("Table 1: Self-Preference Ablation (N=25)")
    print("=" * 80)
    print(f"{'Judge':<15} {'Patches':<15} {'N':<5} {'Mean Conf':<12} {'AUROC':<8} {'ECE':<8} {'Brier':<8}")
    print("-" * 80)

    for cond in conditions:
        auroc = compute_auroc(cond.predictions, cond.ground_truth)
        ece = compute_ece(cond.predictions, cond.ground_truth)
        brier = compute_brier(cond.predictions, cond.ground_truth)

        patch_label = f"{cond.patch_source} ({cond.condition_type})"
        print(f"{cond.judge:<15} {patch_label:<15} {cond.n_samples:<5} {cond.mean_confidence:.3f}        {auroc:.3f}    {ece:.3f}    {brier:.3f}")

    print("=" * 80)


def print_self_preference_effects(gpt_results: Tuple, gemini_results: Tuple):
    """Print Table 2: Self-Preference Effect."""
    print("\n" + "=" * 70)
    print("Table 2: Self-Preference Effect")
    print("=" * 70)
    print(f"{'Judge':<15} {'Delta Confidence (self - cross)':<35} {'p-value':<10}")
    print("-" * 70)

    gpt_delta, gpt_std, gpt_p = gpt_results
    gemini_delta, gemini_std, gemini_p = gemini_results

    print(f"{'GPT':<15} {gpt_delta:+.3f} (+/- {gpt_std:.3f})                    {gpt_p:.4f}")
    print(f"{'Gemini':<15} {gemini_delta:+.3f} (+/- {gemini_std:.3f})                    {gemini_p:.4f}")

    print("=" * 70)
    print("\nInterpretation:")
    print("  Delta > 0: Self-preference (judges more confident on own-model patches)")
    print("  Delta < 0: Cross-preference (judges more confident on other-model patches)")
    print("  Delta ~ 0: No bias")
    print("")

    # Statistical significance
    alpha = 0.05
    if gpt_p < alpha:
        gpt_sig = "significant" if gpt_delta > 0 else "significant (reverse)"
    else:
        gpt_sig = "not significant"

    if gemini_p < alpha:
        gemini_sig = "significant" if gemini_delta > 0 else "significant (reverse)"
    else:
        gemini_sig = "not significant"

    print(f"GPT self-preference: {gpt_sig} (p={gpt_p:.4f})")
    print(f"Gemini self-preference: {gemini_sig} (p={gemini_p:.4f})")


def main():
    base_dir = Path("/home/jean/ac_paper/code/agentic_uncertainty")

    # Load instance IDs
    with open(base_dir / "data/self_preference_ablation/instances_25.json") as f:
        instance_ids = json.load(f)

    print(f"Analyzing {len(instance_ids)} instances")

    # Load ground truth
    gpt_gt = load_ground_truth(base_dir / "data/trajectories/gpt-5.2-codex/evaluation/eval_results.json")
    gemini_gt = load_ground_truth(base_dir / "data/trajectories/gemini-3-pro-preview/evaluation/eval_results.json")

    # Load cache results for each condition
    conditions = []

    # 1. GPT-self: GPT judge on GPT patches (from existing cache)
    gpt_on_gpt_cache = load_cache_results(
        base_dir / "cache/gpt-5.2-codex/review/gpt-5.2-codex/review_direct",
        instance_ids
    )
    gpt_self = extract_condition_results(gpt_on_gpt_cache, gpt_gt, "GPT", "GPT", "self")
    conditions.append(gpt_self)
    print(f"GPT-self: loaded {len(gpt_on_gpt_cache)} results")

    # 2. GPT-cross: GPT judge on Gemini patches (new experiment)
    gpt_on_gemini_cache = load_cache_results(
        base_dir / "cache/self_preference_ablation/gpt_on_gemini/review/gpt-5.2-codex/review_direct",
        instance_ids
    )
    gpt_cross = extract_condition_results(gpt_on_gemini_cache, gemini_gt, "GPT", "Gemini", "cross")
    conditions.append(gpt_cross)
    print(f"GPT-cross: loaded {len(gpt_on_gemini_cache)} results")

    # 3. Gemini-self: Gemini judge on Gemini patches (new experiment)
    gemini_on_gemini_cache = load_cache_results(
        base_dir / "cache/self_preference_ablation/gemini_on_gemini/review/gemini-3-pro-preview/review_direct",
        instance_ids
    )
    gemini_self = extract_condition_results(gemini_on_gemini_cache, gemini_gt, "Gemini", "Gemini", "self")
    conditions.append(gemini_self)
    print(f"Gemini-self: loaded {len(gemini_on_gemini_cache)} results")

    # 4. Gemini-cross: Gemini judge on GPT patches (new experiment)
    gemini_on_gpt_cache = load_cache_results(
        base_dir / "cache/self_preference_ablation/gemini_on_gpt/review/gemini-3-pro-preview/review_direct",
        instance_ids
    )
    gemini_cross = extract_condition_results(gemini_on_gpt_cache, gpt_gt, "Gemini", "GPT", "cross")
    conditions.append(gemini_cross)
    print(f"Gemini-cross: loaded {len(gemini_on_gpt_cache)} results")

    # Print results table
    print_results_table(conditions)

    # Compute self-preference effects
    gpt_effect = compute_self_preference_delta(gpt_self, gpt_cross)
    gemini_effect = compute_self_preference_delta(gemini_self, gemini_cross)

    print_self_preference_effects(gpt_effect, gemini_effect)

    # Save results to JSON for paper
    output = {
        "conditions": [
            {
                "judge": c.judge,
                "patch_source": c.patch_source,
                "condition_type": c.condition_type,
                "n_samples": c.n_samples,
                "mean_confidence": c.mean_confidence,
                "base_rate": c.base_rate,
                "overconfidence": c.overconfidence,
                "auroc": compute_auroc(c.predictions, c.ground_truth),
                "ece": compute_ece(c.predictions, c.ground_truth),
                "brier": compute_brier(c.predictions, c.ground_truth)
            }
            for c in conditions
        ],
        "self_preference_effects": {
            "gpt": {
                "delta_mean": gpt_effect[0],
                "delta_std": gpt_effect[1],
                "p_value": gpt_effect[2]
            },
            "gemini": {
                "delta_mean": gemini_effect[0],
                "delta_std": gemini_effect[1],
                "p_value": gemini_effect[2]
            }
        }
    }

    output_path = base_dir / "results/self_preference_ablation/analysis_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
