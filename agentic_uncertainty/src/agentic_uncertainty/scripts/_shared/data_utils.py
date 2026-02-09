"""Data utilities for experiment scripts.

Provides common data loading, matching, and preparation utilities.
"""

from pathlib import Path
from typing import Any


def normalize_instance_id(instance_id: str) -> str:
    """Normalize instance ID for consistent matching.

    Strips the 'instance_' prefix if present.

    Args:
        instance_id: The instance ID to normalize.

    Returns:
        Normalized instance ID without 'instance_' prefix.
    """
    if instance_id.startswith("instance_"):
        return instance_id[9:]
    return instance_id


def get_instance_id_variants(instance_id: str) -> list[str]:
    """Get all format variations of an instance ID.

    Handles the common case where instance IDs may use either '-' or '__'
    as separators (e.g., 'django-django-12345' vs 'django__django__12345').

    Args:
        instance_id: The original instance ID.

    Returns:
        List of possible ID formats to try when matching.
    """
    return [
        instance_id,
        instance_id.replace("-", "__"),
        instance_id.replace("__", "-"),
    ]


def match_instance_to_ground_truth(
    instance_id: str,
    ground_truth: dict[str, bool],
) -> tuple[str | None, bool | None]:
    """Match a single instance ID to ground truth, handling format variations.

    Args:
        instance_id: The instance ID to match.
        ground_truth: Dictionary mapping instance IDs to resolved status.

    Returns:
        Tuple of (matched_id, resolved) or (None, None) if no match found.
    """
    for iid in get_instance_id_variants(instance_id):
        if iid in ground_truth:
            return iid, ground_truth[iid]
    return None, None


def match_trajectories_to_ground_truth(
    trajectories: list[dict[str, Any]],
    ground_truth: dict[str, bool],
    instance_id_key: str = "instance_id",
) -> list[dict[str, Any]]:
    """Match trajectories to ground truth, handling ID format variations.

    Adds 'ground_truth' (bool) and 'matched_id' (str) fields to each
    successfully matched trajectory.

    Args:
        trajectories: List of trajectory dicts.
        ground_truth: Dictionary mapping instance IDs to resolved status.
        instance_id_key: Key in trajectory dict containing the instance ID.

    Returns:
        List of trajectories that were successfully matched (with added fields).
    """
    matched = []
    for traj in trajectories:
        instance_id = traj.get(instance_id_key, "")
        matched_id, resolved = match_instance_to_ground_truth(instance_id, ground_truth)

        if matched_id is not None:
            traj["ground_truth"] = resolved
            traj["matched_id"] = matched_id
            matched.append(traj)

    return matched


def sample_items(
    items: list[Any],
    num_samples: int | None,
    seed: int = 42,
) -> list[Any]:
    """Sample a random subset of items.

    Args:
        items: List of items to sample from.
        num_samples: Number of items to sample (None = return all).
        seed: Random seed for reproducibility.

    Returns:
        Sampled list of items.
    """
    if num_samples is None or num_samples >= len(items):
        return items

    import random
    rng = random.Random(seed)
    return rng.sample(items, num_samples)
