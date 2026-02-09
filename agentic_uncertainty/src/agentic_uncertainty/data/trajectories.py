"""Trajectory parsing and feature extraction for SWE-agent .traj files."""

import json
import re
from pathlib import Path


def parse_trajectory(traj_path: Path | str) -> dict:
    """Parse a .traj JSON file.

    Args:
        traj_path: Path to the .traj file.

    Returns:
        Parsed trajectory dict with keys:
        - trajectory: list of steps
        - info: submission, exit_status, model_stats
        - instance_id: extracted from filename or path
    """
    traj_path = Path(traj_path)
    with open(traj_path) as f:
        data = json.load(f)

    # Extract instance_id from parent folder name (e.g., "instance_element-hq__element-web-...")
    # This matches the format used in eval_results.json ground truth
    instance_id = traj_path.parent.name
    if not instance_id or instance_id == ".":
        # Fallback to file stem if parent folder name is not useful
        instance_id = traj_path.stem

    data["instance_id"] = instance_id
    data["traj_path"] = str(traj_path)
    return data


def classify_action(action: str) -> str:
    """Classify an action into categories.

    Returns one of: edit, search, test, navigation, submit, bash, other
    """
    action_lower = action.lower()

    if action_lower.startswith("submit"):
        return "submit"
    if any(x in action_lower for x in ["edit ", "insert ", "create "]):
        return "edit"
    if any(x in action_lower for x in ["find_file", "search_dir", "search_file", "grep", "find "]):
        return "search"
    if any(x in action_lower for x in ["pytest", "python", "test", "npm test", "cargo test"]):
        return "test"
    if any(x in action_lower for x in ["open ", "goto ", "scroll_", "cd ", "ls"]):
        return "navigation"
    if any(x in action_lower for x in ["bash", "cat ", "echo ", "mkdir", "rm ", "cp ", "mv "]):
        return "bash"
    return "other"


def has_error(observation: str) -> bool:
    """Check if observation contains an error signal."""
    error_patterns = [
        r"error:",
        r"Error:",
        r"ERROR",
        r"Traceback \(most recent call last\)",
        r"Exception:",
        r"FAILED",
        r"syntax error",
        r"SyntaxError",
        r"NameError",
        r"TypeError",
        r"ValueError",
        r"ImportError",
        r"ModuleNotFoundError",
        r"FileNotFoundError",
        r"Permission denied",
        r"command not found",
    ]
    for pattern in error_patterns:
        if re.search(pattern, observation, re.IGNORECASE):
            return True
    return False


def extract_features(traj: dict, step: int) -> dict:
    """Extract features at a checkpoint.

    Args:
        traj: Parsed trajectory dict.
        step: Step number (1-indexed, will extract features up to this step).

    Returns:
        Dict of features for ML model.
    """
    steps = traj.get("trajectory", [])
    n_total = len(steps)
    steps_to_analyze = steps[:step]

    # Count action types
    action_counts = {"edit": 0, "search": 0, "test": 0, "navigation": 0, "submit": 0, "bash": 0, "other": 0}
    error_count = 0

    for s in steps_to_analyze:
        action = s.get("action", "")
        observation = s.get("observation", "")

        action_type = classify_action(action)
        action_counts[action_type] += 1

        if has_error(observation):
            error_count += 1

    n_steps = len(steps_to_analyze)

    return {
        "step": step,
        "n_steps": n_steps,
        "n_total_steps": n_total,
        "progress": step / max(n_total, 1),
        # Action counts
        "n_edits": action_counts["edit"],
        "n_searches": action_counts["search"],
        "n_tests": action_counts["test"],
        "n_navigation": action_counts["navigation"],
        "n_bash": action_counts["bash"],
        # Rates
        "edit_rate": action_counts["edit"] / max(n_steps, 1),
        "search_rate": action_counts["search"] / max(n_steps, 1),
        "test_rate": action_counts["test"] / max(n_steps, 1),
        # Errors
        "n_errors": error_count,
        "error_rate": error_count / max(n_steps, 1),
    }


def get_partial_trajectory(traj: dict, up_to_step: int, max_chars: int = 10000) -> str:
    """Format trajectory up to a step as context string for LLM.

    Args:
        traj: Parsed trajectory dict.
        up_to_step: Include steps up to this index (1-indexed).
        max_chars: Maximum characters (truncates from beginning if exceeded).

    Returns:
        Formatted string representation of the trajectory.
    """
    steps = traj.get("trajectory", [])[:up_to_step]

    lines = []
    for i, s in enumerate(steps, 1):
        action = s.get("action", "").strip()
        observation = s.get("observation", "").strip()
        thought = s.get("thought", "").strip()

        # Truncate long observations
        if len(observation) > 500:
            observation = observation[:500] + "..."

        step_text = f"Step {i}:"
        if thought:
            step_text += f"\n  Thought: {thought[:200]}..." if len(thought) > 200 else f"\n  Thought: {thought}"
        step_text += f"\n  Action: {action}"
        step_text += f"\n  Result: {observation[:300]}..." if len(observation) > 300 else f"\n  Result: {observation}"

        lines.append(step_text)

    result = "\n\n".join(lines)

    # Truncate from beginning if too long (keep recent context)
    if len(result) > max_chars:
        result = "... [earlier steps truncated] ...\n\n" + result[-max_chars:]

    return result


def get_submission(traj: dict) -> str | None:
    """Get the submission (patch) from a trajectory."""
    info = traj.get("info", {})
    return info.get("submission")


def get_cost(traj: dict) -> float:
    """Get the cost from a trajectory."""
    info = traj.get("info", {})
    model_stats = info.get("model_stats", {})
    return model_stats.get("instance_cost", 0.0)


def load_trajectories(traj_dir: Path | str, pattern: str = "**/*.traj*") -> list[dict]:
    """Load all trajectories from a directory.

    Args:
        traj_dir: Directory containing .traj files.
        pattern: Glob pattern for finding .traj files.

    Returns:
        List of parsed trajectory dicts.
    """
    traj_dir = Path(traj_dir)
    trajectories = []
    for traj_path in traj_dir.glob(pattern):
        try:
            traj = parse_trajectory(traj_path)
            trajectories.append(traj)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {traj_path}: {e}")
    return trajectories


def parse_run_index(filename: str) -> tuple[str, int]:
    """Parse instance_id and run index from trajectory filename.

    Supports both new format (instance_id_run_N.traj.json) and legacy format (instance_id.traj.json).

    Args:
        filename: Trajectory filename (stem without .traj.json suffix).

    Returns:
        Tuple of (instance_id, run_index). Returns run_index=0 for legacy format.
    """
    # Try new format: instance_id_run_N
    match = re.match(r"(.+)_run_(\d+)$", filename)
    if match:
        return match.group(1), int(match.group(2))
    # Legacy format: no run index
    return filename, 0


def load_trajectories_by_instance(
    traj_dir: Path | str,
    pattern: str = "**/*.traj*",
) -> dict[str, list[dict]]:
    """Load trajectories grouped by instance_id.

    Supports both new format (instance_id_run_N.traj.json) and legacy format.
    Trajectories for each instance are sorted by run index.

    Args:
        traj_dir: Directory containing .traj files.
        pattern: Glob pattern for finding .traj files.

    Returns:
        Dict mapping instance_id to list of trajectory dicts, sorted by run index.
        Example: {"instance_id": [traj_run_0, traj_run_1, traj_run_2]}
    """
    traj_dir = Path(traj_dir)
    by_instance: dict[str, list[tuple[int, dict]]] = {}

    for traj_path in traj_dir.glob(pattern):
        try:
            traj = parse_trajectory(traj_path)

            # Parse run index from filename
            filename = traj_path.stem
            if filename.endswith(".traj"):
                filename = filename[:-5]  # Remove .traj suffix
            instance_id, run_idx = parse_run_index(filename)

            # Override instance_id from filename parsing (more reliable for multi-run)
            traj["instance_id"] = instance_id
            traj["run_index"] = run_idx

            if instance_id not in by_instance:
                by_instance[instance_id] = []
            by_instance[instance_id].append((run_idx, traj))

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {traj_path}: {e}")

    # Sort each instance's trajectories by run index and return just the dicts
    result = {}
    for instance_id, trajs in by_instance.items():
        sorted_trajs = sorted(trajs, key=lambda x: x[0])
        result[instance_id] = [t[1] for t in sorted_trajs]

    return result


def get_ground_truth(traj: dict) -> bool:
    """Get ground truth (resolved status) from a trajectory.

    Args:
        traj: Parsed trajectory dict.

    Returns:
        True if the trajectory resulted in a successful resolution.
    """
    info = traj.get("info", {})
    exit_status = info.get("exit_status")

    # Handle different exit_status formats
    if isinstance(exit_status, bool):
        return exit_status
    if isinstance(exit_status, str):
        return exit_status.lower() in ("true", "resolved", "success", "submitted")
    return False
