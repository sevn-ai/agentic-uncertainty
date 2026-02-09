"""Simple per-instance result caching.

Cache structure:
    cache_dir/{model}/{method}/{hash}.json

This enables incremental runs - run on 30 instances first,
then expand to 100 without re-running the first 30.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console


def _hash_id(instance_id: str) -> str:
    """Create a short hash of instance_id for filesystem-safe filenames."""
    return hashlib.sha256(instance_id.encode()).hexdigest()[:16]


class ResultCache:
    """Cache for per-instance experiment results."""

    def __init__(self, cache_dir: Path | str, model_name: str, console: Console | None = None):
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.console = console or Console()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, method: str, instance_id: str) -> Path:
        """Get cache file path: cache_dir/model/method/hash.json"""
        method_dir = self.cache_dir / self.model_name / method
        method_dir.mkdir(parents=True, exist_ok=True)
        return method_dir / f"{_hash_id(instance_id)}.json"

    def get(self, method: str, instance_id: str) -> dict | None:
        """Get cached result, or None if not found/invalid."""
        path = self._get_path(method, instance_id)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            # Verify instance_id matches (handles hash collisions)
            if data.get("instance_id") != instance_id:
                return None
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    def set(self, method: str, instance_id: str, prediction: float | None, raw_response: str, metadata: dict | None = None):
        """Cache a result. Prediction can be None for failed/filtered results."""
        path = self._get_path(method, instance_id)
        data = {
            "instance_id": instance_id,
            "method": method,
            "model": self.model_name,
            "prediction": prediction,
            "raw_response": raw_response,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_cached_instance_ids(self, method: str) -> set[str]:
        """Get set of instance IDs that have cached results for a method."""
        method_dir = self.cache_dir / self.model_name / method
        if not method_dir.exists():
            return set()
        cached_ids = set()
        for cache_file in method_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                cached_ids.add(data.get("instance_id", ""))
            except (json.JSONDecodeError, KeyError):
                continue
        return cached_ids

    def save_trajectory(
        self,
        method: str,
        instance_id: str,
        messages: list[dict] | None,
        exit_status: str,
        cost: float,
        n_steps: int,
        prediction: float | None = None,
        extra_info: dict | None = None,
    ) -> Path | None:
        """Save trajectory in mini-swe-agent inspector-compatible format.

        Creates a .traj.json file that can be viewed with `mini-e i <path>`.

        Args:
            method: Elicitation method name.
            instance_id: The instance ID.
            messages: Full message history from the agent.
            exit_status: Agent exit status (e.g., "Submitted", "LimitsExceeded").
            cost: Total cost in $.
            n_steps: Number of agent steps.
            prediction: Confidence prediction (0-1), if available.
            extra_info: Additional info to include in the trajectory.

        Returns:
            Path to the saved trajectory file, or None if no messages available.
        """
        if not messages:
            return None

        traj_dir = self.cache_dir / self.model_name / method / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        traj_path = traj_dir / f"{_hash_id(instance_id)}.traj.json"

        # Build inspector-compatible format
        info = {
            "exit_status": exit_status,
            "model_stats": {
                "instance_cost": cost,
                "api_calls": n_steps,
            },
            "instance_id": instance_id,
        }
        if prediction is not None:
            info["prediction"] = prediction
        if extra_info:
            info.update(extra_info)

        trajectory = {
            "info": info,
            "messages": messages,
        }

        with open(traj_path, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        return traj_path

    def get_trajectory_path(self, method: str, instance_id: str) -> Path | None:
        """Get path to trajectory file if it exists."""
        traj_path = (
            self.cache_dir
            / self.model_name
            / method
            / "trajectories"
            / f"{_hash_id(instance_id)}.traj.json"
        )
        return traj_path if traj_path.exists() else None

    # -------------------------------------------------------------------------
    # Checkpoint methods for mid-run state preservation
    # -------------------------------------------------------------------------

    def _get_checkpoint_path(self, method: str, instance_id: str) -> Path:
        """Get checkpoint file path: cache_dir/model/method/checkpoints/hash.checkpoint.json"""
        checkpoint_dir = self.cache_dir / self.model_name / method / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir / f"{_hash_id(instance_id)}.checkpoint.json"

    def save_checkpoint(
        self,
        method: str,
        instance_id: str,
        messages: list[dict] | None,
        exploration_history: list[dict],
        n_steps: int,
        cost: float,
        extra_state: dict | None = None,
    ) -> Path:
        """Save agent checkpoint for potential resume.

        Called periodically during agent execution or on timeout/error
        to preserve work done so far.

        Args:
            method: Elicitation method name.
            instance_id: The instance ID.
            messages: Full message history from the agent.
            exploration_history: List of command/output pairs.
            n_steps: Number of steps completed.
            cost: Cost accumulated so far.
            extra_state: Additional state to preserve (e.g., partial confidence).

        Returns:
            Path to the saved checkpoint file.
        """
        checkpoint_path = self._get_checkpoint_path(method, instance_id)
        checkpoint = {
            "instance_id": instance_id,
            "method": method,
            "model": self.model_name,
            "messages": messages or [],
            "exploration_history": exploration_history,
            "n_steps": n_steps,
            "cost": cost,
            "extra_state": extra_state or {},
            "timestamp": datetime.now().isoformat(),
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        return checkpoint_path

    def load_checkpoint(self, method: str, instance_id: str) -> dict | None:
        """Load checkpoint if it exists and is valid.

        Returns:
            Checkpoint dict with messages, history, etc., or None if not found.
        """
        checkpoint_path = self._get_checkpoint_path(method, instance_id)
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path) as f:
                data = json.load(f)
            # Verify instance_id matches
            if data.get("instance_id") != instance_id:
                return None
            return data
        except (json.JSONDecodeError, KeyError):
            return None

    def delete_checkpoint(self, method: str, instance_id: str) -> bool:
        """Delete checkpoint after successful completion.

        Returns:
            True if checkpoint was deleted, False if it didn't exist.
        """
        checkpoint_path = self._get_checkpoint_path(method, instance_id)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            return True
        return False

    def has_checkpoint(self, method: str, instance_id: str) -> bool:
        """Check if a checkpoint exists for this instance."""
        return self._get_checkpoint_path(method, instance_id).exists()
