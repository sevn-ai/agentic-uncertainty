"""Ground truth auto-detection and generation utilities.

Handles the workflow of:
1. Detecting if trajectories exist for a model
2. Detecting if evaluation results exist
3. Orchestrating trajectory generation and evaluation with user confirmation
"""

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()


class GroundTruthStatus(Enum):
    """Status of ground truth availability for a model."""

    READY = "ready"  # eval_results.json exists and is valid
    NEEDS_EVALUATION = "needs_evaluation"  # trajectories exist but not evaluated
    NEEDS_TRAJECTORIES = "needs_trajectories"  # no trajectories at all


@dataclass
class GroundTruthInfo:
    """Information about ground truth status for a model."""

    status: GroundTruthStatus
    traj_dir: Path | None
    ground_truth_path: Path | None
    num_trajectories: int = 0
    num_evaluated: int = 0


def normalize_model_name(name: str) -> str:
    """Normalize model name for directory matching.

    Handles variations like:
    - claude-sonnet-4-5 vs sonnet-4.5 vs sonnet_4_5
    """
    return name.lower().replace(".", "-").replace("_", "-")


def find_trajectory_dir(model_name: str, base_dir: Path) -> Path | None:
    """Find trajectory directory for a model.

    Handles various naming patterns:
    - Exact match: {base_dir}/{model_name}/
    - Timestamped: {base_dir}/{model_name}-{timestamp}/ (latest)
    - Normalized: claude-sonnet-4-5 matches sonnet-4.5

    Args:
        model_name: Model name to find.
        base_dir: Base directory for trajectories.

    Returns:
        Path to trajectory directory, or None if not found.
    """
    if not base_dir.exists():
        return None

    # Try exact match first
    exact = base_dir / model_name
    if exact.exists() and exact.is_dir():
        return exact

    # Try with common normalizations
    normalized = normalize_model_name(model_name)

    # Collect candidates
    candidates = []
    for candidate in base_dir.iterdir():
        if not candidate.is_dir():
            continue

        # Check normalized match
        if normalize_model_name(candidate.name) == normalized:
            candidates.append(candidate)
        # Check for timestamped versions
        elif candidate.name.startswith(f"{model_name}-"):
            candidates.append(candidate)

    if candidates:
        # Sort by name descending to get latest timestamp
        return sorted(candidates, key=lambda d: d.name, reverse=True)[0]

    return None


def count_trajectories(traj_dir: Path) -> int:
    """Count trajectory files in a directory."""
    return len(list(traj_dir.glob("**/*.traj.json")))


def count_evaluated(eval_results_path: Path) -> int:
    """Count evaluated instances in eval_results.json."""
    try:
        with open(eval_results_path) as f:
            return len(json.load(f))
    except (json.JSONDecodeError, IOError):
        return 0


def get_ground_truth_status(
    model_name: str,
    traj_base_dir: Path = Path("data/trajectories"),
) -> GroundTruthInfo:
    """Check ground truth status for a model.

    Looks for trajectory directory matching the model name and checks
    if evaluation results exist.

    Args:
        model_name: Model identifier (e.g., "claude-sonnet-4-5")
        traj_base_dir: Base directory for trajectories

    Returns:
        GroundTruthInfo with status and paths
    """
    # Find trajectory directory
    traj_dir = find_trajectory_dir(model_name, traj_base_dir)

    if traj_dir is None:
        return GroundTruthInfo(
            status=GroundTruthStatus.NEEDS_TRAJECTORIES,
            traj_dir=None,
            ground_truth_path=None,
        )

    # Check for preds.json (indicates trajectories exist)
    preds_path = traj_dir / "preds.json"
    if not preds_path.exists():
        # Check if there are any trajectory files directly
        num_trajs = count_trajectories(traj_dir)
        if num_trajs == 0:
            return GroundTruthInfo(
                status=GroundTruthStatus.NEEDS_TRAJECTORIES,
                traj_dir=traj_dir,
                ground_truth_path=None,
            )

    # Count trajectories
    num_trajectories = count_trajectories(traj_dir)

    # Check for eval_results.json
    eval_results_path = traj_dir / "evaluation" / "eval_results.json"
    if eval_results_path.exists():
        num_evaluated = count_evaluated(eval_results_path)
        return GroundTruthInfo(
            status=GroundTruthStatus.READY,
            traj_dir=traj_dir,
            ground_truth_path=eval_results_path,
            num_trajectories=num_trajectories,
            num_evaluated=num_evaluated,
        )

    return GroundTruthInfo(
        status=GroundTruthStatus.NEEDS_EVALUATION,
        traj_dir=traj_dir,
        ground_truth_path=None,
        num_trajectories=num_trajectories,
    )


async def run_with_retries(
    cmd: list[str],
    description: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run subprocess with retries on failure.

    Args:
        cmd: Command to run.
        description: Description for logging.
        max_retries: Maximum number of retry attempts.
        retry_delay: Initial delay between retries (doubles each time).
        cwd: Working directory for the command.

    Returns:
        CompletedProcess on success.

    Raises:
        subprocess.CalledProcessError: If all retries exhausted.
    """
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                cmd,
                check=True,
                cwd=cwd,
            )
            return result
        except subprocess.CalledProcessError as e:
            if attempt < max_retries - 1:
                console.print(
                    f"[yellow]{description} failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {retry_delay:.0f}s...[/yellow]"
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                console.print(f"[red]{description} failed after {max_retries} attempts[/red]")
                raise


def create_filtered_dataset(instance_ids: list[str], output_dir: Path) -> Path:
    """Create a filtered JSONL dataset containing only specified instance IDs.

    Args:
        instance_ids: List of instance IDs to include.
        output_dir: Directory to save the filtered dataset.

    Returns:
        Path to the directory containing test.jsonl.
    """
    from agentic_uncertainty.scripts.generate_trajectories import (
        ensure_local_dataset,
        LOCAL_DATASET_DIR,
    )

    # Ensure full dataset exists
    ensure_local_dataset()

    # Create filtered dataset directory
    filtered_dir = output_dir / "filtered_dataset"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    filtered_jsonl = filtered_dir / "test.jsonl"

    # Load full dataset and filter
    instance_id_set = set(instance_ids)
    count = 0
    with open(LOCAL_DATASET_DIR / "test.jsonl") as f_in:
        with open(filtered_jsonl, "w") as f_out:
            for line in f_in:
                instance = json.loads(line)
                if instance["instance_id"] in instance_id_set:
                    f_out.write(line)
                    count += 1

    console.print(f"[dim]Created filtered dataset with {count} instances[/dim]")
    return filtered_dir


def is_reasoning_model(model_id: str) -> bool:
    """Check if a model is a reasoning model that doesn't support temperature."""
    reasoning_patterns = ["o1", "o3", "codex", "reasoning"]
    model_lower = model_id.lower()
    return any(pattern in model_lower for pattern in reasoning_patterns)


async def run_trajectory_generation(
    model_id: str,
    instance_ids: list[str],
    traj_base_dir: Path,
    workers: int = 4,
    model_class: str = "litellm",
    config_path: Path | None = None,
) -> None:
    """Run trajectory generation via subprocess.

    Args:
        model_id: Model ID for API calls.
        instance_ids: List of instance IDs to generate trajectories for.
        traj_base_dir: Base directory for trajectory storage.
        workers: Number of parallel workers.
        model_class: Model backend class (litellm, anthropic, foundry, openrouter).
        config_path: Optional path to custom config file.
    """
    # Create a filtered dataset containing only the requested instances
    filtered_dataset_dir = create_filtered_dataset(instance_ids, traj_base_dir)

    # Get the output directory for this model
    output_dir = traj_base_dir / model_id

    # Run from mini-swe-agent directory with uv to ensure correct venv
    # Path: ground_truth.py -> _shared -> scripts -> agentic_uncertainty -> src -> agentic_uncertainty -> code -> SWE-bench_Pro-os
    mini_swe_agent_dir = Path(__file__).parent.parent.parent.parent.parent.parent / "SWE-bench_Pro-os" / "mini-swe-agent"

    # Default to shared trajectory-generation config for all models.
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent.parent.parent / "configs" / "swebench.yaml"

    cmd = [
        "uv", "run", "python",
        "-m",
        "minisweagent.run.extra.swebench",
        "--subset",
        str(filtered_dataset_dir.absolute()),  # Must be absolute since running from different cwd
        "--split",
        "test",
        "--output",
        str(output_dir.absolute()),
        "--model",
        model_id,
        "--model-class",
        model_class,
        "--environment-class",
        "modal",
        "--workers",
        str(workers),
    ]

    # Add custom config if specified
    if config_path and config_path.exists():
        cmd.extend(["--config", str(config_path.absolute())])
        console.print(f"[dim]Using custom config: {config_path.name}[/dim]")

    await run_with_retries(cmd, "Trajectory generation", cwd=mini_swe_agent_dir)


async def run_evaluation(traj_dir: Path, workers: int = 4) -> None:
    """Run patch evaluation via subprocess.

    Args:
        traj_dir: Directory containing trajectories.
        workers: Number of parallel workers.
    """
    cmd = [
        sys.executable,
        "-m",
        "agentic_uncertainty.scripts.evaluate_patches",
        "--traj-dir",
        str(traj_dir),
        "--workers",
        str(workers),
    ]

    await run_with_retries(cmd, "SWE-bench evaluation")


def get_or_create_instance_ids(
    num_samples: int,
    seed: int,
    model_name: str,
    output_dir: Path = Path("data/instance_ids"),
) -> list[str]:
    """Get cached instance IDs or sample new ones.

    Args:
        num_samples: Number of instances to sample.
        seed: Random seed for sampling.
        model_name: Model name for cache key.
        output_dir: Directory to store cached instance IDs.

    Returns:
        List of instance IDs.
    """
    from agentic_uncertainty.data import SWEBenchProLoader

    cache_path = output_dir / f"{model_name}-seed{seed}-n{num_samples}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Sample from SWE-bench Pro
    loader = SWEBenchProLoader()
    tasks = loader.sample(n=num_samples, seed=seed)
    instance_ids = [t.instance_id for t in tasks]

    # Cache for reuse
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(instance_ids, f, indent=2)

    return instance_ids


async def ensure_ground_truth(
    model_name: str,
    model_id: str,
    num_samples: int | None = None,
    traj_base_dir: Path = Path("data/trajectories"),
    seed: int = 42,
    workers: int = 4,
    interactive: bool = True,
    model_class: str = "litellm",
) -> GroundTruthInfo | None:
    """Ensure ground truth exists, generating if needed with user confirmation.

    This is the main entry point for auto-detection flow. Uses a single prompt
    for user confirmation, then runs all necessary steps automatically.

    Args:
        model_name: Display name (e.g., "sonnet-4.5").
        model_id: Model ID for API calls (e.g., "claude-sonnet-4-5").
        num_samples: Number of samples to generate (if generating).
        traj_base_dir: Base directory for trajectories.
        seed: Random seed for sampling.
        workers: Parallel workers for generation.
        interactive: If False, skip generation steps.
        model_class: Model backend class for trajectory generation (litellm, anthropic, foundry).

    Returns:
        GroundTruthInfo with READY status, or None if user declines.
    """
    # Check current status
    info = get_ground_truth_status(model_id, traj_base_dir)

    if info.status == GroundTruthStatus.READY:
        console.print(
            f"[green]\u2713 Ground truth ready:[/green] {info.ground_truth_path}"
        )
        return info

    # Determine what needs to be done
    needs_trajectories = info.status == GroundTruthStatus.NEEDS_TRAJECTORIES
    needs_evaluation = info.status in (
        GroundTruthStatus.NEEDS_TRAJECTORIES,
        GroundTruthStatus.NEEDS_EVALUATION,
    )

    if not interactive:
        console.print(
            "[dim]Ground truth not found. Skipping (non-interactive mode).[/dim]"
        )
        return info

    if num_samples is None and needs_trajectories:
        console.print(
            "[yellow]Ground truth not found and num_samples not specified.[/yellow]"
        )
        console.print("[dim]Cannot generate trajectories. Metrics will not be computed.[/dim]")
        return info

    # Build message for single prompt
    steps = []
    if needs_trajectories:
        steps.append(f"Generate trajectories ({num_samples} instances)")
    if needs_evaluation:
        steps.append("Run SWE-bench evaluation")

    message = f"[yellow]Ground truth not found for {model_name}[/yellow]\n\n"
    message += "To compute metrics, need to:\n"
    for step in steps:
        message += f"  \u2022 {step}\n"

    console.print(Panel(message.strip(), title="Ground Truth Required"))

    if not Confirm.ask("Proceed?", default=False):
        console.print("[dim]Skipping. Metrics will not be computed.[/dim]")
        return info

    # Run generation steps
    if needs_trajectories:
        console.print("\n[dim]Generating trajectories...[/dim]")
        instance_ids = get_or_create_instance_ids(
            num_samples=num_samples,
            seed=seed,
            model_name=model_id,
        )
        await run_trajectory_generation(
            model_id=model_id,
            instance_ids=instance_ids,
            traj_base_dir=traj_base_dir,
            workers=workers,
            model_class=model_class,
        )
        # Refresh status after generation
        info = get_ground_truth_status(model_id, traj_base_dir)

    if info.status == GroundTruthStatus.NEEDS_EVALUATION and info.traj_dir:
        console.print("[dim]Running SWE-bench evaluation...[/dim]")
        await run_evaluation(info.traj_dir, workers=workers)
        # Refresh status after evaluation
        info = get_ground_truth_status(model_id, traj_base_dir)

    if info.status == GroundTruthStatus.READY:
        console.print(f"[green]\u2713 Ground truth ready[/green]")
        return info
    else:
        console.print("[red]Failed to generate ground truth[/red]")
        return info
