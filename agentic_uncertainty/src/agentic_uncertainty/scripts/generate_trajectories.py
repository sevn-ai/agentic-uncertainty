"""Generate trajectories for SWE-bench Pro instances using Modal.

This script wraps mini-swe-agent to generate trajectories with proper
caching by model and agent type. By default, it also runs evaluation
to obtain ground truth (pass/fail) for each instance.

Usage:
    uv run python -m agentic_uncertainty.scripts.generate_trajectories \
        --model claude-sonnet-4-5 \
        --num-samples 50 \
        --workers 4

    # Skip evaluation (trajectories only):
    uv run python -m agentic_uncertainty.scripts.generate_trajectories \
        --model claude-sonnet-4-5 \
        --num-samples 50 \
        --no-evaluate

    # For online confidence experiments (ConfidenceAgent):
    uv run python -m agentic_uncertainty.scripts.generate_trajectories \
        --model claude-sonnet-4-5 \
        --agent-class confidence \
        --num-samples 50

Trajectories are cached at: data/trajectories/{model}/{agent_class}/
Re-run is safe - existing trajectories are skipped automatically.

After trajectory generation, evaluation produces:
    data/trajectories/{model}/evaluation/eval_results.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

# Path to SWE-bench_Pro-os relative to this file
# This file is at: agentic_uncertainty/src/agentic_uncertainty/scripts/generate_trajectories.py
# SWE-bench_Pro-os is at: agentic_uncertainty/../SWE-bench_Pro-os
SWE_BENCH_PRO_OS = Path(__file__).parent.parent.parent.parent.parent / "SWE-bench_Pro-os"

# Local dataset file with correct Docker image names
LOCAL_DATASET_DIR = Path(__file__).parent.parent.parent.parent / "data" / "swebench_pro_dataset"

# DockerHub username that hosts the SWE-bench Pro images
DOCKERHUB_USERNAME = "jefzda"


def get_dockerhub_image_uri(uid: str, repo_name: str) -> str:
    """Get Docker Hub image URI for SWE-bench Pro instance.

    Ported from SWE-bench_Pro-os/helper_code/image_uri.py
    """
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{DOCKERHUB_USERNAME}/sweap-images:{tag}"


def get_cache_dir(model: str, agent_class: str = "default") -> Path:
    """Get shared cache directory for model + agent type.

    Args:
        model: Model name (e.g., "claude-sonnet-4-5")
        agent_class: Agent class ("default" or "confidence")

    Returns:
        Path to cache directory.

    Structure (compatible with run_all.py expectations):
        data/trajectories/
        ├── claude-sonnet-4-5/              # DefaultAgent trajectories
        └── claude-sonnet-4-5-confidence/   # ConfidenceAgent trajectories
    """
    base = Path("data/trajectories")
    if agent_class == "confidence":
        return base / f"{model}-confidence"
    return base / model


def ensure_local_dataset() -> Path:
    """Create local JSONL dataset with correct Docker image names.

    SWE-bench Pro uses different Docker image naming than mini-swe-agent defaults.
    This creates a local dataset file with the correct `image_name` field.

    Returns:
        Path to the local JSONL dataset file.
    """
    LOCAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = LOCAL_DATASET_DIR / "test.jsonl"

    if jsonl_path.exists():
        print(f"Using existing local dataset: {jsonl_path}")
        return LOCAL_DATASET_DIR

    print("Downloading SWE-bench Pro and adding Docker image names...")
    dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

    with open(jsonl_path, "w") as f:
        for instance in dataset:
            # Add the correct image_name field
            instance_dict = dict(instance)
            instance_dict["image_name"] = get_dockerhub_image_uri(
                instance_dict["instance_id"],
                instance_dict["repo"],
            )
            f.write(json.dumps(instance_dict) + "\n")

    print(f"Created local dataset with {len(dataset)} instances: {jsonl_path}")
    return LOCAL_DATASET_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Generate SWE-bench Pro trajectories using Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate 50 trajectories with Foundry API (requires FOUNDRY_BASE_URL + FOUNDRY_API_KEY)
    %(prog)s --model claude-sonnet-4-5 --num-samples 50

    # Generate with direct Anthropic API (requires ANTHROPIC_API_KEY)
    %(prog)s --model anthropic/claude-sonnet-4-5-20250929 --model-class anthropic --num-samples 50

    # Generate with ConfidenceAgent for online experiments
    %(prog)s --model claude-sonnet-4-5 --agent-class confidence --num-samples 50

    # Filter to specific instances
    %(prog)s --model claude-sonnet-4-5 --filter "django.*"
""",
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name (e.g., claude-sonnet-4-5, gpt-4o, anthropic/claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--model-class",
        default="litellm",
        choices=["foundry", "anthropic", "litellm", "openrouter"],
        help="Model backend (default: litellm). Use 'anthropic' for direct Anthropic API, 'foundry' for Azure Foundry.",
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        help="Number of instances to process (default: all)",
    )
    parser.add_argument(
        "--agent-class",
        default="default",
        choices=["default", "confidence"],
        help="Agent class: 'default' for standard, 'confidence' for online experiments (default: default)",
    )
    # Note: Only Modal is supported. Local Docker is not scalable for batch runs.
    # The --environment-class argument is kept for compatibility but only 'modal' is allowed.
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--filter",
        help="Regex filter for instance IDs (e.g., 'django.*')",
    )
    parser.add_argument(
        "--redo-existing",
        action="store_true",
        help="Regenerate trajectories even if they exist in cache",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle instances before processing",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        dest="evaluate",
        help="Run evaluation after trajectory generation to get ground truth (default: True)",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_false",
        dest="evaluate",
        help="Skip evaluation after trajectory generation",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=4,
        help="Number of parallel workers for evaluation (default: 4)",
    )

    args = parser.parse_args()

    # Validate SWE-bench_Pro-os exists
    mini_swe_agent_dir = SWE_BENCH_PRO_OS / "mini-swe-agent"
    if not mini_swe_agent_dir.exists():
        print(f"Error: mini-swe-agent not found at {mini_swe_agent_dir}", file=sys.stderr)
        print("Make sure SWE-bench_Pro-os is checked out as a sibling directory.", file=sys.stderr)
        sys.exit(1)

    # Ensure local dataset exists with correct Docker image names
    local_dataset_dir = ensure_local_dataset()

    # Set up cache directory
    cache_dir = get_cache_dir(args.model, args.agent_class)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    # Use local JSONL dataset (not HuggingFace) to get correct Docker image names
    # The "json" loader from HuggingFace datasets loads from local JSONL files
    # Use uv run to ensure we're using the mini-swe-agent venv with all dependencies
    # Always use Modal for scalability (local Docker is not supported)
    cmd = [
        "uv", "run", "python", "-m", "minisweagent.run.extra.swebench",
        "--subset", str(local_dataset_dir),  # HuggingFace loads from dir with train/test.jsonl
        "--split", "test",
        "--output", str(cache_dir.absolute()),
        "--model", args.model,
        "--model-class", args.model_class,  # foundry, anthropic, litellm, openrouter
        "--agent-class", args.agent_class,
        "--environment-class", "modal",  # Always use Modal (local Docker not scalable)
        "--workers", str(args.workers),
    ]

    # Always use a shared trajectory-generation config.
    trajectory_config = Path(__file__).parent.parent.parent.parent / "configs" / "swebench.yaml"
    if not trajectory_config.exists():
        print(f"Error: config file not found: {trajectory_config}", file=sys.stderr)
        sys.exit(1)
    cmd.extend(["--config", str(trajectory_config)])
    print(f"Using trajectory config: {trajectory_config}")

    if args.num_samples:
        cmd.extend(["--slice", f":{args.num_samples}"])
    if args.filter:
        cmd.extend(["--filter", args.filter])
    if args.redo_existing:
        cmd.append("--redo-existing")
    if args.shuffle:
        cmd.append("--shuffle")

    # Print info
    print(f"\nModel: {args.model}")
    print(f"Model class: {args.model_class}")
    print(f"Agent class: {args.agent_class}")
    print(f"Environment: modal (local Docker not supported)")
    print(f"Workers: {args.workers}")
    print(f"Dataset: {local_dataset_dir}")
    print(f"Output dir: {cache_dir}")
    print(f"Evaluate: {args.evaluate}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run mini-swe-agent
    result = subprocess.run(cmd, cwd=mini_swe_agent_dir)

    if result.returncode != 0:
        print(f"\nTrajectory generation failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    # Run evaluation if requested
    if args.evaluate:
        print("\n" + "=" * 60)
        print("Running evaluation to obtain ground truth...")
        print("=" * 60 + "\n")

        eval_result = run_evaluation(cache_dir, args.eval_workers)
        if eval_result != 0:
            print(f"\nEvaluation failed with exit code {eval_result}")
            sys.exit(eval_result)

        print("\n" + "=" * 60)
        print(f"Ground truth saved to: {cache_dir}/evaluation/eval_results.json")
        print("=" * 60)

    sys.exit(0)


def run_evaluation(traj_dir: Path, workers: int = 4) -> int:
    """Run patch evaluation using Modal.

    Args:
        traj_dir: Directory containing preds.json from trajectory generation.
        workers: Number of parallel evaluation workers.

    Returns:
        Exit code (0 for success).
    """
    # Check if preds.json exists
    preds_path = traj_dir / "preds.json"
    if not preds_path.exists():
        print(f"Warning: No preds.json found at {preds_path}, skipping evaluation")
        return 0

    # Run evaluate_patches.py
    cmd = [
        sys.executable,
        "-m",
        "agentic_uncertainty.scripts.evaluate_patches",
        "--traj-dir",
        str(traj_dir),
        "--workers",
        str(workers),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    main()
