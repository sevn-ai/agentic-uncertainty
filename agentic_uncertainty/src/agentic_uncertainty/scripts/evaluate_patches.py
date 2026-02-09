"""Evaluate generated patches using Modal.

This script wraps swe_bench_pro_eval.py to evaluate patches generated
by generate_trajectories.py.

Usage:
    uv run python -m agentic_uncertainty.scripts.evaluate_patches \
        --traj-dir data/trajectories/claude-sonnet-4-5/default \
        --workers 4

    # Use local Docker instead of Modal
    uv run python -m agentic_uncertainty.scripts.evaluate_patches \
        --traj-dir data/trajectories/claude-sonnet-4-5/default \
        --use-local-docker

Results are saved to: {traj-dir}/evaluation/eval_results.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Path to SWE-bench_Pro-os relative to this file
SWE_BENCH_PRO_OS = Path(__file__).parent.parent.parent.parent.parent / "SWE-bench_Pro-os"


def convert_preds_format(preds_path: Path) -> Path:
    """Convert mini-swe-agent preds.json to swe_bench_pro_eval format.

    mini-swe-agent outputs:
        {"instance_id": {"model_name_or_path": "...", "model_patch": "..."}, ...}

    swe_bench_pro_eval expects:
        [{"instance_id": "...", "patch": "..."}, ...]

    Returns path to converted file.
    """
    with open(preds_path) as f:
        preds_data = json.load(f)

    # Check if already in list format
    if isinstance(preds_data, list):
        return preds_path

    # Convert dict format to list format
    converted = []
    for instance_id, data in preds_data.items():
        converted.append({
            "instance_id": instance_id,
            "patch": data.get("model_patch", data.get("patch", "")),
            "prefix": "",
        })

    # Write to a sibling file (not temp, so it persists for debugging)
    converted_path = preds_path.parent / "preds_converted.json"
    with open(converted_path, "w") as f:
        json.dump(converted, f, indent=2)

    print(f"Converted preds.json format: {len(converted)} patches")
    return converted_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SWE-bench patches using Modal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate patches from a trajectory directory
    %(prog)s --traj-dir data/trajectories/claude-sonnet-4-5/default

    # Use more workers for faster evaluation
    %(prog)s --traj-dir data/trajectories/claude-sonnet-4-5/default --workers 10

    # Use local Docker instead of Modal
    %(prog)s --traj-dir data/trajectories/claude-sonnet-4-5/default --use-local-docker
""",
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="Directory containing preds.json from trajectory generation",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel evaluation workers (default: 4)",
    )
    parser.add_argument(
        "--use-local-docker",
        action="store_true",
        help="Use local Docker instead of Modal for evaluation",
    )
    parser.add_argument(
        "--dockerhub-username",
        default="jefzda",
        help="Docker Hub username with SWE-bench images (default: jefzda)",
    )

    args = parser.parse_args()

    # Validate paths
    preds_path = args.traj_dir / "preds.json"
    if not preds_path.exists():
        print(f"Error: preds.json not found at {preds_path}", file=sys.stderr)
        print("Run generate_trajectories.py first to generate trajectories.", file=sys.stderr)
        sys.exit(1)

    if not SWE_BENCH_PRO_OS.exists():
        print(f"Error: SWE-bench_Pro-os not found at {SWE_BENCH_PRO_OS}", file=sys.stderr)
        sys.exit(1)

    eval_script = SWE_BENCH_PRO_OS / "swe_bench_pro_eval.py"
    if not eval_script.exists():
        print(f"Error: Evaluation script not found at {eval_script}", file=sys.stderr)
        sys.exit(1)

    # Convert preds.json format if needed (mini-swe-agent dict -> eval list)
    converted_preds_path = convert_preds_format(preds_path)

    # Output directory for evaluation results
    output_dir = args.traj_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        "python", "swe_bench_pro_eval.py",
        "--raw_sample_path", "swe_bench_pro_full.csv",
        "--patch_path", str(converted_preds_path.absolute()),
        "--output_dir", str(output_dir.absolute()),
        "--scripts_dir", "run_scripts",
        "--dockerhub_username", args.dockerhub_username,
        "--num_workers", str(args.workers),
    ]

    if args.use_local_docker:
        cmd.append("--use_local_docker")

    # Print info
    print(f"Patches: {converted_preds_path}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Mode: {'Local Docker' if args.use_local_docker else 'Modal'}")
    print(f"\nRunning: {' '.join(cmd)}\n")

    # Run evaluation
    result = subprocess.run(cmd, cwd=SWE_BENCH_PRO_OS)

    if result.returncode == 0:
        eval_results = output_dir / "eval_results.json"
        if eval_results.exists():
            print(f"\nResults saved to: {eval_results}")
        else:
            print(f"\nEvaluation complete. Check {output_dir} for results.")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
