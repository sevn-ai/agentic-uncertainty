"""CLI argument helpers for experiment scripts.

Provides common argument definitions to reduce duplication across experiment scripts.
"""

from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ArgumentTypeError(f"Expected integer >= 1, got {value!r}") from exc
    if parsed <= 0:
        raise ArgumentTypeError(f"Expected integer >= 1, got {parsed}")
    return parsed


def add_trajectory_args(parser: ArgumentParser) -> None:
    """Add trajectory-related arguments.

    Adds:
        --traj-dir: Directory containing trajectory files
        --ground-truth: Path to eval_results.json
    """
    parser.add_argument(
        "--traj-dir",
        type=Path,
        required=True,
        help="Directory containing .traj files",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to eval_results.json with ground truth outcomes",
    )


def add_output_args(parser: ArgumentParser) -> None:
    """Add output-related arguments.

    Adds:
        --output-dir: Output directory for results
    """
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/<experiment>_YYYYMMDD_HHMMSS)",
    )


def add_sampling_args(parser: ArgumentParser) -> None:
    """Add sampling-related arguments.

    Adds:
        -n/--num-samples: Number of items to sample
        --seed: Random seed for reproducibility
    """
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=None,
        help="Number of instances to sample (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )


def add_parallel_args(parser: ArgumentParser) -> None:
    """Add parallelization arguments.

    Adds:
        -k/--parallel: Number of parallel tasks
    """
    parser.add_argument(
        "-k", "--parallel",
        type=int,
        default=1,
        help="Number of parallel tasks (default: 1)",
    )


def add_method_args(
    parser: ArgumentParser,
    choices: list[str],
    default: list[str] | None = None,
) -> None:
    """Add method selection arguments.

    Args:
        parser: ArgumentParser to add to.
        choices: List of valid method names.
        default: Default methods to use (default: ["direct"]).

    Adds:
        --methods: Methods to run
    """
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=choices,
        default=default or ["direct"],
        help=f"Methods to run (choices: {choices})",
    )


def add_standard_experiment_args(parser: ArgumentParser) -> None:
    """Add all standard experiment arguments.

    This is a convenience function that adds:
    - Output args (--output-dir)
    - Sampling args (-n, --seed)
    - Parallel args (-k)
    """
    add_output_args(parser)
    add_sampling_args(parser)
    add_parallel_args(parser)


def add_agent_selection_args(parser: ArgumentParser) -> None:
    """Add agent selection argument for unified CLI.

    Adds:
        --agents: List of agents to run (exploration, review, checkpoint, checkpoint_posthoc, mid_execution)
    """
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=["exploration", "review", "checkpoint", "checkpoint_posthoc", "mid_execution"],
        default=["exploration"],
        help="Agents to run: exploration (pre-execution), review (post-execution), "
             "checkpoint (live confidence tracking), checkpoint_posthoc (trajectory analysis), "
             "mid_execution (evaluate partial trajectory)",
    )


def add_exploration_args(parser: ArgumentParser) -> None:
    """Add exploration-specific arguments with --exploration- prefix.

    Adds:
        --exploration-methods: Exploration methods to run
        --exploration-step-limit: Max API calls for exploration
        --exploration-timeout: Timeout in seconds
        --exploration-step-timeout: Per-step timeout in seconds
    """
    parser.add_argument(
        "--exploration-methods",
        nargs="+",
        default=["exploration_direct"],
        help="Exploration methods to run (default: exploration_direct)",
    )
    parser.add_argument(
        "--exploration-step-limit",
        type=int,
        default=30,
        help="Max API calls for exploration agent (default: 30)",
    )
    parser.add_argument(
        "--exploration-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for exploration agent (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--exploration-step-timeout",
        type=int,
        default=120,
        help="Per-step timeout in seconds for individual commands (default: 120 = 2 min)",
    )


def add_review_args(parser: ArgumentParser) -> None:
    """Add review-specific arguments with --review- prefix.

    Adds:
        --review-methods: Review methods to run
        --review-step-limit: Max steps for review
        --review-cost-limit: Max cost for review
        --review-timeout: Timeout in seconds
        --review-step-timeout: Per-step timeout in seconds
        --traj-dir: Directory containing trajectory files (required for review)
    """
    parser.add_argument(
        "--review-methods",
        nargs="+",
        default=["direct"],
        help="Review methods to run (default: direct)",
    )
    parser.add_argument(
        "--review-step-limit",
        type=int,
        default=25,
        help="Max steps for review agent (default: 25)",
    )
    parser.add_argument(
        "--review-cost-limit",
        type=float,
        default=1.0,
        help="Max cost for review agent (default: 1.0)",
    )
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for review agent (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--review-step-timeout",
        type=int,
        default=120,
        help="Per-step timeout in seconds for individual commands (default: 120 = 2 min)",
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=None,
        help="Directory containing .traj files (required when --agents includes 'review')",
    )


def add_environment_args(parser: ArgumentParser) -> None:
    """Add environment and model arguments.

    Adds:
        --environment-class: Environment type (docker, singularity, modal)
        --model: Model name for agents
    """
    parser.add_argument(
        "--environment-class",
        type=str,
        default="modal",
        choices=["docker", "singularity", "modal"],
        help="Environment type for agents (default: modal)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2-codex",
        help="Model name for agents (default: gpt-5.2-codex)",
    )
    parser.add_argument(
        "--model-class",
        type=str,
        default="",
        help="Model class for mini-swe-agent (e.g. 'anthropic' for direct Anthropic API). "
             "Auto-detected from environment if not specified.",
    )


def add_cache_args(parser: ArgumentParser) -> None:
    """Add caching arguments.

    Adds:
        --cache-dir: Directory for caching per-instance results
    """
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching per-instance results. "
             "Enables incremental runs - pilot results are reused when expanding to full set.",
    )


def add_checkpoint_args(parser: ArgumentParser) -> None:
    """Add checkpoint-specific arguments with --checkpoint- prefix.

    Adds:
        --checkpoint-step-limit: Max API calls for checkpoint agent
        --checkpoint-timeout: Timeout in seconds
        --checkpoint-step-timeout: Per-step timeout in seconds
        --checkpoint-confidence-interval: Steps between confidence checkpoints
    """
    parser.add_argument(
        "--checkpoint-step-limit",
        type=int,
        default=50,
        help="Max API calls for checkpoint agent (default: 50)",
    )
    parser.add_argument(
        "--checkpoint-timeout",
        type=int,
        default=1200,
        help="Timeout in seconds for checkpoint agent (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--checkpoint-step-timeout",
        type=int,
        default=120,
        help="Per-step timeout in seconds for individual commands (default: 120 = 2 min)",
    )
    parser.add_argument(
        "--checkpoint-confidence-interval",
        type=_positive_int,
        default=5,
        help="Elicit confidence every N steps (default: 5)",
    )


def add_checkpoint_posthoc_args(parser: ArgumentParser) -> None:
    """Add checkpoint_posthoc-specific arguments with --checkpoint-posthoc- prefix.

    Adds:
        --checkpoint-posthoc-confidence-interval: Steps between confidence checkpoints
        --checkpoint-posthoc-traj-dir: Directory containing trajectory files
    """
    parser.add_argument(
        "--checkpoint-posthoc-confidence-interval",
        type=_positive_int,
        default=5,
        help="Steps between confidence checkpoints in post-hoc analysis (default: 5)",
    )
    parser.add_argument(
        "--checkpoint-posthoc-traj-dir",
        type=Path,
        default=None,
        help="Directory containing .traj files for checkpoint_posthoc analysis "
             "(falls back to --traj-dir if not specified)",
    )


def _float_fraction(value: str) -> float:
    """Parse a float in (0.0, 1.0]."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ArgumentTypeError(f"Expected float in (0.0, 1.0], got {value!r}") from exc
    if not (0.0 < parsed <= 1.0):
        raise ArgumentTypeError(f"Expected float in (0.0, 1.0], got {parsed}")
    return parsed


def add_mid_execution_args(parser: ArgumentParser) -> None:
    """Add mid_execution-specific arguments with --mid-execution- prefix.

    Adds:
        --mid-execution-step-limit: Max steps for mid-execution agent
        --mid-execution-timeout: Timeout in seconds
        --mid-execution-step-timeout: Per-step timeout in seconds
        --mid-execution-traj-dir: Directory containing trajectory files
        --progress-fraction: What fraction of trajectory to show (0.0-1.0)
    """
    parser.add_argument(
        "--mid-execution-step-limit",
        type=int,
        default=25,
        help="Max steps for mid-execution evaluation agent (default: 25)",
    )
    parser.add_argument(
        "--mid-execution-timeout",
        type=int,
        default=900,
        help="Timeout in seconds for mid-execution agent (default: 900 = 15 min)",
    )
    parser.add_argument(
        "--mid-execution-step-timeout",
        type=int,
        default=120,
        help="Per-step timeout in seconds for individual commands (default: 120 = 2 min)",
    )
    parser.add_argument(
        "--mid-execution-traj-dir",
        type=Path,
        default=None,
        help="Directory containing .traj files for mid_execution analysis "
             "(falls back to --traj-dir if not specified)",
    )
    parser.add_argument(
        "--progress-fraction",
        type=_float_fraction,
        default=0.5,
        help="Fraction of trajectory to show to evaluator (0.0-1.0, default: 0.5 = 50%%)",
    )
