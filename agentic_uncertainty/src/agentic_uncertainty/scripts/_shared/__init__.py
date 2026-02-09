"""Shared utilities for experiment scripts."""

from .runner import (
    run_estimators_on_tasks,
    save_experiment_results,
    print_summary_statistics,
    get_output_dir,
    run_async_with_progress,
)
from .metrics import (
    compute_metrics,
    MetricsResult,
    compute_metrics_for_method,
    compute_standard_metrics,
    compute_false_commitment_rate,
)
from .tables import (
    # Formatters
    fmt_metric,
    fmt_signed,
    fmt_percent,
    fmt_currency,
    fmt_int,
    fmt_auroc_ci,
    # Column definitions
    Column,
    METRICS_COLUMNS,
    # Table builders
    print_metrics_table,
    print_position_table,
    print_checkpoint_table,
    print_kv_table,
    print_efficiency_table,
    print_summary_table,
)
from .data_utils import (
    normalize_instance_id,
    get_instance_id_variants,
    match_instance_to_ground_truth,
    match_trajectories_to_ground_truth,
    sample_items,
)
from .cli import (
    add_trajectory_args,
    add_output_args,
    add_sampling_args,
    add_parallel_args,
    add_method_args,
    add_standard_experiment_args,
    add_agent_selection_args,
    add_exploration_args,
    add_review_args,
    add_environment_args,
    add_cache_args,
    add_checkpoint_args,
    add_checkpoint_posthoc_args,
    add_mid_execution_args,
)
from .ground_truth import (
    GroundTruthStatus,
    GroundTruthInfo,
    get_ground_truth_status,
    ensure_ground_truth,
    get_or_create_instance_ids,
)
from .cache import ResultCache
from .config import (
    AgentRunConfig,
    ExplorationConfig,
    ReviewConfig,
    CheckpointConfig,
    CheckpointPosthocConfig,
    MidExecutionConfig,
    ExperimentConfig,
)
from .runners import (
    AgentRunner,
    BaseAgentRunner,
    AGENT_RUNNERS,
    register_agent,
)
from .shutdown import (
    is_shutdown_requested,
    request_shutdown,
    reset_shutdown,
    register_environment,
    unregister_environment,
    get_active_environment_count,
    cleanup_all_environments,
    atexit_cleanup,
)

__all__ = [
    # Runner utilities
    "run_estimators_on_tasks",
    "save_experiment_results",
    "print_summary_statistics",
    "get_output_dir",
    "run_async_with_progress",
    # Metrics
    "compute_metrics",
    "MetricsResult",
    "compute_metrics_for_method",
    "compute_standard_metrics",
    "compute_false_commitment_rate",
    # Table formatters
    "fmt_metric",
    "fmt_signed",
    "fmt_percent",
    "fmt_currency",
    "fmt_int",
    "fmt_auroc_ci",
    # Table definitions
    "Column",
    "METRICS_COLUMNS",
    # Table builders
    "print_metrics_table",
    "print_position_table",
    "print_checkpoint_table",
    "print_kv_table",
    "print_efficiency_table",
    "print_summary_table",
    # Data utilities
    "normalize_instance_id",
    "get_instance_id_variants",
    "match_instance_to_ground_truth",
    "match_trajectories_to_ground_truth",
    "sample_items",
    # CLI utilities
    "add_trajectory_args",
    "add_output_args",
    "add_sampling_args",
    "add_parallel_args",
    "add_method_args",
    "add_standard_experiment_args",
    "add_agent_selection_args",
    "add_exploration_args",
    "add_review_args",
    "add_environment_args",
    "add_cache_args",
    "add_checkpoint_args",
    "add_checkpoint_posthoc_args",
    "add_mid_execution_args",
    # Ground truth utilities
    "GroundTruthStatus",
    "GroundTruthInfo",
    "get_ground_truth_status",
    "ensure_ground_truth",
    "get_or_create_instance_ids",
    # Caching
    "ResultCache",
    # Configuration
    "AgentRunConfig",
    "ExplorationConfig",
    "ReviewConfig",
    "CheckpointConfig",
    "CheckpointPosthocConfig",
    "MidExecutionConfig",
    "ExperimentConfig",
    # Runners
    "AgentRunner",
    "BaseAgentRunner",
    "AGENT_RUNNERS",
    "register_agent",
    # Shutdown
    "is_shutdown_requested",
    "request_shutdown",
    "reset_shutdown",
    "register_environment",
    "unregister_environment",
    "get_active_environment_count",
    "cleanup_all_environments",
    "atexit_cleanup",
]
