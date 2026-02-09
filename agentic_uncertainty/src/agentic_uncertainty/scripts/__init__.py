"""Scripts for running agentic uncertainty experiments.

Structure:
    _shared/            Internal utilities (runner, metrics)
    experiments/        Experiment entry points
        run             Unified runner (exploration/review/mid_execution)
        online          Online confidence analysis
    analysis/           Post-hoc analysis scripts
        analyze_mid_execution
        false_commitment
        generate_tables
    generate_trajectories.py  SWE-bench Pro trajectory generation (Modal)
    evaluate_patches.py       Patch evaluation wrapper

Entry points (install with `uv pip install -e .`):
    run-experiment          Unified experiment runner
    run-online              Online confidence analysis
    analyze-mid-execution   Analyze mid-execution checkpoints
    generate-trajectories   Generate trajectories on SWE-bench Pro
    evaluate-patches        Evaluate generated patches
    generate-tables         Generate LaTeX tables from results
    generate-paper-figures  Generate publication figures
    analyze-false-commitment Analyze high-confidence failures
"""
