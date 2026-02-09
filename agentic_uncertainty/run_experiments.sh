#!/bin/bash
set -e

# =============================================================================
# Unified Experiment Runner — Agentic Uncertainty
# =============================================================================
#
# Reproduces ALL experiments from the paper:
#   "Agentic Uncertainty Reveals Agentic Overconfidence"
#
# Pipeline:
#   Phase 0: Setup & installation
#   Phase 1: Trajectory generation
#   Phase 2: Patch evaluation (SWE-bench Pro)
#   Phase 3: Uncertainty experiments (pre/post/adversarial/mid-execution)
#   Phase 4: Self-preference ablation (Table 3)
#   Phase 5: Evaluation & figure generation
#
# Usage:
#   ./run_experiments.sh                          # Run everything
#   ./run_experiments.sh --phase 3                # Resume from phase 3
#   ./run_experiments.sh --phase 3 --model gpt    # Phase 3+ for GPT only
#
# Prerequisites:
#   - API keys in ~/.config/mini-swe-agent/.env
#   - Modal configured (~/.modal.toml)
#   - Run install.sh first if dependencies are not installed
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AGENTIC_DIR="$REPO_ROOT/code/agentic_uncertainty"
SWE_BENCH_DIR="$REPO_ROOT/code/SWE-bench_Pro-os"
VENV_PYTHON="$SWE_BENCH_DIR/mini-swe-agent/.venv/bin/python"
TRAJ_CONFIG="$AGENTIC_DIR/configs/swebench.yaml"

INSTANCES="$AGENTIC_DIR/data/instances_final.json"
INSTANCES_FULL="$AGENTIC_DIR/data/instances_final_full.json"
SELF_PREF_INSTANCES="$AGENTIC_DIR/data/self_preference_ablation/instances_25.json"

PARALLEL=10
EVAL_WORKERS=10

# ---------------------------------------------------------------------------
# Per-model configuration
# ---------------------------------------------------------------------------
# Sets: MODEL, TRAJ_SUBDIR, TRAJ_DIR, GROUND_TRUTH, CACHE_PREFIX,
#       MC_FLAG (model-class), COST_LIMIT, TIMEOUT_FLAGS, TRAJ_WORKERS

setup_model() {
  TRAJ_SUBDIR="" MC_FLAG="" COST_LIMIT=2.5 TIMEOUT_FLAGS="" TRAJ_WORKERS=10
  case $1 in
    gpt)
      MODEL="gpt-5.2-codex"
      TRAJ_SUBDIR="gpt-5.2-codex"
      MC_FLAG="--model-class litellm_response"
      ;;
    gemini)
      MODEL="gemini-3-pro-preview"
      TRAJ_SUBDIR="gemini-3-pro-preview"
      COST_LIMIT=1.0
      TIMEOUT_FLAGS="--exploration-timeout 1800 --review-timeout 1800"
      ;;
    claude)
      MODEL="claude-opus-4-5-20251101"
      TRAJ_SUBDIR="claude-opus-4-5"
      MC_FLAG="--model-class anthropic"
      TRAJ_WORKERS=4
      ;;
  esac
  TRAJ_DIR="$AGENTIC_DIR/data/trajectories/$TRAJ_SUBDIR"
  GROUND_TRUTH="$TRAJ_DIR/evaluation/eval_results.json"
  CACHE_PREFIX="cache/$TRAJ_SUBDIR"
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

START_PHASE=0
MODEL_FILTER="all"

while [[ $# -gt 0 ]]; do
  case $1 in
    --phase)  START_PHASE="$2"; shift 2 ;;
    --model)  MODEL_FILTER="$2"; shift 2 ;;
    --help|-h)
      echo "Usage: $0 [--phase N] [--model gpt|gemini|claude|all]"
      echo "  --phase N    Start from phase N (0-5, default: 0)"
      echo "  --model M    Run only for model M (default: all)"
      exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

should_run() { [[ "$MODEL_FILTER" == "all" || "$MODEL_FILTER" == "$1" ]]; }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log_phase() { echo -e "\n=== PHASE $1: $2 ===\n  $(date)\n"; }
log_step()  { echo -e "\n--- $1 ---\n"; }

convert_preds() {
  cd "$AGENTIC_DIR/data"
  python3 -c "
import json, os
ids = set(json.load(open('instances_final.json')))
p = 'trajectories/$TRAJ_SUBDIR/preds.json'
if os.path.exists(p):
    preds = json.load(open(p))
    out = [{'instance_id': k, 'patch': v.get('model_patch', '')}
           for k, v in preds.items() if k in ids]
    json.dump(out, open('trajectories/$TRAJ_SUBDIR/preds_converted.json', 'w'), indent=2)
    print(f'Converted {len(out)} predictions for $TRAJ_SUBDIR')
else:
    print('Warning: preds.json not found for $TRAJ_SUBDIR')
"
  cd "$AGENTIC_DIR"
}

run_swebench_eval() {
  cd "$SWE_BENCH_DIR"
  "$VENV_PYTHON" swe_bench_pro_eval.py \
    --raw_sample_path swe_bench_pro_full.csv \
    --patch_path "$TRAJ_DIR/preds_converted.json" \
    --output_dir "$TRAJ_DIR/evaluation" \
    --scripts_dir run_scripts \
    --num_workers "$EVAL_WORKERS"
  cd "$AGENTIC_DIR"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

cd "$AGENTIC_DIR"

echo "================================================================="
echo "  Agentic Uncertainty — Full Experiment Reproduction"
echo "  Repo root: $REPO_ROOT | Phase: $START_PHASE | Model: $MODEL_FILTER"
echo "  Started: $(date)"
echo "================================================================="

# ===========================
# PHASE 0 — Setup
# ===========================
if [[ $START_PHASE -le 0 ]]; then
  log_phase 0 "Setup & Dependency Installation"
  bash "$AGENTIC_DIR/install.sh"
fi

# ===========================
# PHASE 1 — Trajectory Generation
# ===========================
if [[ $START_PHASE -le 1 ]]; then
  log_phase 1 "Trajectory Generation"
  for m in gpt gemini claude; do
    should_run "$m" || continue
    setup_model "$m"
    log_step "Generating $MODEL trajectories"
    cd "$SWE_BENCH_DIR/mini-swe-agent"
    uv run python -m minisweagent.run.extra.run_batch \
      --config "$TRAJ_CONFIG" \
      --output "$TRAJ_DIR" \
      --workers "$TRAJ_WORKERS" \
      --source file \
      --instances-path "$INSTANCES_FULL" \
      --no-shuffle \
      --deployment-type modal \
      --model "$MODEL" \
      $MC_FLAG
    cd "$AGENTIC_DIR"
  done
fi

# ===========================
# PHASE 2 — Patch Evaluation
# ===========================
if [[ $START_PHASE -le 2 ]]; then
  log_phase 2 "Patch Evaluation (SWE-bench Pro)"
  for m in gpt gemini claude; do
    should_run "$m" || continue
    setup_model "$m"
    log_step "Evaluating $MODEL predictions"
    convert_preds
    run_swebench_eval
  done
fi

# ===========================
# PHASE 3 — Uncertainty Experiments
# ===========================
if [[ $START_PHASE -le 3 ]]; then
  log_phase 3 "Uncertainty Experiments"

  for m in gpt gemini claude; do
    should_run "$m" || continue
    setup_model "$m"

    # --- 3a. Pre-Execution (exploration) — Table 1 ---
    log_step "$MODEL: Pre-Execution (exploration)"
    uv run run-experiment \
      --agents exploration \
      --traj-dir "$TRAJ_DIR" \
      --ground-truth "$GROUND_TRUTH" \
      --instance-ids "$INSTANCES" \
      --cache-dir "$CACHE_PREFIX/exploration_direct" \
      --environment-class modal \
      --model "$MODEL" \
      $MC_FLAG $TIMEOUT_FLAGS \
      --exploration-methods direct \
      --exploration-step-limit 100 \
      -k $PARALLEL

    # --- 3b. Post-Execution (review, direct) — Table 1 ---
    log_step "$MODEL: Post-Execution (review, direct)"
    uv run run-experiment \
      --agents review \
      --traj-dir "$TRAJ_DIR" \
      --ground-truth "$GROUND_TRUTH" \
      --instance-ids "$INSTANCES" \
      --cache-dir "$CACHE_PREFIX/review_direct" \
      --environment-class modal \
      --model "$MODEL" \
      $MC_FLAG $TIMEOUT_FLAGS \
      --review-methods direct \
      --review-step-limit 100 \
      --review-cost-limit "$COST_LIMIT" \
      -k $PARALLEL

    # --- 3c. Adversarial Post-Execution — Table 1 ---
    log_step "$MODEL: Adversarial Post-Execution"
    uv run run-experiment \
      --agents review \
      --traj-dir "$TRAJ_DIR" \
      --ground-truth "$GROUND_TRUTH" \
      --instance-ids "$INSTANCES" \
      --cache-dir "$CACHE_PREFIX/review_adversarial" \
      --environment-class modal \
      --model "$MODEL" \
      $MC_FLAG $TIMEOUT_FLAGS \
      --review-methods adversarial \
      --review-step-limit 100 \
      --review-cost-limit "$COST_LIMIT" \
      -k $PARALLEL

    # --- 3d. Mid-Execution (25%, 50%, 75%) — Table 2, Figure 4 ---
    for pct in 25 50 75; do
      frac=$(echo "scale=2; $pct/100" | bc)
      log_step "$MODEL: Mid-Execution @ ${pct}%"
      uv run run-experiment \
        --agents mid_execution \
        --traj-dir "$TRAJ_DIR" \
        --ground-truth "$GROUND_TRUTH" \
        --instance-ids "$INSTANCES" \
        --cache-dir "$CACHE_PREFIX/mid_execution/mid_${pct}" \
        --output-dir "results/$TRAJ_SUBDIR/mid_execution_${pct}" \
        --environment-class modal \
        --model "$MODEL" \
        $MC_FLAG \
        --progress-fraction "$frac" \
        --mid-execution-step-limit 25 \
        -k $PARALLEL
    done
  done
fi

# ===========================
# PHASE 4 — Self-Preference Ablation (Table 3)
# ===========================
if [[ $START_PHASE -le 4 ]]; then
  log_phase 4 "Self-Preference Ablation (Table 3)"

  run_self_pref() {
    # $1=judge_model, $2=patch_traj_dir, $3=ground_truth, $4=label
    log_step "Self-pref: $4"
    uv run run-experiment --agents review \
      --model "$1" \
      --traj-dir "$2" \
      --instance-ids "$SELF_PREF_INSTANCES" \
      --ground-truth "$3" \
      --cache-dir "cache/self_preference_ablation/$4" \
      --output-dir "results/self_preference_ablation/$4" \
      --environment-class modal \
      --review-methods direct \
      -k 5
  }

  setup_model gpt;    GPT_M="$MODEL" GPT_TD="$TRAJ_DIR" GPT_GT="$GROUND_TRUTH"
  setup_model gemini;  GEM_M="$MODEL" GEM_TD="$TRAJ_DIR" GEM_GT="$GROUND_TRUTH"

  run_self_pref "$GPT_M" "$GEM_TD" "$GEM_GT" "gpt_on_gemini"
  run_self_pref "$GEM_M" "$GEM_TD" "$GEM_GT" "gemini_on_gemini"
  run_self_pref "$GEM_M" "$GPT_TD" "$GPT_GT" "gemini_on_gpt"
fi

# ===========================
# PHASE 5 — Evaluation & Figures
# ===========================
if [[ $START_PHASE -le 5 ]]; then
  log_phase 5 "Evaluation, Tables & Figures"

  uv run evaluate-cache \
    --cache-dir cache/ \
    --ground-truth "$AGENTIC_DIR/data/trajectories/gpt-5.2-codex/evaluation/eval_results.json" \
    --output-dir results/evaluation \
    --plots --latex --compare-models

  uv run generate-paper-figures
fi

echo -e "\n=== ALL EXPERIMENTS COMPLETE — $(date) ==="
echo "Results: results/evaluation/ | Figures: paper/figures/"
