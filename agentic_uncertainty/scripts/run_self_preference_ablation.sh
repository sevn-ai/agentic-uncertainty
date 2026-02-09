#!/bin/bash
# Self-Preference Ablation Experiments
# This script runs the 3 experiments needed for the self-preference analysis:
# 1. GPT judge on Gemini patches (GPT-cross)
# 2. Gemini judge on Gemini patches (Gemini-self)
# 3. Gemini judge on GPT patches (Gemini-cross) - re-run for model consistency

set -e

cd /home/jean/ac_paper/code/agentic_uncertainty

INSTANCES="data/self_preference_ablation/instances_25.json"
GPT_TRAJ="data/trajectories/gpt-5.2-codex"
GEMINI_TRAJ="data/trajectories/gemini-3-pro-preview"
GPT_GT="data/trajectories/gpt-5.2-codex/evaluation/eval_results.json"
GEMINI_GT="data/trajectories/gemini-3-pro-preview/evaluation/eval_results.json"

echo "===== Self-Preference Ablation Experiments ====="
echo "Running 75 total evaluations (3 conditions x 25 instances)"
echo ""

# Run 1: GPT judge on Gemini patches (GPT-cross)
echo "[1/3] Running GPT judge on Gemini patches..."
uv run run-experiment --agents review \
  --model gpt-5.2-codex \
  --traj-dir "$GEMINI_TRAJ" \
  --instance-ids "$INSTANCES" \
  --ground-truth "$GEMINI_GT" \
  --cache-dir cache/self_preference_ablation/gpt_on_gemini \
  --output-dir results/self_preference_ablation/gpt_on_gemini \
  --parallel 5 \
  --environment-class modal \
  --review-methods direct

echo ""
echo "[1/3] GPT on Gemini completed!"
echo ""

# Run 2: Gemini judge on Gemini patches (Gemini-self)
echo "[2/3] Running Gemini judge on Gemini patches..."
uv run run-experiment --agents review \
  --model gemini-3-pro-preview \
  --traj-dir "$GEMINI_TRAJ" \
  --instance-ids "$INSTANCES" \
  --ground-truth "$GEMINI_GT" \
  --cache-dir cache/self_preference_ablation/gemini_on_gemini \
  --output-dir results/self_preference_ablation/gemini_on_gemini \
  --parallel 5 \
  --environment-class modal \
  --review-methods direct

echo ""
echo "[2/3] Gemini on Gemini completed!"
echo ""

# Run 3: Gemini judge on GPT patches (Gemini-cross) - for consistency with same judge model
echo "[3/3] Running Gemini judge on GPT patches..."
uv run run-experiment --agents review \
  --model gemini-3-pro-preview \
  --traj-dir "$GPT_TRAJ" \
  --instance-ids "$INSTANCES" \
  --ground-truth "$GPT_GT" \
  --cache-dir cache/self_preference_ablation/gemini_on_gpt \
  --output-dir results/self_preference_ablation/gemini_on_gpt \
  --parallel 5 \
  --environment-class modal \
  --review-methods direct

echo ""
echo "[3/3] Gemini on GPT completed!"
echo ""

echo "===== All experiments completed! ====="
echo "Run 'python scripts/analyze_self_preference.py' to analyze results."
